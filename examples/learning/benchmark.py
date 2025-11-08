from dataclasses import dataclass
from typing import Any

import click
import jax
import jax.numpy as jnp
import jax.random as jrd
from algorithms import q_learning
from flax import struct
from policies import epsilon_greedy, soft_policy
from utils import log_results

from jaxdp import async_sample_step_pi
from jaxdp.base import bellman_optimality_operator as bellman_op
from jaxdp.mdp import MDP
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.mdp.grid_world import grid_world
from jaxdp.mdp.simple_graph import graph_mdp

jax.config.update("jax_enable_x64", True)


@struct.dataclass
class LoopState:
    """Training loop state - manages MDP interaction and episode tracking"""
    alg_state: Any  # q_learning.State
    policy_state: Any  # epsilon_greedy.State or softmax.State, etc.
    mdp_state: jnp.ndarray
    ep_step: jnp.ndarray
    ep_return: jnp.ndarray
    last_return: jnp.ndarray


@struct.dataclass
class Metrics:
    l1: jnp.ndarray
    l2: jnp.ndarray
    linf: jnp.ndarray
    bellman_err: jnp.ndarray
    iteration: jnp.ndarray
    ep_return: jnp.ndarray
    ep_len: jnp.ndarray


def compute_metrics(prev: LoopState, new: LoopState, mdp: MDP, step: int):
    """ Compute metrics for the current iteration """
    prev_alg = prev.alg_state
    new_alg = new.alg_state

    diff = new_alg.q_vals - prev_alg.q_vals
    l1 = jnp.sum(jnp.abs(diff))
    l2 = jnp.sqrt(jnp.sum(diff**2))
    linf = jnp.max(jnp.abs(diff))

    bellman_target = bellman_op.q(mdp, prev_alg.q_vals, prev_alg.gamma)
    bellman_err = jnp.max(jnp.abs(prev_alg.q_vals - bellman_target))

    # Episode completed when ep_step resets to 0
    ep_done = (new.ep_step == 0) & (prev.ep_step > 0)
    ep_return = jnp.where(ep_done, new.last_return, 0.0)
    ep_len = jnp.where(ep_done, prev.ep_step, 0.0)

    return Metrics(
        l1=l1,
        l2=l2,
        linf=linf,
        bellman_err=bellman_err,
        iteration=step,
        ep_return=ep_return,
        ep_len=ep_len
    )


@dataclass(frozen=True)
class LoopArgs:
    seed: int
    n_steps: int
    max_ep_len: int = 50
    n_envs: int = 1  # Number of parallel environments


from jaxdp.typehints import StaticMeta


class loop(metaclass=StaticMeta):
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Training Loop for Value-based RL Algorithms

    Provides init(), train(), and evaluate() functions for managing the
    training loop state and running value-based RL algorithms with various
    exploration policies.
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    def init(value_fn, policy_ns, mdp: MDP, args: LoopArgs,
             alg_params: dict, policy_params: dict) -> LoopState:
        """Initialize loop state with algorithm and policy parameters

        Args:
            value_fn: Algorithm namespace (e.g., q_learning)
            policy_ns: Policy namespace (e.g., epsilon_greedy)
            mdp: MDP environment
            args: Loop arguments (seed, n_steps, etc.)
            alg_params: Parameters for algorithm init (gamma, alpha, etc.)
            policy_params: Parameters for policy init (epsilon, etc.)
        """
        key = jrd.PRNGKey(args.seed)
        key, init_key = jrd.split(key)

        # Initialize algorithm state
        alg_state = value_fn.init(mdp, init_key, **alg_params)

        # Initialize policy state
        policy_state = policy_ns.init(**policy_params)

        # Initialize environment state(s) - always use vmap
        env_keys = jrd.split(key, args.n_envs)
        mdp_state = jax.vmap(mdp.init_state)(env_keys)
        ep_step = jnp.zeros(args.n_envs)
        ep_return = jnp.zeros(args.n_envs)
        last_return = jnp.zeros(args.n_envs)

        return LoopState(
            alg_state=alg_state,
            policy_state=policy_state,
            mdp_state=mdp_state,
            ep_step=ep_step,
            ep_return=ep_return,
            last_return=last_return
        )

    def train(value_fn, policy_ns, mdp: MDP, state: LoopState,
              args: LoopArgs, metrics_fn) -> tuple[LoopState, Any]:
        """Run training loop for n_steps with specified value function and policy

        Args:
            value_fn: Algorithm namespace (e.g., q_learning)
            policy_ns: Policy namespace (e.g., epsilon_greedy)
            mdp: MDP environment
            state: Initial loop state
            args: Loop arguments (n_steps, max_ep_len, n_envs)
            metrics_fn: Function to compute metrics at each step

        Returns:
            Final loop state and all metrics collected during training

        Note: Uses vmap to handle n_envs environments in parallel at each step.
        """
        def step_fn(state: LoopState, step_and_keys):
            step, keys = step_and_keys  # keys shape: [n_envs, 2]
            prev = state

            # Sample from all environments in parallel
            policy = policy_ns.get_policy(state.alg_state.q_vals, state.policy_state)

            # Vmap over environments
            vmap_sample = jax.vmap(
                lambda s, ep, k: async_sample_step_pi(mdp, policy, s, ep, args.max_ep_len, k),
                in_axes=(0, 0, 0)
            )
            actions, next_states, rewards, terms, timeouts, stepped_states, ep_steps = vmap_sample(
                state.mdp_state, state.ep_step, keys
            )

            # Update Q-values for each transition in parallel, then average
            vmap_update = jax.vmap(value_fn.update, in_axes=(None, 0, 0, 0, 0, 0))
            updated_states = vmap_update(
                state.alg_state, state.mdp_state, actions, next_states, rewards, terms
            )
            # Average Q-values across all parallel updates
            avg_q_vals = jnp.mean(updated_states.q_vals, axis=0)
            new_alg_state = state.alg_state.replace(q_vals=avg_q_vals)

            # Update policy state (use any episode completion signal)
            dones = terms + timeouts > 0
            any_done = jnp.any(dones)
            new_policy_state = policy_ns.update(state.policy_state, any_done)

            # Update loop state (track all environments)
            new_returns = state.ep_return + rewards
            last_returns = jnp.where(dones, new_returns, state.last_return)
            ep_returns = jnp.where(dones, 0.0, new_returns)

            new_state = LoopState(
                alg_state=new_alg_state,
                policy_state=new_policy_state,
                mdp_state=stepped_states,
                ep_step=ep_steps,
                ep_return=ep_returns,
                last_return=last_returns
            )

            metrics = metrics_fn(prev, new_state, mdp, step)
            return new_state, metrics

        keys = jrd.split(jrd.PRNGKey(args.seed), args.n_steps * args.n_envs)
        keys = keys.reshape(args.n_steps, args.n_envs, -1)
        steps_and_keys = (jnp.arange(args.n_steps), keys)

        final_state, all_metrics = jax.lax.scan(step_fn, state, steps_and_keys)
        return final_state, all_metrics

    def evaluate(value_fn, policy_ns, mdp: MDP, state: LoopState,
                 n_episodes: int = 10, max_ep_len: int = 50, seed: int = 42) -> dict:
        """Evaluate the learned policy (greedy, no exploration)

        Args:
            value_fn: Algorithm namespace (not used in evaluation, but kept for consistency)
            policy_ns: Policy namespace (not used in evaluation)
            mdp: MDP environment
            state: Current loop state with learned Q-values
            n_episodes: Number of episodes to evaluate
            max_ep_len: Maximum episode length
            seed: Random seed for evaluation

        Returns:
            dict with keys: mean_return, std_return, mean_length, std_length
        """
        from jaxdp.base import greedy_policy

        def run_episode(key):
            mdp_state = mdp.init_state(key)
            ep_step = jnp.array(0.0)
            ep_return = jnp.array(0.0)

            def step_fn(carry, _):
                mdp_state, ep_step, ep_return = carry
                # Use greedy policy for evaluation
                policy = greedy_policy.q(state.alg_state.q_vals)
                action, next_s, reward, term, timeout, stepped_s, new_ep_step = async_sample_step_pi(
                    mdp, policy, mdp_state, ep_step, max_ep_len, key
                )
                new_return = ep_return + reward
                done = term + timeout > 0
                return (stepped_s, new_ep_step, new_return), done

            (final_mdp_state, final_ep_step, final_return), _ = jax.lax.scan(
                step_fn, (mdp_state, ep_step, ep_return), None, length=max_ep_len
            )
            return final_return, final_ep_step

        keys = jrd.split(jrd.PRNGKey(seed), n_episodes)
        returns, lengths = jax.vmap(run_episode)(keys)

        return {
            "mean_return": jnp.mean(returns),
            "std_return": jnp.std(returns),
            "mean_length": jnp.mean(lengths),
            "std_length": jnp.std(lengths)
        }


def grid_mdp_factory() -> MDP:
    """ Create a GridWorld MDP """
    board = [
        "#####",
        "#  @#",
        "# #X#",
        "#P  #",
        "#####"
    ]
    return grid_world(board=board, p_slip=0.0)


def garnet_mdp_factory(key: jrd.PRNGKey, state_size: int,
                       action_size: int, branch_size: int) -> MDP:
    """ Create a Garnet MDP """
    return garnet_mdp(
        state_size=state_size,
        action_size=action_size,
        branch_size=branch_size,
        key=key
    )


def graph_mdp_factory() -> MDP:
    """ Create a Graph MDP """
    return graph_mdp()


def q_learning_parallel_envs():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning with Parallel Environments
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = grid_mdp_factory()
    alg_name = "Q-Learning (Parallel Envs)"
    loop_args = LoopArgs(seed=0, n_steps=5000, max_ep_len=50, n_envs=4)

    # Initialize loop state
    state = loop.init(
        q_learning, epsilon_greedy, mdp, loop_args,
        alg_params={"gamma": 0.99, "alpha": 0.5},
        policy_params={"epsilon": 1.0, "eps_decay": 0.997, "eps_min": 0.1}
    )

    # Train
    final_state, metrics = loop.train(
        q_learning, epsilon_greedy, mdp, state, loop_args, compute_metrics
    )

    results = {"GridWorld": (metrics, final_state.alg_state.q_vals)}
    log_results(results, alg_name)
    return final_state.alg_state.q_vals


def q_learning_multi_seed():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning Multi-Seed (Parallel Independent Runs)
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = grid_mdp_factory()
    alg_name = "Q-Learning (Multi-Seed)"
    n_seeds = 5

    # Run independent training loops with different seeds
    def run_one_seed(seed):
        loop_args = LoopArgs(seed=seed, n_steps=5000, max_ep_len=50, n_envs=1)

        # Initialize loop state
        state = loop.init(
            q_learning, epsilon_greedy, mdp, loop_args,
            alg_params={"gamma": 0.99, "alpha": 0.5},
            policy_params={"epsilon": 1.0, "eps_decay": 0.997, "eps_min": 0.1}
        )

        # Train
        final_state, metrics = loop.train(
            q_learning, epsilon_greedy, mdp, state, loop_args, compute_metrics
        )
        return final_state, metrics

    # Vmap over different seeds
    seeds = jnp.arange(n_seeds)
    vmap_run = jax.vmap(run_one_seed)
    final_states, all_metrics = vmap_run(seeds)

    # Average across seeds
    avg_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), all_metrics)
    avg_q = jnp.mean(final_states.alg_state.q_vals, axis=0)

    results = {"GridWorld": (avg_metrics, avg_q)}
    log_results(results, alg_name)
    return results


def q_learning_grid_world():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning GridWorld
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = grid_mdp_factory()
    alg_name = "Q-Learning"
    loop_args = LoopArgs(seed=0, n_steps=5000, max_ep_len=50, n_envs=1)

    # Initialize loop state
    state = loop.init(
        q_learning, epsilon_greedy, mdp, loop_args,
        alg_params={"gamma": 0.99, "alpha": 0.5},
        policy_params={"epsilon": 1.0, "eps_decay": 0.997, "eps_min": 0.1}
    )

    # Train
    final_state, metrics = loop.train(
        q_learning, epsilon_greedy, mdp, state, loop_args, compute_metrics
    )

    results = {"GridWorld": (metrics, final_state.alg_state.q_vals)}
    log_results(results, alg_name)
    return final_state.alg_state.q_vals


def q_learning_garnet():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning Garnet MDP
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = garnet_mdp_factory(jrd.PRNGKey(42), state_size=10, action_size=4, branch_size=2)
    alg_name = "Q-Learning"
    loop_args = LoopArgs(seed=0, n_steps=30000, max_ep_len=50, n_envs=1)

    # Initialize loop state
    state = loop.init(
        q_learning, epsilon_greedy, mdp, loop_args,
        alg_params={"gamma": 0.99, "alpha": 0.3},
        policy_params={"epsilon": 1.0, "eps_decay": 0.9995, "eps_min": 0.05}
    )

    # Train
    final_state, metrics = loop.train(
        q_learning, epsilon_greedy, mdp, state, loop_args, compute_metrics
    )

    results = {"GarnetMDP": (metrics, final_state.alg_state.q_vals)}
    log_results(results, alg_name)
    return final_state.alg_state.q_vals


def q_learning_graph():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning Graph MDP
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = graph_mdp_factory()
    alg_name = "Q-Learning"
    loop_args = LoopArgs(seed=0, n_steps=40000, max_ep_len=50, n_envs=1)

    # Initialize loop state
    state = loop.init(
        q_learning, epsilon_greedy, mdp, loop_args,
        alg_params={"gamma": 0.99, "alpha": 0.5},
        policy_params={"epsilon": 1.0, "eps_decay": 0.9996, "eps_min": 0.05}
    )

    # Train
    final_state, metrics = loop.train(
        q_learning, epsilon_greedy, mdp, state, loop_args, compute_metrics
    )

    results = {"GraphMDP": (metrics, final_state.alg_state.q_vals)}
    log_results(results, alg_name)
    return final_state.alg_state.q_vals


def q_learning_benchmark():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning Comprehensive Benchmark
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    alg_name = "Q-Learning"
    max_ep_len = 50

    # Tuned hyperparameters for each MDP
    mdp_configs = {
        "GridWorld": {
            "mdp": grid_mdp_factory(),
            "n_steps": 5000,
            "alpha": 0.5,
            "epsilon": 1.0,
            "eps_decay": 0.997,
            "eps_min": 0.1
        },
        "GarnetMDP": {
            "mdp": garnet_mdp_factory(jrd.PRNGKey(42), state_size=10, action_size=4, branch_size=2),
            "n_steps": 30000,
            "alpha": 0.3,
            "epsilon": 1.0,
            "eps_decay": 0.9995,
            "eps_min": 0.05
        },
        "GraphMDP": {
            "mdp": graph_mdp_factory(),
            "n_steps": 40000,
            "alpha": 0.5,
            "epsilon": 1.0,
            "eps_decay": 0.9996,
            "eps_min": 0.05
        }
    }

    results = {}
    for mdp_name, config in mdp_configs.items():
        loop_args = LoopArgs(seed=0, n_steps=config["n_steps"], max_ep_len=max_ep_len, n_envs=1)

        # Initialize loop state
        state = loop.init(
            q_learning, epsilon_greedy, config["mdp"], loop_args,
            alg_params={"gamma": 0.99, "alpha": config["alpha"]},
            policy_params={
                "epsilon": config["epsilon"],
                "eps_decay": config["eps_decay"],
                "eps_min": config["eps_min"]
            }
        )

        # Train
        final_state, metrics = loop.train(
            q_learning, epsilon_greedy, config["mdp"], state, loop_args, compute_metrics
        )

        results[mdp_name] = (metrics, final_state.alg_state.q_vals)

    log_results(results, alg_name)
    return results


@click.command()
@click.argument(
    "benchmark_type",
    type=click.Choice(["q_learning", "parallel_envs", "multi_seed", "q_learning_garnet", "q_learning_graph", "benchmark"])
)
def main(benchmark_type):
    """Run JAX Learning benchmarks"""
    if benchmark_type == "q_learning":
        q_learning_grid_world()
    elif benchmark_type == "parallel_envs":
        q_learning_parallel_envs()
    elif benchmark_type == "multi_seed":
        q_learning_multi_seed()
    elif benchmark_type == "q_learning_garnet":
        q_learning_garnet()
    elif benchmark_type == "q_learning_graph":
        q_learning_graph()
    elif benchmark_type == "benchmark":
        q_learning_benchmark()


if __name__ == "__main__":
    main()
