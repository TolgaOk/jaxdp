from typing import Any

import click
import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct

from jaxdp import async_sample_step_pi
from jaxdp.base import bellman_optimality_operator as bellman_op
from jaxdp.mdp import MDP
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.mdp.grid_world import grid_world
from jaxdp.mdp.simple_graph import graph_mdp
from jaxdp.typehints import F, StaticMeta

from algorithms import Transition, q_learning
from policies import epsilon_greedy, soft_policy
from utils import log_results

jax.config.update("jax_enable_x64", True)


class metrics(metaclass=StaticMeta):
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Metrics Namespace

    Computes and stores training metrics.
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    @struct.dataclass
    class State:
        """Metrics collected during training"""

        l1: F[""]  # L1 norm of Q-value change (scalar)
        l2: F[""]  # L2 norm of Q-value change (scalar)
        linf: F[""]  # L-infinity norm of Q-value change (scalar)
        bellman_err: F[""]  # Bellman error (scalar)
        iteration: F[""]  # Iteration number (scalar)
        ep_return: F["..."]  # Episode returns [n_envs] - NaN if episode not complete
        ep_len: F["..."]  # Episode lengths [n_envs] - NaN if episode not complete
        eval_mean_return: F[""]  # Mean evaluation return (scalar) - NaN if no eval
        eval_std_return: F[""]  # Std evaluation return (scalar) - NaN if no eval

    def compute(
        prev: "loop.State",
        new: "loop.State",
        args: "loop.Args",
        step: int,
        dones: F["..."],
        eval_results: "loop.EvalResult",
    ) -> "metrics.State":
        """Compute metrics for the current iteration

        Args:
            prev: Previous loop state
            new: New loop state
            args: Loop arguments
            step: Current step number
            dones: Boolean array indicating which environments completed episodes [n_envs]
            eval_results: Evaluation results (loop.EvalResult dataclass)

        Returns:
            Metrics state with NaN values where no information is available
        """
        prev_alg = prev.alg_state
        new_alg = new.alg_state

        diff = new_alg.q_vals - prev_alg.q_vals
        l1 = jnp.sum(jnp.abs(diff))
        l2 = jnp.sqrt(jnp.sum(diff**2))
        linf = jnp.max(jnp.abs(diff))

        bellman_target = bellman_op.q(args.mdp, prev_alg.q_vals, prev_alg.gamma)
        bellman_err = jnp.max(jnp.abs(prev_alg.q_vals - bellman_target))

        ep_return = jnp.where(dones, new.last_return, jnp.nan)
        ep_len = jnp.where(dones, prev.ep_step, jnp.nan)

        return metrics.State(
            l1=l1,
            l2=l2,
            linf=linf,
            bellman_err=bellman_err,
            iteration=step,
            ep_return=ep_return,
            ep_len=ep_len,
            eval_mean_return=eval_results.mean_return,
            eval_std_return=eval_results.std_return,
        )


class loop(metaclass=StaticMeta):
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Training Loop for Value-based RL Algorithms

    Provides init(), train(), and evaluate() functions for managing the
    training loop state and running value-based RL algorithms with various
    exploration policies.
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    @struct.dataclass
    class State:
        """Training loop state - manages MDP interaction and episode tracking"""

        alg_state: Any  # q_learning.State
        policy_state: Any  # epsilon_greedy.State or soft_policy.State, etc.
        mdp_state: F["... S"]  # MDP states [n_envs, n_states]
        ep_step: F["..."]  # Episode step counters [n_envs]
        ep_return: F["..."]  # Current episode returns [n_envs]
        last_return: F["..."]  # Last completed episode returns [n_envs]

    @struct.dataclass
    class Args:
        """Training loop arguments - static configuration"""

        value_fn: Any  # Algorithm namespace (e.g., q_learning)
        policy_ns: Any  # Policy namespace (e.g., epsilon_greedy)
        mdp: MDP
        seed: int
        n_steps: int
        max_ep_len: int
        n_envs: int
        eval_period: int = 0  # Evaluate every N steps (0 = no evaluation)
        n_eval_episodes: int = 10  # Number of episodes for evaluation
        eval_seed: int = 42  # Seed for evaluation

    @struct.dataclass
    class EvalResult:
        """Evaluation results"""

        mean_return: F[""]  # Mean return across eval episodes (scalar)
        std_return: F[""]  # Std return across eval episodes (scalar)
        mean_length: F[""]  # Mean episode length (scalar)
        std_length: F[""]  # Std episode length (scalar)

    def init(alg_state: Any, policy_state: Any, args: "loop.Args") -> "loop.State":
        """Initialize loop state with algorithm and policy states

        Args:
            alg_state: Initialized algorithm state (e.g., q_learning.State)
            policy_state: Initialized policy state (e.g., epsilon_greedy.State)
            args: Loop arguments (includes mdp, seed, n_steps, n_envs, etc.)

        Returns:
            Initialized loop state with environment states and episode tracking
        """
        key = jrd.PRNGKey(args.seed)

        env_keys = jrd.split(key, args.n_envs)
        mdp_state = jax.vmap(args.mdp.init_state)(env_keys)
        ep_step = jnp.zeros(args.n_envs)
        ep_return = jnp.zeros(args.n_envs)
        last_return = jnp.zeros(args.n_envs)

        return loop.State(
            alg_state=alg_state,
            policy_state=policy_state,
            mdp_state=mdp_state,
            ep_step=ep_step,
            ep_return=ep_return,
            last_return=last_return,
        )

    def train(state: "loop.State", args: "loop.Args") -> tuple["loop.State", Any]:
        """Run training loop for n_steps with specified value function and policy

        Args:
            state: Initial loop state
            args: Loop arguments (includes value_fn, policy_ns, mdp, n_steps, etc.)

        Returns:
            Final loop state and all metrics collected during training
        """

        def step_fn(state: loop.State, step_and_keys):
            step, keys = step_and_keys
            prev = state

            policy = args.policy_ns.get_policy(state.alg_state.q_vals, state.policy_state)

            vmap_sample = jax.vmap(
                lambda s, ep, k: async_sample_step_pi(args.mdp, policy, s, ep, args.max_ep_len, k),
                in_axes=(0, 0, 0),
            )
            actions, next_states, rewards, terms, timeouts, stepped_states, ep_steps = vmap_sample(
                state.mdp_state, state.ep_step, keys
            )

            transitions = Transition(
                state=state.mdp_state,
                action=actions,
                reward=rewards,
                next_state=next_states,
                terminal=terms,
            )

            new_alg_state = args.value_fn.batch_update(state.alg_state, transitions)

            dones = terms + timeouts > 0
            any_done = jnp.any(dones)
            new_policy_state = args.policy_ns.update(state.policy_state, any_done)

            new_returns = state.ep_return + rewards
            last_returns = jnp.where(dones, new_returns, state.last_return)
            ep_returns = jnp.where(dones, 0.0, new_returns)

            new_state = state.replace(
                alg_state=new_alg_state,
                policy_state=new_policy_state,
                mdp_state=stepped_states,
                ep_step=ep_steps,
                ep_return=ep_returns,
                last_return=last_returns,
            )

            should_eval = (args.eval_period > 0) & ((step + 1) % args.eval_period == 0)
            eval_results = jax.lax.cond(
                should_eval,
                lambda s: loop.evaluate(s, args),
                lambda s: loop.EvalResult(
                    mean_return=jnp.nan, std_return=jnp.nan, mean_length=jnp.nan, std_length=jnp.nan
                ),
                new_state,
            )

            metrics_state = metrics.compute(prev, new_state, args, step, dones, eval_results)
            return new_state, metrics_state

        keys = jrd.split(jrd.PRNGKey(args.seed), args.n_steps * args.n_envs)
        keys = keys.reshape(args.n_steps, args.n_envs, -1)
        steps_and_keys = (jnp.arange(args.n_steps), keys)

        final_state, all_metrics = jax.lax.scan(step_fn, state, steps_and_keys)
        return final_state, all_metrics

    def evaluate(state: "loop.State", args: "loop.Args") -> "loop.EvalResult":
        """Evaluate the learned policy (greedy, no exploration)

        Args:
            state: Current loop state with learned Q-values
            args: Loop arguments (includes mdp, max_ep_len, n_eval_episodes, eval_seed)

        Returns:
            loop.EvalResult dataclass with evaluation statistics
        """
        from jaxdp.base import greedy_policy

        def run_episode(key):
            mdp_state = args.mdp.init_state(key)
            ep_step = jnp.array(0.0)
            ep_return = jnp.array(0.0)
            done = jnp.array(False)

            def step_fn(carry, _):
                mdp_state, ep_step, ep_return, done = carry
                policy = greedy_policy.q(state.alg_state.q_vals)
                action, next_s, reward, term, timeout, stepped_s, new_ep_step = (
                    async_sample_step_pi(args.mdp, policy, mdp_state, ep_step, args.max_ep_len, key)
                )
                new_return = jnp.where(done, ep_return, ep_return + reward)
                new_done = done | (term + timeout > 0)
                return (stepped_s, new_ep_step, new_return, new_done), None

            (final_mdp_state, final_ep_step, final_return, final_done), _ = jax.lax.scan(
                step_fn, (mdp_state, ep_step, ep_return, done), None, length=args.max_ep_len
            )
            return final_return, final_ep_step

        keys = jrd.split(jrd.PRNGKey(args.eval_seed), args.n_eval_episodes)
        returns, lengths = jax.vmap(run_episode)(keys)

        return loop.EvalResult(
            mean_return=jnp.mean(returns),
            std_return=jnp.std(returns),
            mean_length=jnp.mean(lengths),
            std_length=jnp.std(lengths),
        )


def grid_mdp_factory() -> MDP:
    """
    Create a GridWorld MDP with fixed 5x5 layout.

    Returns:
        GridWorld MDP with player at (1,3), goal at (1,3), and wall at (2,2)
    """
    board = ["#####", "#  @#", "# #X#", "#P  #", "#####"]
    return grid_world(board=board, p_slip=0.0)


def garnet_mdp_factory(
    key: jrd.PRNGKey, state_size: int, action_size: int, branch_size: int
) -> MDP:
    """
    Create a random Garnet MDP.

    Args:
        key: Random key for MDP generation
        state_size: Number of states in the MDP
        action_size: Number of actions available
        branch_size: Branching factor for transitions

    Returns:
        Randomly generated Garnet MDP
    """
    return garnet_mdp(
        state_size=state_size, action_size=action_size, branch_size=branch_size, key=key
    )


def graph_mdp_factory() -> MDP:
    """
    Create a simple graph-based MDP.

    Returns:
        Graph MDP with predefined structure
    """
    return graph_mdp()


def q_learning_parallel_envs():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning with Parallel Environments
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = grid_mdp_factory()
    alg_name = "Q-Learning (Parallel Envs)"

    args = loop.Args(
        value_fn=q_learning,
        policy_ns=epsilon_greedy,
        mdp=mdp,
        seed=0,
        n_steps=5000,
        max_ep_len=50,
        n_envs=16,
        eval_period=1000,
        n_eval_episodes=10,
        eval_seed=42,
    )

    key = jrd.PRNGKey(args.seed)
    alg_state = q_learning.init(mdp, key, gamma=0.99, alpha=0.5)
    policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.997, eps_min=0.1)

    state = loop.init(alg_state, policy_state, args)

    final_state, all_metrics = loop.train(state, args)

    results = {"GridWorld": (all_metrics, final_state.alg_state.q_vals)}
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

    def run_one_seed(seed):
        args = loop.Args(
            value_fn=q_learning,
            policy_ns=epsilon_greedy,
            mdp=mdp,
            seed=seed,
            n_steps=5000,
            max_ep_len=50,
            n_envs=1,
        )

        key = jrd.PRNGKey(seed)
        alg_state = q_learning.init(mdp, key, gamma=0.99, alpha=0.5)
        policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.997, eps_min=0.1)

        state = loop.init(alg_state, policy_state, args)

        final_state, all_metrics = loop.train(state, args)
        return final_state, all_metrics

    seeds = jnp.arange(n_seeds)
    vmap_run = jax.vmap(run_one_seed)
    final_states, all_metrics = vmap_run(seeds)

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

    args = loop.Args(
        value_fn=q_learning,
        policy_ns=epsilon_greedy,
        mdp=mdp,
        seed=0,
        n_steps=5000,
        max_ep_len=50,
        n_envs=1,
        eval_period=1000,
        n_eval_episodes=10,
        eval_seed=42,
    )

    key = jrd.PRNGKey(args.seed)
    alg_state = q_learning.init(mdp, key, gamma=0.99, alpha=0.5)
    policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.997, eps_min=0.1)

    state = loop.init(alg_state, policy_state, args)

    final_state, all_metrics = loop.train(state, args)

    results = {"GridWorld": (all_metrics, final_state.alg_state.q_vals)}
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

    args = loop.Args(
        value_fn=q_learning,
        policy_ns=epsilon_greedy,
        mdp=mdp,
        seed=0,
        n_steps=30000,
        max_ep_len=50,
        n_envs=1,
    )

    key = jrd.PRNGKey(args.seed)
    alg_state = q_learning.init(mdp, key, gamma=0.99, alpha=0.3)
    policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.9995, eps_min=0.05)

    state = loop.init(alg_state, policy_state, args)

    final_state, all_metrics = loop.train(state, args)

    results = {"GarnetMDP": (all_metrics, final_state.alg_state.q_vals)}
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

    args = loop.Args(
        value_fn=q_learning,
        policy_ns=epsilon_greedy,
        mdp=mdp,
        seed=0,
        n_steps=40000,
        max_ep_len=50,
        n_envs=1,
    )

    key = jrd.PRNGKey(args.seed)
    alg_state = q_learning.init(mdp, key, gamma=0.99, alpha=0.5)
    policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.9996, eps_min=0.05)

    state = loop.init(alg_state, policy_state, args)

    final_state, all_metrics = loop.train(state, args)

    results = {"GraphMDP": (all_metrics, final_state.alg_state.q_vals)}
    log_results(results, alg_name)
    return final_state.alg_state.q_vals


def q_learning_benchmark():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning Comprehensive Benchmark
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    alg_name = "Q-Learning"

    @struct.dataclass
    class MDPConfig:
        name: str
        mdp: MDP
        n_steps: int
        alpha: float
        epsilon: float
        eps_decay: float
        eps_min: float

    configs = [
        MDPConfig(
            name="GridWorld",
            mdp=grid_mdp_factory(),
            n_steps=5000,
            alpha=0.5,
            epsilon=1.0,
            eps_decay=0.997,
            eps_min=0.1,
        ),
        MDPConfig(
            name="GarnetMDP",
            mdp=garnet_mdp_factory(jrd.PRNGKey(42), state_size=10, action_size=4, branch_size=2),
            n_steps=30000,
            alpha=0.3,
            epsilon=1.0,
            eps_decay=0.9995,
            eps_min=0.05,
        ),
        MDPConfig(
            name="GraphMDP",
            mdp=graph_mdp_factory(),
            n_steps=40000,
            alpha=0.5,
            epsilon=1.0,
            eps_decay=0.9996,
            eps_min=0.05,
        ),
    ]

    results = {}
    for config in configs:
        args = loop.Args(
            value_fn=q_learning,
            policy_ns=epsilon_greedy,
            mdp=config.mdp,
            seed=0,
            n_steps=config.n_steps,
            max_ep_len=50,
            n_envs=1,
        )

        key = jrd.PRNGKey(args.seed)
        alg_state = q_learning.init(config.mdp, key, gamma=0.99, alpha=config.alpha)
        policy_state = epsilon_greedy.init(
            epsilon=config.epsilon, eps_decay=config.eps_decay, eps_min=config.eps_min
        )

        state = loop.init(alg_state, policy_state, args)

        final_state, all_metrics = loop.train(state, args)

        results[config.name] = (all_metrics, final_state.alg_state.q_vals)

    log_results(results, alg_name)
    return results


@click.command()
@click.argument(
    "benchmark_type",
    type=click.Choice(
        [
            "q_learning",
            "parallel_envs",
            "multi_seed",
            "q_learning_garnet",
            "q_learning_graph",
            "benchmark",
        ]
    ),
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
