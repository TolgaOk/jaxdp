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


def loop(mdp: MDP, init: LoopState, args: LoopArgs, policy_module, metrics_fn):
    """Run Q-learning loop for n_steps with specified exploration policy

    If args.n_envs > 1, runs multiple environments in parallel at each step.
    """
    if args.n_envs == 1:
        # Single environment mode
        def step_fn(state: LoopState, step_and_key):
            step, key = step_and_key
            prev = state

            # Sample from MDP using policy
            policy = policy_module.get_policy(state.alg_state.q_vals, state.policy_state)
            action, next_s, reward, term, timeout, stepped_s, ep_step = async_sample_step_pi(
                mdp, policy, state.mdp_state, state.ep_step, args.max_ep_len, key
            )

            # Update algorithm state (Q-values)
            new_alg_state = q_learning.update(
                state.alg_state, state.mdp_state, action, next_s, reward, term
            )

            # Update policy state (decay exploration parameter)
            done = term + timeout > 0
            new_policy_state = policy_module.update(state.policy_state, done)

            # Update loop state (MDP state, episode tracking)
            new_return = state.ep_return + reward
            last_return = jnp.where(done, new_return, state.last_return)
            ep_return = jnp.where(done, 0.0, new_return)

            new_state = LoopState(
                alg_state=new_alg_state,
                policy_state=new_policy_state,
                mdp_state=stepped_s,
                ep_step=ep_step,
                ep_return=ep_return,
                last_return=last_return
            )

            metrics = metrics_fn(prev, new_state, mdp, step)
            return new_state, metrics
    else:
        # Parallel environments mode
        def step_fn(state: LoopState, step_and_keys):
            step, keys = step_and_keys  # keys shape: [n_envs]
            prev = state

            # Sample from all environments in parallel
            policy = policy_module.get_policy(state.alg_state.q_vals, state.policy_state)

            # Vmap over environments
            vmap_sample = jax.vmap(
                lambda s, ep, k: async_sample_step_pi(mdp, policy, s, ep, args.max_ep_len, k),
                in_axes=(0, 0, 0)
            )
            actions, next_states, rewards, terms, timeouts, stepped_states, ep_steps = vmap_sample(
                state.mdp_state, state.ep_step, keys
            )

            # Update Q-values for each transition in parallel, then average
            vmap_update = jax.vmap(q_learning.update, in_axes=(None, 0, 0, 0, 0, 0))
            updated_states = vmap_update(
                state.alg_state, state.mdp_state, actions, next_states, rewards, terms
            )
            # Average Q-values across all parallel updates
            avg_q_vals = jnp.mean(updated_states.q_vals, axis=0)
            new_alg_state = state.alg_state.replace(q_vals=avg_q_vals)

            # Update policy state (use any episode completion signal)
            dones = terms + timeouts > 0
            any_done = jnp.any(dones)
            new_policy_state = policy_module.update(state.policy_state, any_done)

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

    if args.n_envs == 1:
        keys = keys[:, 0, :]  # Shape: [n_steps, 2]
        steps_and_keys = (jnp.arange(args.n_steps), keys)
    else:
        steps_and_keys = (jnp.arange(args.n_steps), keys)

    final_state, all_metrics = jax.lax.scan(step_fn, init, steps_and_keys)
    return final_state, all_metrics


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
    n_envs = 4
    loop_args = LoopArgs(seed=0, n_steps=5000, max_ep_len=50, n_envs=n_envs)
    policy_module = epsilon_greedy

    key = jrd.PRNGKey(loop_args.seed)
    key, init_key = jrd.split(key)

    alg_state = q_learning.init(mdp, init_key, gamma=0.99, alpha=0.5)
    policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.997, eps_min=0.1)

    # Initialize parallel environment states
    env_keys = jrd.split(key, n_envs)
    mdp_states = jax.vmap(mdp.init_state)(env_keys)

    state = LoopState(
        alg_state=alg_state,
        policy_state=policy_state,
        mdp_state=mdp_states,  # Shape: [n_envs, n_states]
        ep_step=jnp.zeros(n_envs),
        ep_return=jnp.zeros(n_envs),
        last_return=jnp.zeros(n_envs)
    )

    final_state, metrics = loop(mdp, state, loop_args, policy_module, compute_metrics)

    results = {"GridWorld": (metrics, final_state.alg_state.q_vals)}
    log_results(results, alg_name)
    return final_state.alg_state.q_vals


def q_learning_multi_seed():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning Multi-Seed (Parallel Environments)
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = grid_mdp_factory()
    alg_name = "Q-Learning (Multi-Seed)"
    loop_args = LoopArgs(seed=42, n_steps=5000, max_ep_len=50)
    n_seeds = 5
    policy_module = epsilon_greedy

    # Create multiple seeds
    seed_keys = jrd.split(jrd.PRNGKey(loop_args.seed), n_seeds)

    # Initialize vmap'd algorithm states
    def init_alg(key):
        return q_learning.init(mdp, key, gamma=0.99, alpha=0.5)

    vmap_init_alg = jax.vmap(init_alg)
    alg_states = vmap_init_alg(seed_keys)

    # Initialize vmap'd loop states
    def init_loop_state(alg_state, key):
        return LoopState(
            alg_state=alg_state,
            policy_state=epsilon_greedy.init(epsilon=1.0, eps_decay=0.997, eps_min=0.1),
            mdp_state=mdp.init_state(key),
            ep_step=jnp.array(0.0),
            ep_return=jnp.array(0.0),
            last_return=jnp.array(0.0)
        )

    vmap_init_loop = jax.vmap(init_loop_state)
    init_states = vmap_init_loop(alg_states, seed_keys)

    # Run vmap'd training loops
    vmap_loop = jax.vmap(loop, in_axes=(None, 0, None, None, None))
    final_states, all_metrics = vmap_loop(mdp, init_states, loop_args, policy_module, compute_metrics)

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
    loop_args = LoopArgs(seed=0, n_steps=5000, max_ep_len=50)
    policy_module = epsilon_greedy

    key = jrd.PRNGKey(loop_args.seed)
    key, init_key = jrd.split(key)

    alg_state = q_learning.init(mdp, init_key, gamma=0.99, alpha=0.5)
    policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.997, eps_min=0.1)

    state = LoopState(
        alg_state=alg_state,
        policy_state=policy_state,
        mdp_state=mdp.init_state(key),
        ep_step=jnp.array(0.0),
        ep_return=jnp.array(0.0),
        last_return=jnp.array(0.0)
    )

    final_state, metrics = loop(mdp, state, loop_args, policy_module, compute_metrics)

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
    loop_args = LoopArgs(seed=0, n_steps=30000, max_ep_len=50)
    policy_module = epsilon_greedy

    key = jrd.PRNGKey(loop_args.seed)
    key, init_key = jrd.split(key)

    alg_state = q_learning.init(mdp, init_key, gamma=0.99, alpha=0.3)
    policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.9995, eps_min=0.05)

    state = LoopState(
        alg_state=alg_state,
        policy_state=policy_state,
        mdp_state=mdp.init_state(key),
        ep_step=jnp.array(0.0),
        ep_return=jnp.array(0.0),
        last_return=jnp.array(0.0)
    )

    final_state, metrics = loop(mdp, state, loop_args, policy_module, compute_metrics)

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
    loop_args = LoopArgs(seed=0, n_steps=40000, max_ep_len=50)
    policy_module = epsilon_greedy

    key = jrd.PRNGKey(loop_args.seed)
    key, init_key = jrd.split(key)

    alg_state = q_learning.init(mdp, init_key, gamma=0.99, alpha=0.5)
    policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.9996, eps_min=0.05)

    state = LoopState(
        alg_state=alg_state,
        policy_state=policy_state,
        mdp_state=mdp.init_state(key),
        ep_step=jnp.array(0.0),
        ep_return=jnp.array(0.0),
        last_return=jnp.array(0.0)
    )

    final_state, metrics = loop(mdp, state, loop_args, policy_module, compute_metrics)

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

    policy_module = epsilon_greedy

    results = {}
    for mdp_name, config in mdp_configs.items():
        loop_args = LoopArgs(seed=0, n_steps=config["n_steps"], max_ep_len=max_ep_len)

        key = jrd.PRNGKey(loop_args.seed)
        key, init_key = jrd.split(key)

        alg_state = q_learning.init(config["mdp"], init_key, gamma=0.99, alpha=config["alpha"])
        policy_state = epsilon_greedy.init(
            epsilon=config["epsilon"],
            eps_decay=config["eps_decay"],
            eps_min=config["eps_min"]
        )

        state = LoopState(
            alg_state=alg_state,
            policy_state=policy_state,
            mdp_state=config["mdp"].init_state(key),
            ep_step=jnp.array(0.0),
            ep_return=jnp.array(0.0),
            last_return=jnp.array(0.0)
        )

        final_state, metrics = loop(config["mdp"], state, loop_args, policy_module, compute_metrics)

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
