import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrd
from algorithms import q_learning
from flax import struct
from utils import log_learning_progress, log_results

from jaxdp.base import bellman_optimality_operator as bellman_op
from jaxdp.mdp import MDP
from jaxdp.mdp.grid_world import grid_world

jax.config.update("jax_enable_x64", True)


@struct.dataclass
class Metrics:
    l1: jnp.ndarray
    l2: jnp.ndarray
    linf: jnp.ndarray
    bellman_err: jnp.ndarray
    iteration: jnp.ndarray
    episode_reward: jnp.ndarray


def compute_metrics(prev_state, new_state, mdp, step):
    """Compute metrics for the current iteration"""
    new_q = new_state.q_vals
    prev_q = prev_state.q_vals
    gamma = prev_state.gamma

    diff = new_q - prev_q
    l1 = jnp.sum(jnp.abs(diff))
    l2 = jnp.sqrt(jnp.sum(diff**2))
    linf = jnp.max(jnp.abs(diff))

    bellman_target = bellman_op.q(mdp, prev_q, gamma)
    bellman_err = jnp.max(jnp.abs(prev_q - bellman_target))

    # For now, we don't track episode rewards in metrics
    episode_reward = jnp.array(0.0)

    return Metrics(
        l1=l1,
        l2=l2,
        linf=linf,
        bellman_err=bellman_err,
        iteration=step,
        episode_reward=episode_reward
    )


@dataclass(frozen=True)
class LoopArgs:
    seed: int
    n_steps: int
    max_episode_len: int = 1000


def loop(mdp: MDP,
         alg_state,
         args: LoopArgs,
         update_fn,
         metrics_fn):
    """
    Run the learning loop for a fixed number of steps.

    Args:
        mdp: Markov Decision Process
        alg_state: Initial state of the algorithm
        args: Loop arguments
        update_fn: Function to update the algorithm state
        metrics_fn: Function to compute metrics

    Returns:
        Final algorithm state and all metrics collected during the loop
    """
    # Generate all random keys upfront
    master_key = jrd.PRNGKey(args.seed)
    step_keys = jrd.split(master_key, args.n_steps)

    def scan_body(state, inputs):
        iter_idx, key = inputs
        prev_state = state
        new_state = update_fn(state, mdp, iter_idx, args.max_episode_len, key)

        metrics = metrics_fn(prev_state, new_state, mdp, iter_idx)

        return new_state, metrics

    final_state, all_metrics = jax.lax.scan(
        scan_body,
        alg_state,
        (jnp.arange(args.n_steps), step_keys)
    )

    return final_state, all_metrics


def grid_mdp_factory() -> MDP:
    """Create a GridWorld MDP"""
    board = [
        "#####",
        "#  @#",
        "# #X#",
        "#P  #",
        "#####"
    ]
    return grid_world(board=board, p_slip=0.0)


def q_learning_grid_world():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Single Q-Learning GridWorld
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = grid_mdp_factory()
    alg_name = "Q-Learning"
    loop_args = LoopArgs(seed=42, n_steps=50000, max_episode_len=50)

    # Q-learning hyperparameters
    gamma = 0.99
    alpha = 0.3  # Learning rate
    epsilon = 0.2  # Exploration rate

    init_state = q_learning.init(
        mdp, jrd.PRNGKey(loop_args.seed),
        gamma=gamma, alpha=alpha, epsilon=epsilon
    )

    update_fn = q_learning.update

    final_state, metrics = loop(
        mdp=mdp,
        alg_state=init_state,
        args=loop_args,
        update_fn=update_fn,
        metrics_fn=compute_metrics
    )

    # Log learning progress
    log_learning_progress(metrics, final_state, loop_args.n_steps, print_every=2000)

    # Log final results
    results = {"GridWorld": (metrics, final_state)}
    log_results(results, alg_name)

    return final_state.q_vals, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JAX Learning benchmarks")
    parser.add_argument(
        "benchmark_type",
        choices=["q_learning"],
        help="Type of benchmark to run"
    )

    args = parser.parse_args()

    if args.benchmark_type == "q_learning":
        q_learning_grid_world()
