import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrd
from algorithms import q_learning
from flax import struct
from utils import log_results

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
    ep_return: jnp.ndarray
    ep_len: jnp.ndarray


def compute_metrics(prev_state, new_state, mdp, step):
    """ Compute metrics for the current iteration """
    diff = new_state.q_vals - prev_state.q_vals
    l1 = jnp.sum(jnp.abs(diff))
    l2 = jnp.sqrt(jnp.sum(diff**2))
    linf = jnp.max(jnp.abs(diff))

    bellman_target = bellman_op.q(mdp, prev_state.q_vals, prev_state.gamma)
    bellman_err = jnp.max(jnp.abs(prev_state.q_vals - bellman_target))

    # Episode completed when ep_step resets to 0
    ep_done = (new_state.ep_step == 0) & (prev_state.ep_step > 0)
    ep_return = jnp.where(ep_done, new_state.last_return, 0.0)
    ep_len = jnp.where(ep_done, prev_state.ep_step, 0.0)

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


def loop(mdp, alg_state, args, update_fn, metrics_fn):
    """Run learning loop for n_steps"""
    keys = jrd.split(jrd.PRNGKey(args.seed), args.n_steps)

    state = alg_state
    metrics_list = []

    for i in range(args.n_steps):
        prev = state
        state = update_fn(state, mdp, i, args.max_ep_len, keys[i])
        metrics_list.append(metrics_fn(prev, state, mdp, i))

    all_metrics = jax.tree.map(lambda *x: jnp.stack(x), *metrics_list)
    return state, all_metrics


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


def q_learning_grid_world():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning GridWorld
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = grid_mdp_factory()
    alg_name = "Q-Learning"
    loop_args = LoopArgs(seed=12345, n_steps=10000, max_ep_len=50)

    init_state = q_learning.init(mdp, jrd.PRNGKey(loop_args.seed),
                                  gamma=0.99, alpha=0.5, epsilon=0.4)

    final_state, metrics = loop(
        mdp, init_state, loop_args,
        q_learning.update, compute_metrics
    )

    results = {"GridWorld": (metrics, final_state.q_vals)}
    log_results(results, alg_name)
    return final_state.q_vals


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
