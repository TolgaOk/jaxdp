import argparse
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrd
from algorithms import q_learning
from flax import struct
from utils import log_results

from jaxdp import async_sample_step_pi
from jaxdp.base import bellman_optimality_operator as bellman_op
from jaxdp.base import e_greedy_policy
from jaxdp.mdp import MDP
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.mdp.grid_world import grid_world
from jaxdp.mdp.simple_graph import graph_mdp

jax.config.update("jax_enable_x64", True)


@struct.dataclass
class LoopState:
    """Training loop state - manages MDP interaction and episode tracking"""
    alg_state: Any  # q_learning.State
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


def compute_metrics(prev_loop_state: LoopState, new_loop_state: LoopState, mdp: MDP, step: int):
    """ Compute metrics for the current iteration """
    prev_alg = prev_loop_state.alg_state
    new_alg = new_loop_state.alg_state

    diff = new_alg.q_vals - prev_alg.q_vals
    l1 = jnp.sum(jnp.abs(diff))
    l2 = jnp.sqrt(jnp.sum(diff**2))
    linf = jnp.max(jnp.abs(diff))

    bellman_target = bellman_op.q(mdp, prev_alg.q_vals, prev_alg.gamma)
    bellman_err = jnp.max(jnp.abs(prev_alg.q_vals - bellman_target))

    # Episode completed when ep_step resets to 0
    ep_done = (new_loop_state.ep_step == 0) & (prev_loop_state.ep_step > 0)
    ep_return = jnp.where(ep_done, new_loop_state.last_return, 0.0)
    ep_len = jnp.where(ep_done, prev_loop_state.ep_step, 0.0)

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


def loop(mdp: MDP, init_loop_state: LoopState, args: LoopArgs, metrics_fn):
    """Run Q-learning loop for n_steps"""
    keys = jrd.split(jrd.PRNGKey(args.seed), args.n_steps)

    loop_state = init_loop_state
    metrics_list = []

    for i in range(args.n_steps):
        prev_loop_state = loop_state

        # Sample from MDP using epsilon-greedy policy
        policy = e_greedy_policy.q(loop_state.alg_state.q_vals, loop_state.alg_state.epsilon)
        action, next_s, reward, term, timeout, stepped_s, ep_step = async_sample_step_pi(
            mdp, policy, loop_state.mdp_state, loop_state.ep_step, args.max_ep_len, keys[i]
        )

        # Update algorithm state (Q-values)
        done = term + timeout > 0
        new_alg_state = q_learning.update(
            loop_state.alg_state, loop_state.mdp_state, action, next_s, reward, term, done
        )

        # Update loop state (MDP state, episode tracking)
        new_return = loop_state.ep_return + reward
        last_return = jnp.where(done, new_return, loop_state.last_return)
        ep_return = jnp.where(done, 0.0, new_return)

        loop_state = LoopState(
            alg_state=new_alg_state,
            mdp_state=stepped_s,
            ep_step=ep_step,
            ep_return=ep_return,
            last_return=last_return
        )

        metrics_list.append(metrics_fn(prev_loop_state, loop_state, mdp, i))

    all_metrics = jax.tree.map(lambda *x: jnp.stack(x), *metrics_list)
    return loop_state, all_metrics


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


def q_learning_grid_world():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning GridWorld
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = grid_mdp_factory()
    alg_name = "Q-Learning"
    loop_args = LoopArgs(seed=0, n_steps=5000, max_ep_len=50)

    key = jrd.PRNGKey(loop_args.seed)
    key, init_key = jrd.split(key)

    alg_state = q_learning.init(
        mdp, init_key, gamma=0.99, alpha=0.5, epsilon=1.0,
        eps_decay=0.997, eps_min=0.1
    )

    init_loop_state = LoopState(
        alg_state=alg_state,
        mdp_state=mdp.init_state(key),
        ep_step=jnp.array(0.0),
        ep_return=jnp.array(0.0),
        last_return=jnp.array(0.0)
    )

    final_state, metrics = loop(mdp, init_loop_state, loop_args, compute_metrics)

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

    key = jrd.PRNGKey(loop_args.seed)
    key, init_key = jrd.split(key)

    alg_state = q_learning.init(
        mdp, init_key, gamma=0.99, alpha=0.3, epsilon=1.0,
        eps_decay=0.9995, eps_min=0.05
    )

    init_loop_state = LoopState(
        alg_state=alg_state,
        mdp_state=mdp.init_state(key),
        ep_step=jnp.array(0.0),
        ep_return=jnp.array(0.0),
        last_return=jnp.array(0.0)
    )

    final_state, metrics = loop(mdp, init_loop_state, loop_args, compute_metrics)

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

    key = jrd.PRNGKey(loop_args.seed)
    key, init_key = jrd.split(key)

    alg_state = q_learning.init(
        mdp, init_key, gamma=0.99, alpha=0.5, epsilon=1.0,
        eps_decay=0.9996, eps_min=0.05
    )

    init_loop_state = LoopState(
        alg_state=alg_state,
        mdp_state=mdp.init_state(key),
        ep_step=jnp.array(0.0),
        ep_return=jnp.array(0.0),
        last_return=jnp.array(0.0)
    )

    final_state, metrics = loop(mdp, init_loop_state, loop_args, compute_metrics)

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
        loop_args = LoopArgs(seed=0, n_steps=config["n_steps"], max_ep_len=max_ep_len)

        key = jrd.PRNGKey(loop_args.seed)
        key, init_key = jrd.split(key)

        alg_state = q_learning.init(
            config["mdp"], init_key,
            gamma=0.99, alpha=config["alpha"], epsilon=config["epsilon"],
            eps_decay=config["eps_decay"], eps_min=config["eps_min"]
        )

        init_loop_state = LoopState(
            alg_state=alg_state,
            mdp_state=config["mdp"].init_state(key),
            ep_step=jnp.array(0.0),
            ep_return=jnp.array(0.0),
            last_return=jnp.array(0.0)
        )

        final_state, metrics = loop(config["mdp"], init_loop_state, loop_args, compute_metrics)

        results[mdp_name] = (metrics, final_state.alg_state.q_vals)

    log_results(results, alg_name)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JAX Learning benchmarks")
    parser.add_argument(
        "benchmark_type",
        choices=["q_learning", "q_learning_garnet", "q_learning_graph", "benchmark"],
        help="Type of benchmark to run"
    )

    args = parser.parse_args()

    if args.benchmark_type == "q_learning":
        q_learning_grid_world()
    elif args.benchmark_type == "q_learning_garnet":
        q_learning_garnet()
    elif args.benchmark_type == "q_learning_graph":
        q_learning_graph()
    elif args.benchmark_type == "benchmark":
        q_learning_benchmark()
