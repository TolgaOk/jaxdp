import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct
from typing import Callable, Optional, Tuple, Any
import argparse

from dataclasses import dataclass

from jaxdp.mdp.grid_world import grid_world
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.mdp.simple_graph import graph_mdp
from jaxdp.mdp import MDP
from jaxdp.base import bellman_optimality_operator as bellman_op

from algorithms import vi, nesterov_vi, pi
from utils import log_results, log_multi_gamma_results, log_comprehensive_benchmark

jax.config.update("jax_enable_x64", True)


@struct.dataclass
class Metrics:
    l1: jnp.ndarray
    l2: jnp.ndarray
    linf: jnp.ndarray
    bellman_err: jnp.ndarray
    iteration: jnp.ndarray


def compute_metrics(prev_state, new_state, mdp, step):
    new_q = new_state.q_vals
    prev_q = prev_state.q_vals
    gamma = prev_state.gamma

    diff = new_q - prev_q
    l1 = jnp.sum(jnp.abs(diff))
    l2 = jnp.sqrt(jnp.sum(diff**2))
    linf = jnp.max(jnp.abs(diff))

    bellman_target = bellman_op.q(mdp, prev_q, gamma)
    bellman_err = jnp.max(jnp.abs(prev_q - bellman_target))

    return Metrics(
        l1=l1,
        l2=l2,
        linf=linf,
        bellman_err=bellman_err,
        iteration=step
    )


@dataclass(frozen=True)
class LoopArgs:
    seed: int
    n_iters: int
    n_seed: int = 1


def loop(mdp: MDP,
         alg_state: Any,
         args: LoopArgs,
         update_fn: Callable[[Any, MDP, jnp.ndarray], Any],
         metrics_fn: Callable[[Any, Any, MDP, jnp.ndarray], Any],
         callback: Optional[Callable[[int, Any], None]] = None,
         ) -> Tuple[Any, Any]:

    def scan_body(state: Any, iter_idx: jnp.ndarray) -> Tuple[Any, Any]:
        prev_state = state
        new_state = update_fn(state, mdp, iter_idx)

        metrics = metrics_fn(prev_state, new_state, mdp, iter_idx)

        if callback is not None:
            jax.debug.callback(callback, iter_idx, metrics)

        return new_state, metrics

    final_state, all_metrics = jax.lax.scan(
        scan_body,
        alg_state,
        jnp.arange(args.n_iters)
    )

    return final_state, all_metrics


def grid_mdp_factory() -> MDP:
    board = [
        "#####",
        "#  @#",
        "# #X#",
        "#P  #",
        "#####"
    ]
    return grid_world(board=board, p_slip=0.0)


def garnet_mdp_factory(key: jrd.PRNGKey, state_size: int, action_size: int, branch_size: int) -> MDP:
    return garnet_mdp(state_size=state_size, action_size=action_size, branch_size=branch_size, key=key)


def graph_mdp_factory() -> MDP:
    return graph_mdp()


def value_iteration_grid_world():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Single VI GridWorld
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = grid_mdp_factory()
    alg_name = "Value Iteration"
    loop_args = LoopArgs(seed=42, n_iters=10)

    init_state = vi.init(mdp, jrd.PRNGKey(loop_args.seed), 0.9)
    update_fn = vi.update

    final_state, metrics = loop(
        mdp=mdp,
        alg_state=init_state,
        args=loop_args,
        update_fn=update_fn,
        metrics_fn=compute_metrics
    )

    results = {"GridWorld": (metrics, final_state.q_vals)}
    log_results(results, alg_name)
    return final_state.q_vals


def value_iteration_multi_seed():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Multi-Seed VI
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    alg_name = "Value Iteration (Multi-Seed)"
    loop_args = LoopArgs(seed=42, n_iters=100, n_seed=5)

    mdp = grid_mdp_factory()

    update_fn = vi.update
    vmap_update = jax.vmap(update_fn, in_axes=(0, None, None))

    seed_keys = jrd.split(jrd.PRNGKey(loop_args.seed), loop_args.n_seed)
    vmap_init = jax.vmap(vi.init, in_axes=(None, 0, None))
    init_states = vmap_init(mdp, seed_keys, 0.99)

    vmap_metrics = jax.vmap(compute_metrics, in_axes=(0, 0, None, None))

    final_states, all_metrics = loop(
        mdp=mdp,
        alg_state=init_states,
        args=loop_args,
        update_fn=vmap_update,
        metrics_fn=vmap_metrics
    )

    avg_q = jnp.mean(final_states.q_vals, axis=0)
    avg_metrics = jax.tree.map(
        lambda x: jnp.mean(x, axis=1),
        all_metrics
    )
    results = {"GridWorld": (avg_metrics, avg_q)}

    log_results(results, alg_name)
    return results


def value_iteration_multi_gamma():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Multi-Gamma VI
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    alg_name = "Value Iteration (Multi-Gamma)"
    gammas = jnp.array([0.9, 0.95, 0.99, 0.995, 0.999])
    loop_args = LoopArgs(seed=42, n_iters=100, n_seed=1)

    mdp = grid_mdp_factory()

    key = jrd.PRNGKey(loop_args.seed)

    vmap_init = jax.vmap(vi.init, in_axes=(None, None, 0))
    init_states = vmap_init(mdp, key, gammas)

    update_fn = vi.update
    vmap_update = jax.vmap(update_fn, in_axes=(0, None, None))

    vmap_metrics = jax.vmap(compute_metrics, in_axes=(0, 0, None, None))

    final_states, metrics = loop(
        mdp=mdp,
        alg_state=init_states,
        args=loop_args,
        update_fn=vmap_update,
        metrics_fn=vmap_metrics
    )

    results = {"GridWorld": (metrics, final_states.q_vals, gammas)}

    log_multi_gamma_results(results, alg_name)
    return results


def benchmark():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Comprehensive Benchmark
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    loop_args = LoopArgs(seed=42, n_iters=100, n_seed=5)
    gamma = 0.99

    mdps = {
        "GridWorld": grid_mdp_factory(),
        "GarnetMDP": garnet_mdp_factory(
            jrd.PRNGKey(42), state_size=10, action_size=4, branch_size=2),
        "GraphMDP": graph_mdp_factory()
    }

    algs = {
        "Value Iteration": vi,
        "Nesterov VI": nesterov_vi,
        "Policy Iteration": pi
    }

    all_results = {}

    for alg_name, alg_module in algs.items():
        
        alg_results = {}
        
        for mdp_name, mdp in mdps.items():
            keys = jrd.split(jrd.PRNGKey(loop_args.seed), loop_args.n_seed)
            
            vmap_init = jax.vmap(alg_module.init, in_axes=(None, 0, None))
            init_states = vmap_init(mdp, keys, gamma)
            
            update_fn = alg_module.update
            vmap_update = jax.vmap(update_fn, in_axes=(0, None, None))
            
            vmap_metrics = jax.vmap(compute_metrics, in_axes=(0, 0, None, None))
            
            final_states, metrics = loop(
                mdp=mdp,
                alg_state=init_states,
                args=loop_args,
                update_fn=vmap_update,
                metrics_fn=vmap_metrics
            )
            
            avg_metrics = jax.tree.map(
                lambda x: jnp.mean(x, axis=1) if x.ndim > 1 else x,
                metrics
            )
            
            alg_results[mdp_name] = (avg_metrics, None)
        
        all_results[alg_name] = alg_results
    
    settings = f"{loop_args.n_seed} seeds, {loop_args.n_iters} iterations, gamma={gamma}"
    log_comprehensive_benchmark(all_results, settings)
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JAX DP benchmarks")
    parser.add_argument(
        "benchmark_type",
        choices=["vi", "multi_seed_vi", "multi_gamma_vi", "benchmark"],
        help="Type of benchmark to run"
    )
    
    args = parser.parse_args()
    
    if args.benchmark_type == "vi":
        value_iteration_grid_world()
    elif args.benchmark_type == "multi_seed_vi":
        value_iteration_multi_seed()
    elif args.benchmark_type == "multi_gamma_vi":
        value_iteration_multi_gamma()
    elif args.benchmark_type == "benchmark":
        benchmark()
