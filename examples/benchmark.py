import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct
from typing import Callable, Dict, Tuple, Any

from jaxdp.mdp.grid_world import grid_world
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.mdp.simple_graph import graph_mdp
from jaxdp.mdp import MDP

from loop import loop, LoopArgs, LoopState
from algorithms import vi
from utils import display_results, display_arguments, create_progress_callback

# Enable float64 for better numerical precision
jax.config.update("jax_enable_x64", True)


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


def run_value_iteration_benchmark():
    """Run value iteration on multiple MDPs and return benchmark state."""
    
    # Set up arguments
    algorithm_name = "Value Iteration"
    alg_args = vi.Args(gamma=0.99)
    loop_args = LoopArgs(seed=42, max_iters=100, gamma=0.99, tolerance=1e-6)
    
    # Display configuration
    display_arguments(loop_args, alg_args, algorithm_name)
    
    # Create MDPs
    mdps_to_benchmark = {
        "GridWorld": grid_mdp_factory(),
        "GarnetMDP": garnet_mdp_factory(jrd.PRNGKey(42), state_size=10, action_size=4, branch_size=2),
        "GraphMDP": graph_mdp_factory()
    }
    
    # Run benchmarks
    benchmark_results = {}
    for mdp_name, mdp in mdps_to_benchmark.items():
        final_state, all_loop_states = loop(
            mdp=mdp, 
            update_fn=vi.update,
            init_fn=vi.init,
            args=loop_args,
            alg_args=alg_args,
            progress_callback=create_progress_callback(mdp_name, loop_args.max_iters)
        )
        
        # Report convergence
        actual_iterations = int(all_loop_states.iteration[-1])
        if actual_iterations < loop_args.max_iters:
            print(f"  -> Converged after {actual_iterations} iterations")
        
        benchmark_results[mdp_name] = (all_loop_states, final_state.q_values)
    
    # Display results
    display_results(benchmark_results, algorithm_name)
    return benchmark_results


if __name__ == "__main__":
    benchmark_state = run_value_iteration_benchmark()



