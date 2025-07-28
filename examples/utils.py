import jax.numpy as jnp
from typing import Dict, Tuple, Any
from loop import LoopState


def display_results(benchmark_results: Dict[str, Tuple[LoopState, jnp.ndarray]], algorithm_name: str = "Value Iteration") -> None:
    """Display value iteration results with focus on convergence metrics."""
    # Ensure algorithm name doesn't exceed 30 characters
    algorithm_name = algorithm_name[:30]
    
    print()
    print("┌" + "─" * 58 + "┐")
    print("│" + f" {algorithm_name.upper()} BENCHMARK RESULTS".center(58) + "│")
    print("├" + "─" * 58 + "┤")
    
    for mdp_name, (metrics, final_q_values) in benchmark_results.items():
        bellman_error = float(metrics.bellman_err[-1])
        value_error = float(metrics.linf[-1])  # L-infinity norm as value error
        
        print(f"│ {mdp_name:<18} │ Bellman: {bellman_error:8.6f} │ Value: {value_error:8.6f} │")
    
    print("└" + "─" * 58 + "┘")
    print()


def display_arguments(loop_args, alg_args=None, algorithm_name: str = "Value Iteration") -> None:
    """Display loop and algorithm arguments in a clean ASCII table."""
    # Ensure algorithm name doesn't exceed 30 characters
    algorithm_name = algorithm_name[:30]
    
    # Extract loop parameters (numeric only) from loop_args
    loop_params = {}
    for attr_name in dir(loop_args):
        if not attr_name.startswith('_') and not callable(getattr(loop_args, attr_name)):
            attr_value = getattr(loop_args, attr_name)
            if isinstance(attr_value, (int, float)):
                display_name = attr_name.replace('_', ' ').title()
                loop_params[display_name] = attr_value
    
    # Extract algorithm parameters from alg_args if provided
    alg_params = {}
    if alg_args:
        for attr_name in dir(alg_args):
            if not attr_name.startswith('_') and not callable(getattr(alg_args, attr_name)):
                attr_value = getattr(alg_args, attr_name)
                if isinstance(attr_value, (int, float)):
                    display_name = attr_name.replace('_', ' ').title()
                    alg_params[display_name] = attr_value
    
    # Get parameter lists
    loop_items = list(loop_params.items())
    alg_items = list(alg_params.items())
    
    def format_parameter_value(param_value: Any, width: int) -> str:
        """Format a parameter value with appropriate precision and width."""
        if isinstance(param_value, float):
            if param_value < 1e-3 or param_value > 1e3:
                return f"{param_value:>{width}.2e}"
            else:
                return f"{param_value:>{width}.6f}"
        else:
            return f"{param_value:>{width}}"
    
    def format_column_content(param_name: str, param_value: Any, total_width: int, value_width: int) -> str:
        """Format content for a table column with proper alignment."""
        name_section_width = total_width - value_width - 2  # -2 for leading/trailing spaces
        spaces_after_colon = max(1, name_section_width - len(param_name) - 1)  # -1 for colon
        formatted_value = format_parameter_value(param_value, value_width)
        return f" {param_name}:{' ' * spaces_after_colon}{formatted_value} "
    
    print()
    print("┌" + "─" * 70 + "┐")
    print("│" + f" {algorithm_name.upper()} CONFIGURATION".center(70) + "│")
    print("├" + "─" * 34 + "┬" + "─" * 35 + "┤")
    print("│" + " LOOP".center(34) + "│" + " ALGORITHM".center(35) + "│")
    print("├" + "─" * 34 + "┼" + "─" * 35 + "┤")
    
    # Display rows with parameters from both columns
    max_rows = max(len(loop_items), len(alg_items)) if loop_items or alg_items else 1
    
    for i in range(max_rows):
        # Format left column (loop parameters)
        left_content = (" " * 34 if i >= len(loop_items) 
                       else format_column_content(loop_items[i][0], loop_items[i][1], 34, 8))
        
        # Format right column (algorithm parameters) 
        right_content = (" " * 35 if i >= len(alg_items)
                        else format_column_content(alg_items[i][0], alg_items[i][1], 35, 9))
        
        print(f"│{left_content}│{right_content}│")
    
    print("└" + "─" * 34 + "┴" + "─" * 35 + "┘")
    print()


def create_progress_callback(name: str, max_iters: int):
    """Create a progress callback for a specific MDP."""
    def callback(iteration):
        # Convert JAX array to Python int
        iter_int = int(iteration) if hasattr(iteration, 'item') else iteration
        progress = iter_int / max_iters
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        percent = progress * 100
        print(f'\r{name} Progress: |{bar}| {percent:.1f}% ({iter_int}/{max_iters})', end='', flush=True)
        if iter_int == max_iters - 1:  # Last iteration (0-indexed)
            # Show final 100% completion
            bar_full = '█' * bar_length
            print(f'\r{name} Progress: |{bar_full}| 100.0% ({max_iters}/{max_iters})', end='', flush=True)
            print()  # New line when complete
    return callback
