import jax.numpy as jnp
from typing import Dict, Tuple, Any
from rich.console import Console
from rich.table import Table


def log_results(results, alg_name):
    console = Console()
    
    table = Table(
        title=f"[bold]{alg_name.upper()} BENCHMARK RESULTS[/bold]",
        show_header=True,
        header_style="bold white",
        border_style="white",
        title_style="bold white"
    )
    table.add_column("MDP", style="bold white", width=20)
    table.add_column("Bellman Error", justify="right", style="white")
    table.add_column("Value Error", justify="right", style="white")
    
    for mdp_name, (metrics, _) in results.items():
        bellman_err = float(metrics.bellman_err[-1])
        value_err = float(metrics.linf[-1])
        
        table.add_row(
            mdp_name,
            f"{bellman_err:.6f}",
            f"{value_err:.6f}"
        )
    
    console.print()
    console.print(table)
    console.print()


def log_multi_gamma_results(results, alg_name):
    console = Console()

    table = Table(
        title=f"[bold]{alg_name.upper()} BENCHMARK RESULTS[/bold]",
        show_header=True,
        header_style="bold white",
        border_style="white",
        title_style="bold white"
    )
    table.add_column("MDP", style="bold white", width=12)
    table.add_column("Gamma", justify="center", style="white")
    table.add_column("Bellman Err", justify="right", style="white")
    table.add_column("L-inf", justify="right", style="white")
    table.add_column("Iterations", justify="right", style="white")

    for mdp_name, (metrics, final_q_vals, gammas) in results.items():
        for i, gamma in enumerate(gammas):
            bellman_err = float(metrics.bellman_err[-1, i])
            linf = float(metrics.linf[-1, i])
            iters = int(metrics.iteration[-1, i])

            mdp_display = mdp_name if i == 0 else ""

            table.add_row(
                mdp_display,
                f"{gamma:.3f}",
                f"{bellman_err:.6f}",
                f"{linf:.6f}",
                f"{iters:d}"
            )

    console.print()
    console.print(table)
    console.print()


def log_comprehensive_benchmark(all_results, settings):
    console = Console()
    
    alg_avg_errors = {}
    
    table = Table(
        title=f"[bold] Bellman Error [/bold]",
        show_header=True,
        header_style="bold white",
        border_style="white", 
        title_style="bold white"
    )
    
    table.add_column("Algorithm", style="bold white", width=18)
    table.add_column("GridWorld", justify="right", style="white")
    table.add_column("GarnetMDP", justify="right", style="white") 
    table.add_column("GraphMDP", justify="right", style="white")
    table.add_column("Avg Error", justify="right", style="bold white")
    
    for alg_name, alg_results in all_results.items():
        row_data = [alg_name]
        bellman_errs = []
        
        for env_name in ["GridWorld", "GarnetMDP", "GraphMDP"]:
            if env_name in alg_results:
                metrics, _ = alg_results[env_name]
                bellman_err = float(metrics.bellman_err[-1])
                bellman_errs.append(bellman_err)
                row_data.append(f"{bellman_err:.6f}")
            else:
                row_data.append("N/A")
        
        avg_err = sum(bellman_errs) / len(bellman_errs) if bellman_errs else 0.0
        alg_avg_errors[alg_name] = avg_err
        row_data.append(f"{avg_err:.6f}")
        
        table.add_row(*row_data)
    
    if settings:
        console.print(f"\n[bold white]Settings:[/bold white] {settings}")
    console.print()
    console.print(table)
    console.print()
