from rich.console import Console
from rich.table import Table
import jax.numpy as jnp


def log_results(results, alg_name):
    console = Console()

    table = Table(
        title=f"[bold]{alg_name.upper()} [/bold]",
        show_header=True,
        header_style="bold white",
        border_style="white",
        title_style="bold white"
    )
    table.add_column("MDP", style="bold white", width=20)
    table.add_column("Bellman Error", justify="right", style="white")
    table.add_column("Final L-inf", justify="right", style="white")

    for mdp_name, (metrics, final_state) in results.items():
        bellman_err = float(metrics.bellman_err[-1])
        final_linf = float(metrics.linf[-1])

        table.add_row(
            mdp_name,
            f"{bellman_err:.6f}",
            f"{final_linf:.6f}"
        )

    console.print()
    console.print(table)
    console.print()


def log_learning_progress(metrics, final_state, n_steps, print_every=500):
    """Print learning progress at intervals"""
    console = Console()

    console.print(f"\n[bold white]Learning Progress Summary[/bold white]")
    console.print(f"[white]Total steps: {n_steps}[/white]")

    # Sample metrics at intervals
    indices = jnp.arange(0, n_steps, print_every)

    table = Table(show_header=True, header_style="bold white", border_style="white")
    table.add_column("Step", justify="right", style="white")
    table.add_column("Bellman Err", justify="right", style="white")
    table.add_column("L-inf", justify="right", style="white")
    table.add_column("L1", justify="right", style="white")

    for idx in indices:
        if idx < len(metrics.bellman_err):
            bellman = float(metrics.bellman_err[idx])
            linf = float(metrics.linf[idx])
            l1 = float(metrics.l1[idx])

            table.add_row(
                f"{int(idx)}",
                f"{bellman:.6f}",
                f"{linf:.6f}",
                f"{l1:.6f}"
            )

    console.print(table)
    console.print()
