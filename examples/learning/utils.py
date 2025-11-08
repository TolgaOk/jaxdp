from rich.console import Console
from rich.table import Table
import jax.numpy as jnp


def log_results(results, alg_name):
    console = Console()

    table = Table(
        title=f"[bold]{alg_name.upper()}[/bold]",
        show_header=True,
        header_style="bold white",
        border_style="white",
        title_style="bold white"
    )
    table.add_column("MDP", style="bold white", width=20)
    table.add_column("Bellman Error", justify="right", style="white")
    table.add_column("Max L-inf", justify="right", style="white")
    table.add_column("Updates", justify="right", style="white")
    table.add_column("Mean Return", justify="right", style="white")
    table.add_column("Episodes", justify="right", style="white")

    for mdp_name, (metrics, q_vals) in results.items():
        bellman_err = float(metrics.bellman_err[-1])
        max_linf = float(jnp.max(metrics.linf))
        n_updates = int(jnp.sum(metrics.linf > 0))

        # Episode statistics
        ep_mask = metrics.ep_return != 0
        n_episodes = int(jnp.sum(ep_mask))
        mean_return = float(jnp.mean(metrics.ep_return[ep_mask])) if n_episodes > 0 else 0.0

        table.add_row(
            mdp_name,
            f"{bellman_err:.6f}",
            f"{max_linf:.6f}",
            f"{n_updates}",
            f"{mean_return:.3f}",
            f"{n_episodes}"
        )

    console.print()
    console.print(table)
    console.print()
