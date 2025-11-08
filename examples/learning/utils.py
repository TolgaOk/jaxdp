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
    table.add_column("Last 20 Eps", justify="right", style="white")
    table.add_column("Episodes", justify="right", style="white")
    table.add_column("Eval Return", justify="right", style="white")

    for mdp_name, (metrics, q_vals) in results.items():
        bellman_err = float(metrics.bellman_err[-1])
        max_linf = float(jnp.max(metrics.linf))
        n_updates = int(jnp.sum(metrics.linf > 0))

        # Episode statistics - flatten and filter NaN values
        # ep_return shape: [n_steps, n_envs]
        ep_returns_flat = metrics.ep_return.flatten()
        ep_returns = ep_returns_flat[~jnp.isnan(ep_returns_flat)]
        n_episodes = len(ep_returns)
        last_20_return = float(jnp.mean(ep_returns[-20:])) if n_episodes >= 20 else (
            float(jnp.mean(ep_returns)) if n_episodes > 0 else 0.0
        )

        # Evaluation results - get last non-NaN value
        eval_returns = metrics.eval_mean_return[~jnp.isnan(metrics.eval_mean_return)]
        eval_return_str = f"{float(eval_returns[-1]):.3f}" if len(eval_returns) > 0 else "N/A"

        table.add_row(
            mdp_name,
            f"{bellman_err:.6f}",
            f"{max_linf:.6f}",
            f"{n_updates}",
            f"{last_20_return:.3f}",
            f"{n_episodes}",
            eval_return_str
        )

    console.print()
    console.print(table)
    console.print()
