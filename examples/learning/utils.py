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
    table.add_column("Max L-inf", justify="right", style="white")
    table.add_column("Updates", justify="right", style="white")
    table.add_column("Mean Ep. Reward", justify="right", style="white")
    table.add_column("Mean Ep. Length", justify="right", style="white")

    for mdp_name, (metrics, final_state) in results.items():
        bellman_err = float(metrics.bellman_err[-1])
        max_linf = float(jnp.max(metrics.linf))
        n_updates = int(jnp.sum(metrics.linf > 0))

        # Compute episode statistics (only for completed episodes)
        completed_episodes = metrics.episode_reward != 0
        if jnp.sum(completed_episodes) > 0:
            mean_reward = float(jnp.mean(metrics.episode_reward[completed_episodes]))
            mean_length = float(jnp.mean(metrics.episode_length[completed_episodes]))
        else:
            mean_reward = 0.0
            mean_length = 0.0

        table.add_row(
            mdp_name,
            f"{bellman_err:.6f}",
            f"{max_linf:.6f}",
            f"{n_updates}",
            f"{mean_reward:.3f}",
            f"{mean_length:.1f}"
        )

    console.print()
    console.print(table)
    console.print()


def log_learning_progress(metrics, final_state, n_steps, print_every=500):
    """Print learning progress at intervals"""
    console = Console()

    # First, show episode statistics summary
    completed_episodes = metrics.episode_reward != 0
    n_episodes = int(jnp.sum(completed_episodes))

    console.print(f"\n[bold white]Learning Progress Summary[/bold white]")
    console.print(f"[white]Total steps: {n_steps}[/white]")
    console.print(f"[white]Episodes completed: {n_episodes}[/white]")

    if n_episodes > 0:
        mean_reward = float(jnp.mean(metrics.episode_reward[completed_episodes]))
        mean_length = float(jnp.mean(metrics.episode_length[completed_episodes]))
        console.print(f"[white]Mean episode reward: {mean_reward:.3f}[/white]")
        console.print(f"[white]Mean episode length: {mean_length:.1f}[/white]")

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
