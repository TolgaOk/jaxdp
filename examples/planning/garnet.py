"""Compare finite planners by value error on a seeded Garnet MDP."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrd
import tyro

from jaxdp.mdp import MDP
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.operator import bellman_opt_op
from jaxdp.planning import (
    AcceleratedPolicyIteration,
    AnchoredValueIteration,
    AndersonValueIteration,
    DeflatedValueIteration,
    DynamicBoltzmannValueIteration,
    MomentumValueIteration,
    PIDValueIteration,
    PolicyIteration,
    QuasiPolicyIteration,
    RankOneValueIteration,
    SafeAcceleratedValueIteration,
    SafeAndersonValueIteration,
    ValueIteration,
)

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class Args:
    """Garnet value-error experiment arguments."""

    seed: int = 0
    state_size: int = 50
    action_size: int = 5
    branch_size: int = 5
    gamma: float = 0.95
    step: int = 100
    output: Path = Path("garnet_value_error.png")
    show: bool = False


class ValueState(Protocol):
    """State exposing the current state-value iterate."""

    @property
    def v_val(self) -> jax.Array: ...


StateT = TypeVar("StateT", bound=ValueState)


class Planner(Protocol[StateT]):
    """Planner that updates a value-bearing state."""

    def update(self, mdp: MDP, state: StateT) -> StateT: ...


def optimal_value(mdp: MDP, gamma: float, tolerance: float = 1e-10) -> jax.Array:
    """Return an optimal value reference from exact policy iteration."""
    policy = jnp.full(
        (mdp.action_size, mdp.state_size),
        1 / mdp.action_size,
        dtype=mdp.transition.dtype,
    )
    planner = PolicyIteration(gamma=gamma)
    state = planner.init(mdp, policy)

    for _ in range(100):
        previous_policy = state.policy
        state = planner.update(mdp, state)
        if bool(jnp.all(state.policy == previous_policy)):
            residual = jnp.max(
                jnp.abs(state.v_val - bellman_opt_op.v(mdp, state.v_val, gamma))
            )
            if float(residual) > tolerance:
                raise RuntimeError("stable policy did not produce an optimal value reference")
            return state.v_val

    raise RuntimeError("policy iteration did not stabilize within 100 updates")


def value_error(
    planner: Planner[StateT],
    mdp: MDP,
    state: StateT,
    reference: jax.Array,
    step: int,
) -> jax.Array:
    """Return the sup-norm value error before and after each planner update."""
    error = [jnp.max(jnp.abs(state.v_val - reference))]

    for _ in range(step):
        state = planner.update(mdp, state)
        error.append(jnp.max(jnp.abs(state.v_val - reference)))

    return jnp.stack(error)


def benchmark(args: Args) -> dict[str, jax.Array]:
    """Run the public value-based planners on one seeded Garnet MDP."""
    if args.step < 1:
        raise ValueError("step must be positive")
    if not 0 < args.gamma < 1:
        raise ValueError("gamma must be in (0, 1)")

    mdp = garnet_mdp(
        key=jrd.key(args.seed),
        state_size=args.state_size,
        action_size=args.action_size,
        branch_size=args.branch_size,
    )
    reference = optimal_value(mdp, args.gamma)
    initial_v = jnp.zeros((mdp.state_size,), dtype=mdp.reward.dtype)
    initial_policy = jnp.full(
        (mdp.action_size, mdp.state_size),
        1 / mdp.action_size,
        dtype=mdp.transition.dtype,
    )
    error: dict[str, jax.Array] = {}

    planner = ValueIteration(gamma=args.gamma)
    error["VI"] = value_error(planner, mdp, planner.init(mdp), reference, args.step)

    planner = AnchoredValueIteration(gamma=args.gamma)
    error["Anc-VI"] = value_error(
        planner,
        mdp,
        planner.init(mdp, initial_v),
        reference,
        args.step,
    )

    planner = SafeAcceleratedValueIteration(gamma=args.gamma)
    error["S-AVI"] = value_error(
        planner,
        mdp,
        planner.init(mdp, initial_v),
        reference,
        args.step,
    )

    planner = MomentumValueIteration(gamma=args.gamma)
    error["M-VI"] = value_error(
        planner,
        mdp,
        planner.init(mdp, initial_v),
        reference,
        args.step,
    )

    planner = PIDValueIteration(gamma=args.gamma)
    error["PID VI"] = value_error(
        planner,
        mdp,
        planner.init(mdp, initial_v),
        reference,
        args.step,
    )

    planner = AndersonValueIteration(gamma=args.gamma)
    error["Anderson VI"] = value_error(
        planner,
        mdp,
        planner.init(mdp, initial_v),
        reference,
        args.step,
    )

    planner = SafeAndersonValueIteration(gamma=args.gamma)
    error["Safe Anderson"] = value_error(
        planner,
        mdp,
        planner.init(mdp, initial_v),
        reference,
        args.step,
    )

    planner = RankOneValueIteration(gamma=args.gamma)
    error["R1-VI"] = value_error(
        planner,
        mdp,
        planner.init(mdp, initial_v),
        reference,
        args.step,
    )

    planner = DeflatedValueIteration(gamma=args.gamma)
    error["DDVI"] = value_error(
        planner,
        mdp,
        planner.init(mdp, initial_v),
        reference,
        args.step,
    )

    planner = QuasiPolicyIteration(gamma=args.gamma)
    error["QPI"] = value_error(
        planner,
        mdp,
        planner.init(mdp, initial_v),
        reference,
        args.step,
    )

    planner = DynamicBoltzmannValueIteration(gamma=args.gamma)
    error["DBS-VI"] = value_error(
        planner,
        mdp,
        planner.init(mdp, initial_v),
        reference,
        args.step,
    )

    planner = AcceleratedPolicyIteration(gamma=args.gamma)
    error["Acc-PI"] = value_error(
        planner,
        mdp,
        planner.init(mdp, initial_policy, initial_v),
        reference,
        args.step,
    )

    planner = PolicyIteration(gamma=args.gamma)
    error["PI"] = value_error(
        planner,
        mdp,
        planner.init(mdp, initial_policy),
        reference,
        args.step,
    )
    return error


def plot_value_error(error: dict[str, jax.Array], args: Args) -> None:
    """Plot sup-norm value error against planner updates."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    update = jnp.arange(args.step + 1)
    color_map = plt.get_cmap("tab20")
    color_scale = max(len(error) - 1, 1)
    conditional = {"M-VI", "Acc-PI"}

    for index, (name, values) in enumerate(error.items()):
        plotted = jnp.maximum(values, jnp.finfo(values.dtype).eps)
        axis.plot(
            update,
            plotted,
            color=color_map(index / color_scale),
            label=name,
            linestyle="--" if name in conditional else "-",
            linewidth=1.8,
        )

    axis.set(
        title=(
            f"Garnet value error: S={args.state_size}, A={args.action_size}, "
            f"B={args.branch_size}, gamma={args.gamma:g}"
        ),
        xlabel="Planner update",
        ylabel=r"$\|v_k-v^\star\|_\infty$",
        yscale="log",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(ncol=2, fontsize="small")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    if args.show:
        plt.show()
    plt.close(figure)


def print_final_error(error: dict[str, jax.Array]) -> None:
    """Print the final value error for each planner."""
    width = max(map(len, error))
    for name, values in error.items():
        print(f"{name:<{width}}  {float(values[-1]):.3e}")


def main(args: Args) -> None:
    """Run the Garnet comparison and save its value-error plot."""
    error = benchmark(args)
    print_final_error(error)
    plot_value_error(error, args)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
