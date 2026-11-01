"""SysAdmin network benchmarks from the released DBN-UCRL models."""

import chex
import jax
import jax.numpy as jnp

from jaxdp.mdp.mdp import MDP


def sysadmin_mdp(machine_size: int, topology: str = "circular") -> MDP:
    """Create a continuing SysAdmin benchmark with one reboot or idle action.

    Machines update independently conditional on their current states and parents. The circular
    and three-leg variants use the distinct probability tables released with Talebi et al. (2021),
    not a shared failure model. Circular reboots succeed surely; three-leg reboots succeed with
    probability ``0.95``. Reward is the fraction of machines currently working, with no reboot cost.

    State indices encode binary machine states with machine zero as the most significant bit.
    Action ``i < machine_size`` reboots machine ``i``; action ``machine_size`` is idle. Initially
    every machine has failed. There are no terminal states.

    Args:
        machine_size: Positive machine count. Three-leg networks require ``3 * leg_size + 1``
            machines with positive ``leg_size``. Dense storage grows exponentially in this count.
        topology: ``"circular"`` or ``"three-leg"``. Fix both arguments under JAX transformations.

    Returns:
        MDP with ``2**machine_size`` states and ``machine_size + 1`` actions.

    Raises:
        ValueError: If ``topology`` is unknown.
    """
    chex.assert_type(machine_size, int, custom_message="machine_size must be an integer")
    chex.assert_scalar_positive(machine_size, custom_message="machine_size must be positive")

    if topology == "circular":
        parent = (jnp.arange(machine_size) - 1) % machine_size
        probability_table = jnp.asarray(((0.0238, 0.475), (0.0475, 0.95)))
        reboot_probability = 1.0
    elif topology == "three-leg":
        chex.assert_scalar_positive(
            machine_size - 1,
            custom_message="three-leg topology requires three nonempty legs",
        )
        chex.assert_trees_all_equal(
            (machine_size - 1) % 3,
            0,
            custom_message="three-leg topology requires machine_size = 3 * leg_size + 1",
        )
        parent = jnp.maximum(0, jnp.arange(machine_size) - 3)
        probability_table = jnp.asarray(((0.01, 0.67), (0.01, 0.90)))
        reboot_probability = 0.95
    else:
        raise ValueError(f"unknown SysAdmin topology {topology!r}; choose circular or three-leg")

    state_size = 2**machine_size
    machine = jnp.arange(machine_size)
    working = (jnp.arange(state_size)[:, None] >> (machine_size - 1 - machine)) & 1
    probability = probability_table[working[:, parent], working]
    probability = jnp.where(
        jnp.arange(machine_size + 1)[:, None, None] == machine[None, None, :],
        reboot_probability,
        probability[None, :, :],
    )
    transition = jnp.prod(
        jnp.where(
            working[None, :, None, :] == 1,
            probability[:, None, :, :],
            1 - probability[:, None, :, :],
        ),
        axis=-1,
    )
    reward = jnp.broadcast_to(jnp.mean(working, axis=-1)[None, :, None], transition.shape)
    initial = jax.nn.one_hot(0, state_size)
    terminal = jnp.zeros(state_size, dtype=bool)
    mdp = MDP(transition=transition, reward=reward, initial=initial, terminal=terminal)
    mdp.validate()
    return mdp


__all__ = ["sysadmin_mdp"]
