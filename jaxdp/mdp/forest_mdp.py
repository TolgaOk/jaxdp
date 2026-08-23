"""Forest-management MDP factory."""

import jax.numpy as jnp

from jaxdp.mdp import Mdp


def forest_mdp(rotation: int) -> Mdp:
    """Create a forest MDP with wait and harvest actions.

    Args:
        rotation: Nonnegative maximum forest age.

    Returns:
        Forest-management MDP initialized at age zero.
    """
    if rotation < 0:
        raise ValueError("rotation must be nonnegative")

    state_size = rotation + 1
    state = jnp.arange(state_size)
    next_age = jnp.minimum(state + 1, rotation)
    transition = (
        jnp.zeros((2, state_size, state_size))
        .at[0, next_age, state]
        .set(1.0)
        .at[1, 0, state]
        .set(1.0)
    )
    reward = jnp.zeros_like(transition).at[1, state, 0].set(state)
    initial = jnp.zeros(state_size).at[0].set(1.0)
    terminal = jnp.zeros(state_size)
    return Mdp(transition, reward, initial, terminal)


__all__ = ["forest_mdp"]
