"""Sequential finite MDP factory."""

import jax.numpy as jnp

from jaxdp.mdp import Mdp


def sequential_mdp(state_size: int) -> Mdp:
    """Create the sequential MDP from accelerated value-iteration studies.

    Args:
        state_size: Positive number of states.

    Returns:
        Sequential MDP with advance and stay actions.
    """
    if state_size < 1:
        raise ValueError("state_size must be positive")

    state = jnp.arange(state_size)
    transition = (
        jnp.zeros((2, state_size, state_size))
        .at[0, jnp.clip(state + 1, 0, state_size - 1), state]
        .set(1.0)
        .at[1, state, state]
        .set(1.0)
    )
    advance_reward = jnp.broadcast_to(
        (state == state_size - 2)[:, None],
        (state_size, state_size),
    ).astype(transition.dtype)
    reward = jnp.stack((advance_reward, jnp.zeros_like(advance_reward)))

    initial = jnp.zeros(state_size).at[0].set(1.0)
    terminal = jnp.zeros(state_size)
    return Mdp(transition, reward, initial, terminal)


__all__ = ["sequential_mdp"]
