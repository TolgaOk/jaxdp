"""Sequential finite MDP factory."""

import chex
import jax.numpy as jnp

from jaxdp.mdp import MDP


def sequential_mdp(state_size: int) -> MDP:
    """Create the sequential MDP from accelerated value-iteration studies.

    Args:
        state_size: Positive number of states.

    Returns:
        Sequential MDP with advance and stay actions.
    """
    chex.assert_type(state_size, int, custom_message="state_size must be an integer")
    chex.assert_scalar_positive(state_size, custom_message="state_size must be positive")

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
    mdp = MDP(transition=transition, reward=reward, initial=initial, terminal=terminal)
    mdp.validate()
    return mdp


__all__ = ["sequential_mdp"]
