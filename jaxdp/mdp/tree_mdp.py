"""Deterministic binary-tree MDP factory."""

import jax
import jax.numpy as jnp

from jaxdp.mdp import MDP


def tree_mdp(depth: int) -> MDP:
    """Create a binary tree with rewarded outer leaves.

    Args:
        depth: Positive number of transitions from root to leaf.

    Returns:
        Binary-tree MDP with terminal absorbing leaves.
    """
    if depth < 1:
        raise ValueError("depth must be positive")

    state_size = 2 ** (depth + 1) - 1
    non_leaf_size = 2**depth - 1
    state = jnp.arange(state_size)
    terminal = state >= non_leaf_size
    next_state = jnp.stack(
        (
            jnp.where(terminal, state, 2 * state + 1),
            jnp.where(terminal, state, 2 * state + 2),
        )
    )
    transition = jax.nn.one_hot(next_state, state_size, axis=-2)

    reward = (
        jnp.zeros((2, state_size, state_size))
        .at[0, 2 ** (depth - 1) - 1, 2**depth - 1]
        .set(1.0)
        .at[1, 2**depth - 2, state_size - 1]
        .set(0.5)
    )
    initial = jax.nn.one_hot(0, state_size)
    mdp = MDP(transition=transition, reward=reward, initial=initial, terminal=terminal)
    mdp.validate()
    return mdp


__all__ = ["tree_mdp"]
