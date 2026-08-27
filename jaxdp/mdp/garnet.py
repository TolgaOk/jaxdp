"""Random Garnet MDP factory."""

import math

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd

from jaxdp.mdp import MDP


def garnet_mdp(
    key: chex.PRNGKey,
    state_size: int,
    action_size: int,
    branch_size: int,
    min_reward: float = 0.0,
    max_reward: float = 1.0,
) -> MDP:
    """Create a random finite MDP with a fixed number of successors per state-action pair.

    Args:
        key: Key used to generate transitions and rewards.
        state_size: Positive number of states.
        action_size: Positive number of actions.
        branch_size: Maximum number of distinct successors per state-action pair.
        min_reward: Inclusive lower reward bound.
        max_reward: Inclusive upper reward bound.

    Returns:
        Random Garnet MDP.
    """
    if state_size < 1:
        raise ValueError("state_size must be positive")
    if action_size < 1:
        raise ValueError("action_size must be positive")
    if branch_size < 1:
        raise ValueError("branch_size must be positive")
    if not math.isfinite(min_reward) or not math.isfinite(max_reward):
        raise ValueError("reward bounds must be finite")
    if min_reward > max_reward:
        raise ValueError("min_reward must not exceed max_reward")

    branch_key, transition_key, reward_key = jrd.split(key, 3)
    successor_size = min(branch_size, state_size)
    candidates = jnp.broadcast_to(
        jnp.arange(state_size),
        (action_size, state_size, state_size),
    )
    successor = jrd.permutation(
        branch_key,
        candidates,
        axis=-1,
        independent=True,
    )[..., :successor_size]
    weight = jrd.uniform(transition_key, successor.shape)
    weight = weight / jnp.sum(weight, axis=-1, keepdims=True)
    transition = jnp.einsum(
        "asbx,asb->axs",
        jax.nn.one_hot(successor, state_size),
        weight,
    )

    reward_unit = jrd.uniform(reward_key, (action_size, state_size, state_size))
    reward = min_reward + (max_reward - min_reward) * reward_unit
    initial = jnp.full(state_size, 1 / state_size)
    terminal = jnp.zeros(state_size)
    mdp = MDP(transition=transition, reward=reward, initial=initial, terminal=terminal)
    mdp.validate()
    return mdp


__all__ = ["garnet_mdp"]
