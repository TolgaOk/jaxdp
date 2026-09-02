"""Random Garnet MDP factory."""

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd

from jaxdp.mdp.mdp import MDP


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
    chex.assert_type(
        [state_size, action_size, branch_size],
        int,
        custom_message="state_size, action_size, and branch_size must be integers",
    )
    chex.assert_scalar_positive(state_size, custom_message="state_size must be positive")
    chex.assert_scalar_positive(action_size, custom_message="action_size must be positive")
    chex.assert_scalar_positive(branch_size, custom_message="branch_size must be positive")

    reward_bounds = jnp.asarray((min_reward, max_reward))
    chex.assert_tree_all_finite(reward_bounds, custom_message="reward bounds must be finite")
    chex.assert_trees_all_equal(
        reward_bounds[0] <= reward_bounds[1],
        jnp.asarray(True),
        custom_message="min_reward must not exceed max_reward",
    )

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
