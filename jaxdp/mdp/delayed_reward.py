"""Delayed-reward tree MDP factory."""

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd

from jaxdp.mdp.mdp import MDP


def delayed_reward_mdp(
    delay: int,
    action_size: int,
    reward_std: float,
    key: chex.PRNGKey,
) -> MDP:
    """Create a deterministic tree whose payoff is revealed at its leaves.

    The first root branch has mean payoff ``1`` and every other root branch has mean payoff ``-1``.
    Independent leaf noise has standard deviation ``reward_std``.

    Args:
        delay: Nonnegative tree depth.
        action_size: Positive branching factor and action count.
        reward_std: Nonnegative standard deviation of leaf rewards.
        key: Key used to sample leaf rewards.

    Returns:
        Delayed-reward MDP.
    """
    chex.assert_type(
        [delay, action_size],
        int,
        custom_message="delay and action_size must be integers",
    )
    chex.assert_scalar_non_negative(delay, custom_message="delay must be nonnegative")
    chex.assert_scalar_positive(action_size, custom_message="action_size must be positive")

    reward_scale = jnp.asarray(reward_std)
    chex.assert_shape(reward_scale, (), custom_message="reward_std must be scalar")
    chex.assert_tree_all_finite(reward_scale, custom_message="reward_std must be finite")
    chex.assert_trees_all_equal(
        reward_scale >= 0,
        jnp.asarray(True),
        custom_message="reward_std must be nonnegative",
    )

    state_size = sum(action_size**level for level in range(delay + 1))
    leaf_size = action_size**delay
    pre_leaf_size = action_size ** (delay - 1) if delay > 0 else 0

    state = jnp.arange(state_size)
    action = jnp.arange(action_size)[:, None]
    terminal = state >= state_size - leaf_size
    next_state = jnp.where(
        terminal,
        state,
        state * action_size + action + 1,
    )
    transition = jax.nn.one_hot(next_state, state_size, axis=-2)

    reward_size = action_size * pre_leaf_size
    reward_mean = jnp.concatenate(
        (
            jnp.ones(pre_leaf_size),
            -jnp.ones(reward_size - pre_leaf_size),
        )
    )
    reward_value = (
        (reward_mean + reward_scale * jrd.normal(key, (reward_size,)))
        .reshape(pre_leaf_size, action_size)
        .T
    )
    pre_leaf = jnp.arange(
        state_size - leaf_size - pre_leaf_size,
        state_size - leaf_size,
    )
    next_leaf = pre_leaf * action_size + action + 1
    reward = (
        jnp.zeros((action_size, state_size, state_size))
        .at[
            action,
            pre_leaf,
            next_leaf,
        ]
        .set(reward_value)
    )

    initial = jax.nn.one_hot(0, state_size)
    mdp = MDP(transition=transition, reward=reward, initial=initial, terminal=terminal)
    mdp.validate()
    return mdp


__all__ = ["delayed_reward_mdp"]
