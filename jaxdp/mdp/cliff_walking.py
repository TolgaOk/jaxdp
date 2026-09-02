"""Cliff-walking MDP factory."""

import chex
import jax
import jax.numpy as jnp

from jaxdp.mdp.mdp import MDP

_ACTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def cliff_walking_mdp(row_size: int = 4, column_size: int = 12) -> MDP:
    """Create a rectangular cliff-walking MDP.

    The initial and goal states occupy the lower-left and lower-right corners. Entering a cliff
    cell between them gives reward ``-100`` and returns to the initial state without terminating.
    Every other transition gives reward ``-1`` and entering the goal terminates. Cliff cells are
    transition events rather than states.

    Args:
        row_size: Number of rows, at least two.
        column_size: Number of columns, at least three.

    Returns:
        Cliff-walking MDP.
    """
    chex.assert_type(
        [row_size, column_size],
        int,
        custom_message="row_size and column_size must be integers",
    )
    chex.assert_trees_all_equal(
        jnp.asarray(row_size >= 2),
        jnp.asarray(True),
        custom_message="row_size must be at least two",
    )
    chex.assert_trees_all_equal(
        jnp.asarray(column_size >= 3),
        jnp.asarray(True),
        custom_message="column_size must be at least three",
    )

    initial_position = (row_size - 1, 0)
    goal_position = (row_size - 1, column_size - 1)
    cliff = frozenset((row_size - 1, column) for column in range(1, column_size - 1))
    positions = tuple(
        (row, column)
        for row in range(row_size)
        for column in range(column_size)
        if (row, column) not in cliff
    )
    state_index = {position: index for index, position in enumerate(positions)}
    state_size = len(positions)
    initial_index = state_index[initial_position]
    goal_index = state_index[goal_position]

    transition = jnp.zeros((len(_ACTIONS), state_size, state_size))
    reward = jnp.zeros_like(transition)
    for current, (row, column) in enumerate(positions):
        for action, (row_step, column_step) in enumerate(_ACTIONS):
            if current == goal_index:
                successor = goal_index
                reward_value = 0.0
            else:
                target = (
                    min(max(row + row_step, 0), row_size - 1),
                    min(max(column + column_step, 0), column_size - 1),
                )
                successor = initial_index if target in cliff else state_index[target]
                reward_value = -100.0 if target in cliff else -1.0
            transition = transition.at[action, successor, current].set(1.0)
            reward = reward.at[action, current, successor].set(reward_value)

    mdp = MDP(
        transition=transition,
        reward=reward,
        initial=jax.nn.one_hot(initial_index, state_size),
        terminal=jax.nn.one_hot(goal_index, state_size),
    )
    mdp.validate()
    return mdp


__all__ = ["cliff_walking_mdp"]
