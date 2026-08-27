"""Character-grid MDP factory."""

from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp

from jaxdp.mdp import MDP

_ACTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_SLIP_ACTIONS = ((1, 3), (0, 2), (1, 3), (0, 2))
_VALID_CELLS = frozenset("# P@X+=")


def grid_world(board: Sequence[str], p_slip: float = 0.0) -> MDP:
    """Create a four-action MDP from a character grid.

    ``P`` is the initial cell, ``@`` is a terminal goal, ``=`` is a nonterminal absorbing
    reward cell, ``+`` gives reward one, ``X`` gives reward minus one, and ``#`` is a wall.
    Rewards depend on the destination cell. Slip probability is divided equally between the two
    perpendicular actions.

    Args:
        board: Rectangular rows containing supported cell characters.
        p_slip: Probability of taking a perpendicular action.

    Returns:
        Grid-world MDP in row-major state order.
    """
    rows = _validate_board(board, p_slip)
    positions = tuple(
        (row, column)
        for row, cells in enumerate(rows)
        for column, cell in enumerate(cells)
        if cell != "#"
    )
    state_index = {position: index for index, position in enumerate(positions)}
    state_size = len(positions)

    base_transition = jnp.zeros((len(_ACTIONS), state_size, state_size))
    for current, (row, column) in enumerate(positions):
        cell = rows[row][column]
        for action, (row_step, column_step) in enumerate(_ACTIONS):
            target = (row + row_step, column + column_step)
            next_state = current if cell in "@=" else state_index.get(target, current)
            base_transition = base_transition.at[action, next_state, current].set(1.0)

    slip_transition = jnp.stack(
        tuple(jnp.mean(base_transition[jnp.array(actions)], axis=0) for actions in _SLIP_ACTIONS)
    )
    transition = (1 - p_slip) * base_transition + p_slip * slip_transition

    cell_reward = {"@": 1.0, "=": 1.0, "+": 1.0, "X": -1.0}
    destination_reward = jnp.array(
        [cell_reward.get(rows[row][column], 0.0) for row, column in positions]
    )
    terminal = jnp.array([rows[row][column] == "@" for row, column in positions])
    reward = jnp.broadcast_to(destination_reward, transition.shape)
    reward = reward * (1 - terminal)[None, :, None]

    initial_index = next(
        index for index, (row, column) in enumerate(positions) if rows[row][column] == "P"
    )
    initial = jax.nn.one_hot(initial_index, state_size)
    mdp = MDP(transition=transition, reward=reward, initial=initial, terminal=terminal)
    mdp.validate()
    return mdp


def _validate_board(board: Sequence[str], p_slip: float) -> tuple[str, ...]:
    if isinstance(board, str) or not board:
        raise ValueError("board must be a nonempty sequence of rows")
    if any(not isinstance(row, str) or not row for row in board):
        raise ValueError("board rows must be nonempty strings")
    if any(len(row) != len(board[0]) for row in board):
        raise ValueError("board rows must have equal length")

    invalid = set().union(*(set(row) for row in board)) - _VALID_CELLS
    if invalid:
        raise ValueError(f"board contains invalid cells: {sorted(invalid)}")
    if sum(row.count("P") for row in board) != 1:
        raise ValueError("board must contain exactly one initial cell 'P'")
    slip = jnp.asarray(p_slip)
    chex.assert_shape(slip, (), custom_message="p_slip must be scalar")
    chex.assert_tree_all_finite(slip, custom_message="p_slip must be finite")
    chex.assert_trees_all_equal(
        (slip >= 0) & (slip <= 1),
        jnp.asarray(True),
        custom_message="p_slip must be in [0, 1]",
    )
    return tuple(board)


__all__ = ["grid_world"]
