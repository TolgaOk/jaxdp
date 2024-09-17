from typing import List
import jax.numpy as jnp
from itertools import chain
import jax

from jaxdp.mdp import MDP


board = [
    "#####",
    "#  @#",
    "# #X#",
    "#P  #",
    "#####"
]

char_map = {item: ord(item) for item in "# P@X"}


def _numerical_board(board):
    num_board = []
    for row in board:
        num_board.append([char_map[item] for item in row])
    return num_board


def _flatten_state(board, indices, char):
    return (board[indices[:, 0], indices[:, 1]] == char_map[char]).astype("float")


def grid_world(board: List[str], p_slip: float = 0.0) -> MDP:
    # TODO: Add test
    # TODO: Add documentation
    state_size = sum(item != "#" for item in chain(*board))
    action_size = 4


    board_width = len(board[0])

    board = jnp.array(_numerical_board(board))
    passable_index = jnp.argwhere(board != char_map["#"])
    state_index_map = jnp.cumsum(board != char_map["#"]) - 1

    _transition = jnp.zeros((action_size, state_size, state_size))
    terminal = _flatten_state(board, passable_index, "@")
    initial = _flatten_state(board, passable_index, "P")
    goal = _flatten_state(board, passable_index, "@")
    penalty = _flatten_state(board, passable_index, "X")
    _reward = (goal - penalty).reshape(1, 1, -1).repeat(action_size, 0).repeat(state_size, 1)

    reward = jnp.einsum("asx,s->asx", _reward, (1 - terminal)) 

    for act_ind, move in enumerate([[1, 0], [0, 1], [-1, 0], [0, -1]]):
        move_index = passable_index + jnp.array(move).reshape(1, -1)
        violations = (jnp.logical_or(
            _flatten_state(board, move_index, "#"),
            terminal  # To make sure terminal state is a sink state
        ).reshape(-1, 1))
        move_index = ((1 - violations) * move_index +
                      violations * passable_index).astype("int32")
        flat_index = move_index[:, 0] * board_width + move_index[:, 1]

        state_index = state_index_map[flat_index]
        _transition = _transition.at[act_ind].set(
            jax.nn.one_hot(state_index, num_classes=state_size).T)

    transition = jnp.zeros_like(_transition)
    for act_ind, slip_ind in enumerate([[1, 3], [0, 2], [1, 3], [0, 2]]):
        transition = transition.at[act_ind].set(
            (1 - p_slip) * _transition[act_ind] +
            p_slip * _transition[jnp.array(slip_ind)].mean(0))

    return MDP(transition, reward, initial, terminal, name="GridWorld")
