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


def pos_to_states(board):
    _numerical_board(board)
    counter = 0


def flatten_state(board, indices, char):
    return (board[indices[:, 0], indices[:, 1]] == char_map[char]).astype("float32")


def grid_world(board):
    state_size = sum(item != "#" for item in chain(*board))
    action_size = 4

    transition = jnp.zeros((action_size, state_size, state_size))
    reward = jnp.zeros((action_size, state_size))
    initial = jnp.zeros((state_size,))
    terminal = jnp.zeros((state_size,))

    board_width = len(board[0])

    board = jnp.array(_numerical_board(board))
    passable_index = jnp.argwhere(board != char_map["#"])
    state_index_map = jnp.cumsum(board != char_map["#"]) - 1

    terminal = flatten_state(board, passable_index, "@")
    initial = flatten_state(board, passable_index, "P")

    for act_ind, move in enumerate([[1, 0], [0, 1], [-1, 0], [0, -1]]):
        move_index = passable_index + jnp.array(move).reshape(1, -1)
        violations = (jnp.logical_or(
            flatten_state(board, move_index, "#"),
            terminal  # To make sure terminal state is a sink state
        ).reshape(-1, 1))
        move_index = ((1 - violations) * move_index +
                      violations * passable_index).astype("int32")
        flat_index = move_index[:, 0] * board_width + move_index[:, 1]

        state_index = state_index_map[flat_index]
        transition = transition.at[act_ind].set(
            jax.nn.one_hot(state_index, num_classes=state_size).T)

        goal = flatten_state(board, move_index, "@")
        penalty = flatten_state(board, move_index, "X")
        reward = reward.at[act_ind].set((goal - penalty) * (1 - terminal))

    return MDP(transition, reward, initial, terminal, "GridWorld")
