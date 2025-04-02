from typing import List
from itertools import chain
import warnings
import jax
import jax.numpy as jnp

from jaxdp import mdp
from jaxdp.mdp import MDP


# Example board
board = [
    "#####",
    "#  @#",
    "# #X#",
    "#P  #",
    "#####"
]

char_map = {item: ord(item) for item in "# P@X+="}


def _numerical_board(board):
    """Turn the string board into a numerical board."""
    num_board = []
    for row in board:
        num_board.append([char_map[item] for item in row])
    return num_board


def _flatten_state(board, indices, char):
    """Flatten the GridWorld board based on the character."""
    return (board[indices[:, 0], indices[:, 1]] == char_map[char]).astype("float")


def grid_world(board: List[str], p_slip: float = 0.0) -> MDP:
    """
    Constructs a Markov Decision Process (MDP) for a grid world environment.

    This grid world is defined by a 2D board of characters, where:
      - "#" represents impassable cells.
      - "P" marks the initial (agent) state.
      - "@" marks terminal/goal cells (with positive reward).
      - "=" marks absorbing cells (with positive reward).
      - "+" marks cells with a positive reward.
      - "X" marks cells with a penalty.
      - " " represents regular passable space.

    The state space is composed of all passable positions (non-"#") on the board.
    The action space consists of four actions:
      0. Move Down
      1. Move Right
      2. Move Up
      3. Move Left

    When an action is chosen, there is a probability p_slip of slipping to a random
    alternative movement. This slip is implemented as an average over specific
    alternative actions, effectively mixing the intended transition with “slipped”
    transitions.

    The returned MDP includes:
      • transition: a (A, S, S) array, specifying probabilities of moving from one
        state (S) to another for each of the 4 actions.
      • reward: a (A, S, S) array, giving the reward of taking an action from each
        state and ending in another state. Goal states ("@") provide a reward,
        while penalty states ("X") cause a negative reward. The reward is calculated
        based on the next state in the transition.
      • initial: a vector indicating the initial state distribution (based on "P").
      • terminal: a vector indicating terminal states ("@"), which become absorbing
        once reached.

    Parameters:
    -----------
    board : List[str]
        A list of strings representing the 2D layout of the grid world.
    p_slip : float, default=0.0
        Probability of slipping to an unintended action.

    Returns:
    --------
    MDP
        The constructed Markov Decision Process with defined transitions, rewards,
        initial and terminal states for the given grid world.
    """
    # TODO: Add test

    state_size = sum(item != "#" for item in chain(*board))
    action_size = 4

    board_width = len(board[0])

    board = jnp.array(_numerical_board(board))
    passable_index = jnp.argwhere(board != char_map["#"])
    state_index_map = jnp.cumsum(board != char_map["#"]) - 1

    _transition = jnp.zeros((action_size, state_size, state_size))
    terminal = _flatten_state(board, passable_index, "@")
    absorbing = _flatten_state(board, passable_index, "=")
    initial = _flatten_state(board, passable_index, "P")
    reward_state = _flatten_state(board, passable_index, "+")
    penalty = _flatten_state(board, passable_index, "X")
    _reward = ((terminal + reward_state + absorbing - penalty)
               .reshape(1, 1, -1)
               .repeat(action_size, 0)
               .repeat(state_size, 1))
    if reward_state.sum() > 0 and terminal.sum() > 0:
        warnings.warn("The agent may not want to terminate due to existing + cells!")

    reward = jnp.einsum("asx,s->asx", _reward, (1 - terminal))

    for act_ind, move in enumerate([[1, 0], [0, 1], [-1, 0], [0, -1]]):
        move_index = passable_index + jnp.array(move).reshape(1, -1)
        violations = (jnp.logical_or(
            _flatten_state(board, move_index, "#"),
            jnp.logical_or(terminal, absorbing)  # To make sure terminal state is a sink state
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
