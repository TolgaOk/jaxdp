""" From the paper: `Fastest Convergence for Q-Learning`
    https://arxiv.org/pdf/1707.03770.pdf#page=21.79
"""
from typing import Dict, Tuple, Any, Union, Type, List
from functools import partial
import jax.numpy as jnp
import jax.random as jrd
import jax

from jaxdp.mdp import MDP


_edge_info = {0: [0, 4], 1: [1, 3, 5], 2: [2, 3], 3: [1, 2, 3, 4], 4: [0, 3, 4, 5], 5: [1, 4, 5]}
_state_size = 6


def _graph_mdp(state_size: int, edge_info: Dict[str, Tuple[int]]) -> MDP:
    """
    Constructs a Graph MDP from a given state size and edge information.
    
    In this environment:
      - The state space consists of a fixed number of states (state_size).
      - Transition dynamics are defined using edge_info where each state maps to a set
        of feasible actions (edges). For each state, transitions for specified actions are 
        computed by averaging a primary action with alternative actions.
      - Rewards are structured to assign a high reward to the terminal state (last state)
        and a penalty for specific transitions.
      - The initial state distribution is uniform.
      - The terminal state is indicated by specific conditions in the transition dynamics.
      
    Args:
        state_size (int): Number of states in the MDP.
        edge_info (Dict[str, Tuple[int]]): Mapping from state index to a tuple containing 
            feasible edge indices defining allowed transitions.
    
    Returns:
        MDP: The constructed Graph MDP.
    """
    transition = jnp.zeros((state_size, state_size, state_size))

    reward = (jnp.eye(state_size) - jnp.ones((state_size, state_size))) * 5
    reward = reward.at[state_size - 1, :].set(jnp.ones((state_size,)) * 100)
    reward = reward.at[4, 3].set(-100)

    eye = jnp.eye(state_size)
    for state, edges in edge_info.items():
        for edge in edges:
            rest = set(edges) - set((edge,))
            transition = transition.at[edge, :, state].set(
                sum([eye[edge] * 0.8] + [eye[_edge] * 0.2 / len(rest) for _edge in rest]))
        unfeasible = set(range(state_size)) - set(edges)
        for edge in unfeasible:
            transition = transition.at[edge, :, state].set(transition[state, :, state])
            reward = reward.at[edge, state].set(reward[state, state])

    reward = jnp.repeat(jnp.expand_dims(reward, -1), state_size, axis=-1)
    terminal = jnp.zeros((state_size,))
    initial = jnp.ones((state_size,)) / state_size

    return MDP(transition, reward / 100, initial, terminal, name=f"GraphMDP")


graph_mdp = partial(_graph_mdp, state_size=_state_size, edge_info=_edge_info)
