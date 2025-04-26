""" From the paper: `Revisiting Peng's Q(λ) for Modern Reinforcement Learning`
    https://proceedings.mlr.press/v139/kozuno21a/kozuno21a.pdf
"""
import jax.numpy as jnp
from jaxdp.mdp import MDP


def _tree_mdp(depth: int) -> MDP:
    """
    Constructs a binary tree MDP of given depth.
      - The state space consists of nodes in a complete binary tree.
        Total states: 2^(depth+1) - 1.
      - Action space: two actions, 0 for 'Left' and 1 for 'Right'.
      - For non-leaf nodes, action 0 transitions deterministically to the left child
        and action 1 transitions to the right child.
      - For leaf nodes, both actions yield a self loop (absorbing state).
      - Rewards are zero everywhere except when reaching:
            • the leftmost leaf (state index 2**depth - 1): reward 1.
            • the rightmost leaf (state index = total_states - 1): reward 0.5.
      - The initial state is the root (state 0).
      - Terminal states are the leaf nodes.

    Example of the tree structure for depth = 2:

             0
           /   \
          /     \
         /       \
        1         2
       / \       / \
      /   \     /   \
   3(+1) 4(0) 5(0) 6(+0.5)

    Args:
        depth (int): Depth of the tree (number of steps per episode).

    Returns:
        MDP: The constructed Tree MDP.
    """
    n_states = 2 ** (depth + 1) - 1
    n_action = 2
    n_non_leaf = 2 ** depth - 1

    transition = (
        jnp.zeros((n_action, n_states, n_states))
        .at[0, jnp.arange(n_non_leaf) * 2 + 1, jnp.arange(n_non_leaf)].set(1.0)
        .at[0, jnp.arange(n_non_leaf, n_states), jnp.arange(n_non_leaf, n_states)].set(1.0)
        .at[1, jnp.arange(n_non_leaf) * 2 + 2, jnp.arange(n_non_leaf)].set(1.0)
        .at[1, jnp.arange(n_non_leaf, n_states), jnp.arange(n_non_leaf, n_states)].set(1.0)
    )
    reward = (
        jnp.zeros((n_action, n_states, n_states))
        .at[0, 2 ** (depth - 1) - 1, 2 ** depth - 1].set(1.0)
        .at[1, 2 ** depth - 2, -1].set(0.5)
    )
    initial = (
        jnp.zeros(n_states)
        .at[0].set(1.0)
    )
    terminal = (
        jnp.zeros(n_states)
        .at[n_non_leaf:].set(1.0)
    )

    return MDP(transition, reward, initial, terminal, name=f"TreeMDP[depth={depth}]")


tree_mdp = _tree_mdp
