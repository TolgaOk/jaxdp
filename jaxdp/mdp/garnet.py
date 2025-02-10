from typing import Tuple, Any, Union, Type
import jax.numpy as jnp
import jax.random as jrd
import jax

from jaxdp.mdp import MDP
from jax.typing import ArrayLike as KeyType


def garnet_mdp(state_size: int, action_size: int, branch_size: int, key: KeyType,
               min_reward: float = 0, max_reward: float = 1.0) -> MDP:
    """
    Constructs a Garnet MDP.

    Garnet MDPs are randomly generated MDPs characterized by:
      - A specified number of states (state_size) and actions (action_size).
      - Each state-action pair has a fixed number of successor states defined by branch_size.
      - Transitions are stochastic and generated using the provided random key.
      - Rewards are assigned randomly within the interval [min_reward, max_reward].
      
    Args:
        state_size (int): Number of states.
        action_size (int): Number of actions.
        branch_size (int): Number of successor states per state-action pair.
        key (KeyType): JAX random key for generating transitions and rewards.
        min_reward (float): Minimum reward value.
        max_reward (float): Maximum reward value.
        
    Returns:
        MDP: The constructed Garnet MDP.
    """
    # TODO: Make key the first argument
    # TODO: Add test
    branch_key, transition_key, reward_key = jrd.split(key, 3)
    transition = jnp.zeros((action_size, state_size, state_size))

    indices = jrd.permutation(
        branch_key,
        jnp.tile(
            jnp.arange(state_size),
            (action_size, state_size, 1)
        ),
        axis=-1,
        independent=True
    )[:, :, :branch_size]
    raw_transition = jnp.einsum(
        "axbs,axb->axs",
        jax.nn.one_hot(indices, state_size, axis=-1),
        jrd.uniform(transition_key, indices.shape)
    )
    transition = raw_transition / raw_transition.sum(axis=-1, keepdims=True)

    transition = transition.transpose(0, 2, 1)
    terminal = jnp.zeros((state_size,))
    initial = jnp.ones((state_size,)) / state_size
    reward = jrd.uniform(reward_key, (action_size, state_size, state_size),
                         minval=min_reward, maxval=max_reward)

    return MDP(transition, reward, initial, terminal,
               name=f"GarnetMDP[#branch={branch_size}]")
