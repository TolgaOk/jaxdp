from typing import Tuple, Any, Union, Type
import jax.numpy as jnp
import jax.random as jrd
import jax

from jaxdp.mdp import MDP
from jaxtyping import Float, Array
from jax.typing import ArrayLike as KeyType


def garnet_mdp(state_size: int, action_size: int, branch_size: int, key: KeyType,
               min_reward: float = 0, max_reward: float = 1.0) -> MDP:
    # TODO: Add test
    # TODO: Add documentation
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
               name=f"GarnetMDP(#branch={branch_size})")
