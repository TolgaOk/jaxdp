# MDP from: A First-Order Approach to Accelerated Value Iteration
#           Vineet Goyal, Julien Grand-Clément
from typing import Tuple, Any, Union, Type
import jax.numpy as jnp
import jax.random as jrd
import jax

from jaxdp.mdp import MDP


def sequential_mdp(state_size: int) -> MDP:
    # TODO: Add test
    # TODO: Add documentation
    transition = jnp.zeros((2, state_size, state_size))
    transition = transition.at[
        0, jnp.clip(jnp.arange(state_size) + 1, 0, state_size - 1), jnp.arange(state_size)].set(1)
    transition = transition.at[
        1, jnp.arange(state_size), jnp.arange(state_size)].set(1)

    terminal = jnp.zeros((state_size,))
    initial = jnp.zeros((state_size,)).at[0].set(1)
    reward = jnp.zeros((2, state_size)).at[0, state_size-2].set(1)

    return MDP(transition, reward, initial, terminal, name=f"SequentialMDP")
