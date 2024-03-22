from typing import Dict, Union, List
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float

import jaxdp
from jaxdp import MDP
from jaxdp.mdp import MDP


def q_iteration_update(mdp: MDP, value: Float[Array, "A S"], gamma: float) -> Float[Array, "A S"]:
    # TODO: Add docstring
    # TODO: Add test
    non_done = (1 - mdp.terminal)
    reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
    return reward * non_done.reshape(1, -1) + gamma * jnp.einsum(
        "axs,x,x->as",
        mdp.transition,
        jnp.max(value, axis=0),
        non_done)


def value_iteration_update(mdp: MDP, value: Float[Array, "S"], gamma: float) -> Float[Array, "S"]:
    # TODO: Add docstring
    # TODO: Add test
    non_done = (1 - mdp.terminal)
    reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
    return jnp.max(reward + gamma * jnp.einsum(
        "axs,x,x->as",
        mdp.transition,
        value,
        non_done),
        axis=0) * non_done


def policy_iteration_update(mdp: MDP, value: Float[Array, "A S"], gamma: float) -> Float[Array, "A S"]:
    # TODO: Add docstring
    # TODO: Add test
    policy_pi = jaxdp.greedy_policy(value)
    return jaxdp.q_policy_evaluation(mdp, policy_pi, gamma)
