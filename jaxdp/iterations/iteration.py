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

def nesterov_qi_update(mdp: MDP,
                       value: Float[Array, "A S"],
                       prev_value: Float[Array, "A S"],
                       gamma: float,
                       ) -> Float[Array, "A S"]:
    # TODO: Add test
    # TODO: Add docstring
    r"""
    Evaluate the policy for each state-action pair using the true MDP
    via Nesterov accelerated VI
    """
    alpha = 1 / (1 + gamma)
    beta = (1 - jnp.sqrt(1 - gamma ** 2)) / gamma

    momentum = value + beta * (value - prev_value)
    delta = jaxdp.bellman_optimality_operator(mdp, momentum, gamma) - momentum

    return (momentum + alpha * delta.reshape(*momentum.shape)), value


def anderson_qi_update(mdp: MDP,
                       value: Float[Array, "A S"],
                       prev_value: Float[Array, "A S"],
                       gamma: float,
                       ) -> Float[Array, "A S"]:
    # TODO: Add test
    # TODO: Add docstring
    r"""
    Evaluate the policy for each state-action pair using the true MDP
    via Anderson accelerated VI
    """
    value_diff = value - prev_value
    bellman_value = jaxdp.bellman_optimality_operator(mdp, value, gamma)
    bellman_prev_value = jaxdp.bellman_optimality_operator(mdp, prev_value, gamma)
    bellman_value_diff = bellman_value - bellman_prev_value

    delta_numerator = jnp.einsum("as,as->", value_diff, bellman_value - value)
    delta_denumerator = jnp.einsum("as,as->", value_diff, bellman_value_diff - value_diff)

    condition = jnp.isclose(delta_denumerator, 0, atol=1e-2).astype("float32")
    delta = (delta_numerator * (1 - condition)) / (delta_denumerator + condition)

    return (1 - delta) * bellman_value + delta * bellman_prev_value, value
