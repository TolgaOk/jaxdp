""" Implementation of "From Optimization to Control: Quasi Policy Iteration"
    by Mohammad Amin Sharifi Kolarijani & Peyman Mohajerin Esfahani
    https://arxiv.org/abs/2311.11166
"""

from typing import Dict, Union, List
import jax.numpy as jnp
import jax.random as jrd
import jax
from jaxtyping import Array, Float

import jaxdp
from jaxdp import MDP
from jaxdp.mdp import MDP


def bellman_operator(mdp: MDP, value: Float[Array, "S"], gamma: float) -> Float[Array, "S"]:
    policy = jaxdp.greedy_policy_from_v(mdp, value, gamma)
    return jaxdp.bellman_v_operator(mdp, policy, value, gamma)


def greedy_policy_reward(mdp: MDP, value: Float[Array, "S"], gamma: float) -> Float[Array, "S"]:
    return jnp.einsum("as,as->s", mdp.reward,
                      jaxdp.greedy_policy_from_v(mdp, value, gamma))


def g_k(mdp: MDP, value: Float[Array, "S"], gamma: float) -> Float[Array, "S"]:
    bellman_op = bellman_operator(mdp, value, gamma)
    return value - bellman_op


def y_k(mdp: MDP, value: Float[Array, "S"], gamma: float) -> Float[Array, "S"]:
    bellman_error = g_k(mdp, value, gamma)
    return bellman_error - bellman_error.mean(keepdims=True)


def z_k(mdp: MDP, value: Float[Array, "S"], gamma: float) -> Float[Array, "S"]:
    reward = greedy_policy_reward(mdp, value, gamma)
    return reward - reward.mean(keepdims=True)


def delta_k(mdp: MDP, value: Float[Array, "S"], gamma: float) -> Float[Array, ""]:
    normalized_bellman_error = y_k(mdp, value, gamma)
    normalized_reward = z_k(mdp, value, gamma)
    return (jnp.einsum("s,s->", value, normalized_bellman_error) /
            jnp.einsum("s,s->", value, (normalized_bellman_error + normalized_reward)))


def lambda_k(mdp: MDP, value: Float[Array, "S"], gamma: float) -> Float[Array, ""]:
    bellman_error = g_k(mdp, value, gamma)
    reward = greedy_policy_reward(mdp, value, gamma)
    delta_coeff = delta_k(mdp, value, gamma)
    return gamma / (value.shape[-1] * (1 - gamma)) * ((delta_coeff - 1) * bellman_error + delta_coeff * reward).sum(-1)


def quasi_policy_iteration_update(mdp: MDP, value: Float[Array, "S"], gamma: float) -> Float[Array, "S"]:
    bellman_op = bellman_operator(mdp, value, gamma)
    reward = greedy_policy_reward(mdp, value, gamma)
    delta_coeff = delta_k(mdp, value, gamma)
    lambda_coeff = lambda_k(mdp, value, gamma)
    return (1 - delta_coeff) * bellman_op + delta_coeff * reward + lambda_coeff
