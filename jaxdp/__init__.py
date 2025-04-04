from typing import Union, Tuple, Optional
import jax
from jaxtyping import Float, Array
from jaxdp.mdp.mdp import MDP, flatten_mdp, unflatten_mdp

from jaxdp.base import (
    greedy_policy,
    soft_policy,
    e_greedy_policy,
    sample_from,
    to_greedy_state_value,
    to_state_action_value,
    expected_value,
    expected_value,
    _markov_chain_pi,
    markov_chain_eigen_values,
    sample_based_policy_evaluation,
    policy_evaluation,
    bellman_operator,
    bellman_optimality_operator,
    stationary_distribution,
    sync_sample,
    async_sample_step,
    async_sample_step_pi,
    sg,
)


__all__ = [
    "greedy_policy",
    "soft_policy",
    "e_greedy_policy",
    "sample_from",
    "to_greedy_state_value",
    "to_state_action_value",
    "expected_value",
    "expected_value",
    "_markov_chain_pi",
    "markov_chain_eigen_values",
    "sample_based_policy_evaluation",
    "policy_evaluation",
    "policy_evaluation",
    "bellman_operator",
    "bellman_optimality_operator",
    "stationary_distribution",
    "sync_sample",
    "async_sample_step",
    "async_sample_step_pi",
    "sg",
]


jax.tree_util.register_pytree_node(MDP, flatten_mdp, unflatten_mdp)
