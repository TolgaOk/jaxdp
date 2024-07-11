from typing import Union, Tuple, Optional
import jax
from jaxtyping import Float, Array
from jaxdp.mdp.mdp import MDP, flatten_mdp, unflatten_mdp

from jaxdp.base import (
    greedy_policy,
    greedy_policy_from_v,
    soft_policy,
    e_greedy_policy,
    sample_from,
    to_greedy_state_value,
    to_state_action_value,
    expected_state_value,
    expected_q_value,
    _markov_chain_pi,
    sample_based_policy_evaluation,
    policy_evaluation,
    q_policy_evaluation,
    bellman_v_operator,
    bellman_q_operator,
    bellman_optimality_operator,
    sync_sample,
    async_sample_step,
    async_sample_step_pi,
    sg,
)


__all__ = [
    "greedy_policy",
    "greedy_policy_from_v",
    "soft_policy",
    "e_greedy_policy",
    "sample_from",
    "to_greedy_state_value",
    "to_state_action_value",
    "expected_state_value",
    "expected_q_value",
    "_markov_chain_pi",
    "sample_based_policy_evaluation",
    "policy_evaluation",
    "q_policy_evaluation",
    "bellman_v_operator",
    "bellman_q_operator",
    "bellman_optimality_operator",
    "sync_sample",
    "async_sample_step",
    "async_sample_step_pi",
    "sg",
]


jax.tree_util.register_pytree_node(MDP, flatten_mdp, unflatten_mdp)
