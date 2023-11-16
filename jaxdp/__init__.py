from typing import Union, Tuple, Optional
import jax
from jaxtyping import Float, Array
from jaxdp.mdp.mdp import MDP, flatten_mdp, unflatten_mdp

from jaxdp.base import (
    greedy_policy,
    soft_policy,
    e_greedy_policy,
    sample_from,
    expected_state_value,
    expected_q_value,
    _markov_chain_pi,
    sample_based_policy_evaluation,
    policy_evaluation,
    q_policy_evaluation,
    bellman_error,
    sync_sample,
    async_sample_step,
    async_sample_step_pi,
)


__all__ = [
    "greedy_policy",
    "soft_policy",
    "e_greedy_policy",
    "sample_from",
    "expected_state_value",
    "expected_q_value",
    "_markov_chain_pi",
    "sample_based_policy_evaluation",
    "policy_evaluation",
    "q_policy_evaluation",
    "bellman_error",
    "sync_sample",
    "async_sample_step",
    "async_sample_step_pi",
    "MDP",
]


jax.tree_util.register_pytree_node(MDP, flatten_mdp, unflatten_mdp)
