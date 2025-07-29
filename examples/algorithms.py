import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct
from typing import Dict, Tuple
from dataclasses import dataclass

from jaxdp.mdp import MDP
from jaxdp import bellman_optimality_operator as bellman_op
from jaxdp.base import policy_evaluation, greedy_policy
from jaxdp.utils import StaticMeta
from jaxdp.typehints import QType


class vi(metaclass=StaticMeta):
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Value (Q) Iteration
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    @struct.dataclass
    class State:
        q_vals: QType
        gamma: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "vi.State":
        q_vals = jrd.uniform(key, (mdp.action_size, mdp.state_size),
                               dtype="float", minval=0.0, maxval=1.0)
        return vi.State(q_vals=q_vals, gamma=gamma)

    def update(state: "vi.State", mdp: MDP, step: int) -> "vi.State":
        next_q = bellman_op.q(mdp, state.q_vals, state.gamma)
        return state.replace(q_vals=next_q)


class nesterov_vi(metaclass=StaticMeta):
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Nesterov Accelerated Value Iteration
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    @struct.dataclass
    class State:
        q_vals: QType
        prev_q: QType
        gamma: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "nesterov_vi.State":
        q_vals = jrd.uniform(key, (mdp.action_size, mdp.state_size),
                               dtype="float", minval=0.0, maxval=1.0)
        return nesterov_vi.State(
            q_vals=q_vals, 
            prev_q=q_vals.copy(),
            gamma=gamma
        )

    def update(state: "nesterov_vi.State", mdp: MDP, step: int) -> "nesterov_vi.State":
        beta = (1 - jnp.sqrt(1 - state.gamma ** 2)) / state.gamma
        z_vals = state.q_vals + beta * (state.q_vals - state.prev_q)
        bellman_residual = (bellman_op.q(mdp, z_vals, state.gamma) - z_vals)
        next_q = z_vals + (1 / (1 + state.gamma)) * bellman_residual
    
        return state.replace(
            q_vals=next_q,
            prev_q=state.q_vals
        )


class pi(metaclass=StaticMeta):
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Policy Iteration
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    @struct.dataclass
    class State:
        q_vals: QType
        gamma: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "pi.State":
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
        
        return pi.State(
            q_vals=q_vals,
            gamma=gamma
        )

    def update(state: "pi.State", mdp: MDP, step: int) -> "pi.State":
        policy = greedy_policy.q(state.q_vals)
        q_vals = policy_evaluation.q(mdp, policy, state.gamma)
        
        return state.replace(q_vals=q_vals)
