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


class vi(metaclass=StaticMeta):

    @struct.dataclass
    class State:
        q_values: jnp.ndarray
        gamma: jnp.ndarray

    @struct.dataclass
    class Args:
        pass

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "vi.State":
        q_vals = jrd.uniform(key, (mdp.action_size, mdp.state_size),
                               dtype="float", minval=0.0, maxval=1.0)
        return vi.State(q_values=q_vals, gamma=gamma)

    def update(state: "vi.State", mdp: MDP, step: int, args: "vi.Args") -> "vi.State":
        new_q = bellman_op.q(mdp, state.q_values, state.gamma)
        return state.replace(q_values=new_q)


class nesterov_vi(metaclass=StaticMeta):

    @struct.dataclass
    class State:
        q_values: jnp.ndarray
        y_values: jnp.ndarray
        prev_q_values: jnp.ndarray
        gamma: jnp.ndarray
        beta: jnp.ndarray

    @struct.dataclass
    class Args:
        pass

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "nesterov_vi.State":
        q_vals = jrd.uniform(key, (mdp.action_size, mdp.state_size),
                               dtype="float", minval=0.0, maxval=1.0)
        y_vals = q_vals.copy()
        prev_q = q_vals.copy()
        return nesterov_vi.State(
            q_values=q_vals, 
            y_values=y_vals,
            prev_q_values=prev_q,
            gamma=gamma,
            beta=jnp.array(0.0)
        )

    def update(state: "nesterov_vi.State", mdp: MDP, step: int, args: "nesterov_vi.Args") -> "nesterov_vi.State":
        beta_k = 0.1
        momentum = beta_k * (state.q_values - state.prev_q_values)
        y_vals = state.q_values + momentum
        new_q = bellman_op.q(mdp, y_vals, state.gamma)
        
        return state.replace(
            q_values=new_q,
            y_values=y_vals,
            prev_q_values=state.q_values
        )


class policy_iteration(metaclass=StaticMeta):

    @struct.dataclass
    class State:
        q_values: jnp.ndarray
        gamma: jnp.ndarray

    @struct.dataclass  
    class Args:
        pass

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "policy_iteration.State":
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
        
        return policy_iteration.State(
            q_values=q_vals,
            gamma=gamma
        )

    def update(state: "policy_iteration.State", mdp: MDP, step: int, 
               args: "policy_iteration.Args") -> "policy_iteration.State":
        policy = greedy_policy.q(state.q_values)
        q_vals = policy_evaluation.q(mdp, policy, state.gamma)
        
        return state.replace(q_values=q_vals)
