import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct

from jaxdp.mdp import MDP, make_mrp
from jaxdp.operator import BellmanOptimality, PolicyEvaluation
from jaxdp.policy import Greedy

bellman_optimality = BellmanOptimality()
evaluation = PolicyEvaluation()


class vi:
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Value (Q) Iteration
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    class State(struct.PyTreeNode):
        q_vals: jax.Array
        gamma: jax.Array

    @staticmethod
    def init(mdp: MDP, key: chex.PRNGKey, gamma: float | jax.Array) -> "vi.State":
        q_vals = jrd.uniform(
            key, (mdp.action_size, mdp.state_size), dtype="float", minval=0.0, maxval=1.0
        )
        return vi.State(q_vals=q_vals, gamma=jnp.asarray(gamma))

    @staticmethod
    def update(state: "vi.State", mdp: MDP, step: jax.Array) -> "vi.State":
        next_q = bellman_optimality.q(mdp, state.q_vals, state.gamma)
        return state.replace(q_vals=next_q)


class nesterov_vi:
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Nesterov Accelerated Value Iteration
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    class State(struct.PyTreeNode):
        q_vals: jax.Array
        prev_q: jax.Array
        gamma: jax.Array

    @staticmethod
    def init(
        mdp: MDP, key: chex.PRNGKey, gamma: float | jax.Array
    ) -> "nesterov_vi.State":
        q_vals = jrd.uniform(
            key, (mdp.action_size, mdp.state_size), dtype="float", minval=0.0, maxval=1.0
        )
        return nesterov_vi.State(
            q_vals=q_vals,
            prev_q=q_vals.copy(),
            gamma=jnp.asarray(gamma),
        )

    @staticmethod
    def update(state: "nesterov_vi.State", mdp: MDP, step: jax.Array) -> "nesterov_vi.State":
        beta = (1 - jnp.sqrt(1 - state.gamma**2)) / state.gamma
        z_vals = state.q_vals + beta * (state.q_vals - state.prev_q)
        bellman_residual = bellman_optimality.q(mdp, z_vals, state.gamma) - z_vals
        next_q = z_vals + (1 / (1 + state.gamma)) * bellman_residual

        return state.replace(q_vals=next_q, prev_q=state.q_vals)


class pi:
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Policy Iteration
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    class State(struct.PyTreeNode):
        q_vals: jax.Array
        gamma: jax.Array

    @staticmethod
    def init(mdp: MDP, key: chex.PRNGKey, gamma: float | jax.Array) -> "pi.State":
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))

        return pi.State(q_vals=q_vals, gamma=jnp.asarray(gamma))

    @staticmethod
    def update(state: "pi.State", mdp: MDP, step: jax.Array) -> "pi.State":
        policy = Greedy().q(state.q_vals)
        q_vals = evaluation.q(mdp, make_mrp(mdp, policy), state.gamma)

        return state.replace(q_vals=q_vals)
