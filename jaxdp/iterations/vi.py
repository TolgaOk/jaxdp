from typing import Dict, Union, List
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float

import jaxdp
from jaxdp import MDP
from jaxdp.mdp import MDP
from jaxdp.iterations.iteration import BaseIteration


def q_iteration_step(mdp: MDP, value: Float[Array, "A S"], gamma: float) -> Float[Array, "A S"]:
    # TODO: Add docstring
    # TODO: Add test
    non_done = (1 - mdp.terminal)
    return mdp.reward * non_done.reshape(1, -1) + gamma * jnp.einsum(
        "axs,x,x->as",
        mdp.transition,
        jnp.max(value, axis=0),
        non_done)


def value_iteration_step(mdp: MDP, value: Float[Array, "S"], gamma: float) -> Float[Array, "S"]:
    # TODO: Add docstring
    # TODO: Add test
    non_done = (1 - mdp.terminal)
    return jnp.max(mdp.reward + gamma * jnp.einsum(
        "axs,x,x->as",
        mdp.transition,
        value,
        non_done),
        axis=0) * non_done


class QIteration(BaseIteration):

    value_expr: str = "q"

    @property
    def expected_value_expr(self) -> str:
        return fr"${self.init_dist_expr}^T (\max_a {self.value_expr}_k^a)$"

    def expected_value(self, mdp: MDP) -> Float[Array, ""]:
        return jaxdp.expected_q_value(mdp, self.value)

    def greedy_policy(self, *_) -> Float[Array, "A S"]:
        return jaxdp.greedy_policy(self.value)

    def bellman_operator(self, mdp: MDP, policy: Float[Array, "A S"], gamma: float
                         ) -> Float[Array, "S"]:
        return jaxdp.bellman_q_operator(mdp, policy, self.value, gamma)

    def initialize_value(self) -> Float[Array, "A S"]:
        return jrd.uniform(self.init_key, shape=(self.action_size, self.state_size))

    def step_value(self, mdp: MDP, gamma: float) -> Float[Array, "A S"]:
        return q_iteration_step(mdp, self.value, gamma)


class ValueIteration(BaseIteration):

    value_expr: str = "v"

    @property
    def expected_value_expr(self) -> str:
        return fr"${self.init_dist_expr}^T {self.value_expr}_k$"

    def expected_value(self, mdp: MDP) -> Float[Array, ""]:
        return jaxdp.expected_state_value(mdp, self.value)

    def greedy_policy(self, mdp: MDP, gamma: float) -> Float[Array, "A S"]:
        return jaxdp.greedy_policy_from_v(mdp, self.value, gamma)

    def bellman_operator(self, mdp: MDP, policy: Float[Array, "A S"], gamma: float
                         ) -> Float[Array, "S"]:
        return jaxdp.bellman_v_operator(mdp, policy, self.value, gamma)

    def initialize_value(self) -> Float[Array, "S"]:
        return jrd.uniform(self.init_key, shape=(self.state_size,))

    def step_value(self, mdp: MDP, gamma: float) -> Float[Array, "S"]:
        return value_iteration_step(mdp, self.value, gamma)
