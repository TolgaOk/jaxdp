from typing import Dict, Union, List
import jax.numpy as jnp
import jax.random as jrd
import jax
from jaxtyping import Array, Float

import jaxdp
from jaxdp import MDP
from jaxdp.mdp import MDP
from jaxdp.iterations.vi import QIteration


def policy_iteration_step(mdp: MDP, value: Float[Array, "A S"], gamma: float) -> Float[Array, "A S"]:
    # TODO: Add docstring
    # TODO: Add test
    policy_pi = jaxdp.greedy_policy(value)
    return jaxdp.q_policy_evaluation(mdp, policy_pi, gamma)


class PolicyIteration(QIteration):

    def step_value(self, mdp: MDP, gamma: float) -> Float[Array, "S"]:
        return policy_iteration_step(mdp, self.value, gamma)
