from typing import Dict, Union, List, NamedTuple
from abc import abstractmethod
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Float, Array

import jaxdp
from jaxdp.learning.learning import BaseLearning
from jaxdp.learning.sampler import RolloutSample


def q_update(state, next_state, action, reward, terminal, timeout, value, gamma):
    one_hot_value = jnp.einsum("s,a->as", state, action)
    return one_hot_value * (
        gamma * jnp.max(jnp.einsum("x,ux,->u", next_state, value, (1 - terminal)), axis=0) +
        reward -
        jnp.einsum("s,a,as->", state, action, value))


class QLearning(BaseLearning):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 seed: int,
                 alpha: float = 0.01,
                 epsilon: float = 0.1,
                 ) -> None:
        super().__init__(state_size, action_size, seed)
        self.alpha = alpha
        self.epsilon = epsilon
        self.batch_q_update = jax.jit(
            jax.vmap(
                jax.vmap(
                    q_update,
                    (0, 0, 0, 0, 0, 0, None, None)),
                (0, 0, 0, 0, 0, 0, None, None))
        )

    def initialize_value(self) -> Float[Array, "... S"]:
        return jrd.uniform(self.init_key, shape=(self.action_size, self.state_size))

    @abstractmethod
    def policy(self) -> Float[Array, "A S"]:
        return jaxdp.e_greedy_policy(self.value, self.epsilon)

    @abstractmethod
    def value_update(self, rollout: RolloutSample, gamma: float) -> Float[Array, "... S"]:
        target_values = self.batch_q_update(rollout.state, rollout.next_state, rollout.action,
                                            rollout.reward, rollout.terminal, rollout.timeout,
                                            self.value, gamma)
        count = jnp.clip(jnp.einsum(
            "bts,bta->as", rollout.state, rollout.action), 1.0, None)
        return self.value + target_values.sum((0, 1)) / count * self.alpha
