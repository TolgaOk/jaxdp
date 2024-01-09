from tokenize import Floatnumber
from typing import Dict, Union, List, NamedTuple, Tuple
from abc import abstractmethod
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Float, Array

import jaxdp
from jaxdp.learning.learning import BaseLearning
from jaxdp.learning.sampler import RolloutSample


# On-policy & forward-view trace
def td_lambda(rollout: RolloutSample,
              value: Float[Array, "A S"],
              last_action: Float[Array, "A"],
              gamma: float,
              lambda_: float
              ) -> Float[Array, "A S"]:
    returns = jnp.zeros_like(rollout.reward)

    rollout_size = rollout.reward.shape[-1]
    next_action = jnp.concatenate([rollout.action[1:], last_action.reshape(1, -1)], axis=0)
    next_value = jnp.einsum("as,ta,ts->t", value, next_action, rollout.next_state)
    current_value = jnp.einsum("as,ta,ts->t", value, rollout.action, rollout.state)
    td_error = rollout.reward + gamma * next_value * \
        (1 - rollout.terminal) - current_value
    done = jnp.logical_or(rollout.terminal, rollout.timeout)

    def step_fn(index, returns):
        index = rollout_size - index - 1
        step_return = td_error[index] + lambda_ * gamma * returns[index + 1] * (1 - done[index])
        return returns.at[index].set(step_return)

    return jax.lax.fori_loop(0, rollout_size, step_fn, returns)
