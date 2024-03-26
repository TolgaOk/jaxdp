from typing import Dict, Union, List, NamedTuple, Tuple
from abc import abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Float, Array
from jax.experimental.host_callback import call

import jaxdp
from jaxdp.learning.sampler import RolloutSample, StepSample


def td_step(sample: StepSample,
            next_action: Float[Array, "A"],
            value: Float[Array, "A S"],
            gamma: float
            ) -> Float[Array, ""]:
    next_value = jnp.einsum("as,a,s->", value, next_action, sample.next_state)
    current_value = jnp.einsum("as,a,s->", value, sample.action, sample.state)
    return sample.reward + gamma * next_value * (1 - sample.terminal) - current_value


# On-policy & forward-view trace
def td_lambda_step(rollout: RolloutSample,
                   value: Float[Array, "A S"],
                   last_action: Float[Array, "A"],
                   gamma: float,
                   lambda_: float
                   ) -> Float[Array, "T"]:
    returns = jnp.zeros_like(rollout.reward)
    rollout_size = rollout.reward.shape[-1]
    next_action = jnp.concatenate([rollout.action[1:], last_action.reshape(1, -1)], axis=0)
    td = jax.vmap(td_step, (0, 0, None, None))(
        rollout,
        next_action,
        value,
        gamma
    )
    done = jnp.logical_or(rollout.terminal, rollout.timeout)

    def step_fn(index, returns):
        index = rollout_size - index - 1
        step_return = td[index] + lambda_ * gamma * returns[index + 1] * (1 - done[index])
        return returns.at[index].set(step_return)

    return jax.lax.fori_loop(0, rollout_size, step_fn, returns)


def td_lambda_learning_update(rollout: RolloutSample,
                              value: Float[Array, "A S"],
                              gamma: float,
                              alpha: float,
                              lambda_: float,
                              ) -> Float[Array, "A S"]:
    batch_td_lambda_update = jax.vmap(
        td_lambda_step,
        (0, None, 0, None, None))
    max_values = jnp.argmax(jnp.einsum(
        "bx,ux->bu", rollout.next_state[:, -1], value), axis=-1, keepdims=True)
    last_action = (max_values == jnp.arange(
        rollout.action.shape[-1]).reshape(1, -1)).astype("float32")
    target_values = batch_td_lambda_update(rollout, value, last_action, gamma, lambda_)
    reduced_values = jnp.einsum("bts,bta,bt->as", rollout.state, rollout.action, target_values)
    count = jnp.clip(jnp.einsum(
        "bts,bta->as", rollout.state, rollout.action), 1.0, None)
    return value + reduced_values / count * alpha


