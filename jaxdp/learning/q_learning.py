from typing import Dict, Union, List, NamedTuple, Tuple
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Float, Array
from jax.typing import ArrayLike

from jaxdp.learning.sampler import RolloutSample, StepSample


def q_step(sample: StepSample,
           value: Float[Array, "A S"],
           gamma: float
           ) -> Float[Array, "A S"]:
    one_hot_value = jnp.einsum("s,a->as", sample.state, sample.action)
    return one_hot_value * (
        gamma * jnp.max(jnp.einsum("x,ux,->u", sample.next_state,
                        value, (1 - sample.terminal)), axis=0)
        + sample.reward - jnp.einsum("s,a,as->", sample.state, sample.action, value))


def q_learning_update(rollout: RolloutSample,
                      value: Float[Array, "A S"],
                      gamma: float,
                      alpha: float,
                      ) -> Float[Array, "A S"]:
    batch_q_update = jax.vmap(
        jax.vmap(
            q_step,
                (0, None, None)),
        (0, None, None))

    target_values = batch_q_update(rollout, value, gamma)
    count = jnp.clip(jnp.einsum(
        "bts,bta->as", rollout.state, rollout.action), 1.0, None)
    return value + target_values.sum((0, 1)) / count * alpha
