from typing import Dict, Union, List, NamedTuple, Tuple
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Float, Array
from jax.typing import ArrayLike

from jaxdp.learning.sampler import RolloutSample, SamplerState, StepSample, SyncSample


def q_target(next_state: Float[Array, "S"],
             reward: Float[Array, ""],
             terminal: Float[Array, ""],
             value: Float[Array, "A S"],
             gamma: float
             ) -> Float[Array, ""]:
    return (gamma * jnp.max(jnp.einsum("x,ux,->u", next_state,
                                       value, (1 - terminal)), axis=0)
            + reward)


def q_step(sample: StepSample,
           value: Float[Array, "A S"],
           gamma: float
           ) -> Float[Array, ""]:
    return (q_target(sample.next_state, sample.reward, sample.terminal, value, gamma) -
            jnp.einsum("s,a,as->", sample.state, sample.action, value))


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
    target_values = jnp.einsum("bts,bta,bt->btas", rollout.state, rollout.action, target_values)
    count = jnp.clip(jnp.einsum(
        "bts,bta->as", rollout.state, rollout.action), 1.0, None)
    return value + target_values.sum((0, 1)) / count * alpha


def sync_q_learning_update(sample: SyncSample,
                           value: Float[Array, "A S"],
                           gamma: float,
                           alpha: float,
                           ) -> Float[Array, "A S"]:

    batch_q_target = jax.vmap(jax.vmap(q_target, (0, 0, 0, None, None)), (0, 0, 0, None, None))
    target_values = batch_q_target(sample.next_state, sample.reward, sample.terminal, value, gamma)

    return value + (target_values - value) * alpha
