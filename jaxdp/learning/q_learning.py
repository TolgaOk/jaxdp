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


def sync_speedy_q_learning_update(sample: SyncSample,
                                  value: Float[Array, "A S"],
                                  past_value: Float[Array, "A S"],
                                  gamma: float,
                                  alpha: float,
                                  ) -> Tuple[Float[Array, "A S"], Float[Array, "A S"]]:
    batch_q_target = jax.vmap(jax.vmap(q_target, (0, 0, 0, None, None)), (0, 0, 0, None, None))
    bellman_op = batch_q_target(sample.next_state, sample.reward, sample.terminal, value, gamma)
    past_bellma_op = batch_q_target(sample.next_state, sample.reward,
                                    sample.terminal, past_value, gamma)

    return (value + alpha * (past_bellma_op - value) + (1 - alpha) * (bellman_op - past_bellma_op),
            value)


def sync_zap_q_learning_update(sample: SyncSample,
                               value: Float[Array, "A S"],
                               matrix_gain: Float[Array, "AS AS"],
                               gamma: float,
                               alpha: float,
                               beta: float,
                               ) -> Tuple[Float[Array, "A S"], Float[Array, "AS AS"]]:
    act_size, state_size = value.shape
    batch_q_target = jax.vmap(jax.vmap(q_target, (0, 0, 0, None, None)), (0, 0, 0, None, None))
    delta = batch_q_target(sample.next_state, sample.reward, sample.terminal, value, gamma) - value

    next_action = jax.nn.one_hot(jnp.argmax(jnp.einsum(
        "asx,ux->asu", sample.next_state, value), axis=-1), act_size)
    step_matrix_gain = (jnp.eye(act_size * state_size) -
                        gamma * jnp.einsum("asx,asu->asux", sample.next_state, next_action
                                           ).reshape(act_size * state_size, act_size * state_size))
    matrix_gain = matrix_gain + beta * (step_matrix_gain - matrix_gain)
    return (value + alpha * (jnp.linalg.inv(matrix_gain) @ delta.flatten()).reshape(act_size, state_size),
            matrix_gain)
