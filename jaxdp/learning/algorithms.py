from typing import Callable, Dict, Union, List, NamedTuple, Tuple
import jax
import jax.numpy as jnp
import jax.random as jrd
from jax.typing import ArrayLike as KeyType
from flax import struct

import jaxdp.mdp.sampler.mdp as sampler
from jaxdp.typehints import QType, F
from jaxdp.utils import StaticMeta


@struct.dataclass
class StepSample:
    state: F["S"]
    next_state: F["S"]
    action: F["A"]
    reward: F[""]
    terminal: F[""]
    timeout: F[""]


@struct.dataclass
class SyncSample:
    next_state: F["A S S"]
    reward: F["A S"]
    terminal: F["A S"]


class q_learning(metaclass=StaticMeta):

    def target(next_state: F["S"],
               reward: F[""],
               terminal: F[""],
               value: F["A S"],
               gamma: float
               ) -> F[""]:
        return (gamma * jnp.max(jnp.einsum("x,ux,->u", next_state,
                                           value, (1 - terminal)), axis=0)
                + reward)

    def update(value: QType, next_value: QType, alpha: float) -> QType:
        return value + alpha * next_value

    class asynchronous(metaclass=StaticMeta):

        def step(sample: StepSample, value: QType, gamma: float) -> F[""]:
            return (q_learning.target(sample.next_state, sample.reward,
                                      sample.terminal, value, gamma) -
                    jnp.einsum("s,a,as->", sample.state, sample.action, value))

    class synchronous(metaclass=StaticMeta):

        def step(sample: SyncSample, value: QType, gamma: float) -> QType:
            batch_q_target = jax.vmap(
                jax.vmap(q_learning.target, (0, 0, 0, None, None)), (0, 0, 0, None, None))
            target_values = batch_q_target(sample.next_state, sample.reward,
                                           sample.terminal, value, gamma)
            return (target_values - value)


class speedy_q_learning(metaclass=StaticMeta):

    class synchronous(metaclass=StaticMeta):

        def update(sample: SyncSample,
                   value: QType,
                   past_value: QType,
                   gamma: float,
                   alpha: float,
                   ) -> Tuple[QType, QType]:
            batch_q_target = jax.vmap(
                jax.vmap(q_learning.target, (0, 0, 0, None, None)), (0, 0, 0, None, None))
            bellman_op = batch_q_target(sample.next_state, sample.reward,
                                        sample.terminal, value, gamma)
            past_bellman_op = batch_q_target(sample.next_state, sample.reward,
                                             sample.terminal, past_value, gamma)

            return (value + alpha * (past_bellman_op - value) + (1 - alpha) * (bellman_op - past_bellman_op),
                    value)

        def init(init_value: QType, gamma: float, key: KeyType) -> QType:
            return jnp.zeros_like(init_value)


class zap_q_learning(metaclass=StaticMeta):

    class synchronous(metaclass=StaticMeta):

        def update(sample: SyncSample,
                   value: QType,
                   matrix_gain: F["AS AS"],
                   gamma: float,
                   alpha: float,
                   beta: float,
                   ) -> Tuple[QType, F["AS AS"]]:
            act_size, state_size = value.shape
            batch_q_target = jax.vmap(
                jax.vmap(q_learning.target, (0, 0, 0, None, None)), (0, 0, 0, None, None))
            delta = batch_q_target(sample.next_state, sample.reward,
                                   sample.terminal, value, gamma) - value

            next_action = jax.nn.one_hot(jnp.argmax(jnp.einsum(
                "asx,ux->asu", sample.next_state, value), axis=-1), act_size)
            step_matrix_gain = (jnp.eye(act_size * state_size) -
                                gamma * jnp.einsum("asx,asu->asux", sample.next_state, next_action
                                                   ).reshape(act_size * state_size, act_size * state_size))
            matrix_gain = matrix_gain + beta * (step_matrix_gain - matrix_gain)
            return (value + alpha * (jnp.linalg.inv(matrix_gain) @ delta.flatten()).reshape(act_size, state_size),
                    matrix_gain)

        def init(init_value: QType, gamma: float, key: KeyType) -> QType:
            state_size, action_size = init_value.shape
            return jnp.eye((state_size * action_size))


class reducer(metaclass=StaticMeta):

    def every_visit(rollout: StepSample,
                    value: QType,
                    ) -> QType:
        count = jnp.clip(jnp.einsum(
            "...s,...a->as", rollout.state, rollout.action), 1.0, None)
        return jnp.einsum("...,...s,...a,as->as",
                          value,
                          rollout.state,
                          rollout.action,
                          1 / count)
