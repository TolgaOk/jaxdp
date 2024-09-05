from typing import Callable, Dict, Union, List, Tuple, NamedTuple, Type, Any, NewType

import jax
import jax.numpy as jnp
import jax.random as jrd
from jax.typing import ArrayLike as KeyType
from jax.experimental.host_callback import call

from jaxdp.learning.sampler import RolloutSample, SamplerState, SyncSample
from jaxdp.mdp.mdp import MDP
from jaxdp.typehints import QType, VType, PiType, F, I
import jaxdp
from jaxdp.utils import register_as


class ExpDecay(NamedTuple):
    temperature: float = 1.0
    offset: int = 1

    def decay_fn(self) -> Callable[[int], float]:
        def decay(step: int):
            return (1 / (step / self.temperature + self.offset))
        return decay


class TrainMetrics(NamedTuple):
    avg_episode_rewards: F["N"]
    avg_episode_lengths: F["N"]
    std_episode_rewards: F["N"]
    std_episode_lengths: F["N"]
    max_value_diff: F["N"]
    avg_value_eval: F["N"]

    @staticmethod
    def initialize(step_size: int) -> "TrainMetrics":
        return TrainMetrics(*[jnp.full((step_size,), jnp.nan) for _ in range(6)])

    def write(self, index: int, values: Dict[str, float]) -> "TrainMetrics":
        self_dict = self._asdict()
        return TrainMetrics(
            *[self_dict[name].at[index].set(value)
              for name, value in values.items()]
        )


class SyncTrainMetrics(NamedTuple):
    expected_policy_eval: F["N"]
    max_value_diff: F["N"]
    bellman_error: F["N"]
    expected_value: F["N"]
    value_error: F["N"]

    @staticmethod
    def initialize(step_size: int) -> "SyncTrainMetrics":
        return SyncTrainMetrics(*[jnp.full((step_size,), jnp.nan)
                                  for _ in range(5)])


BatchKeys: Type = NewType("BatchKeys", I["B 2"])


@register_as("async")
def train(sampler_state: SamplerState,
          init_value: QType,
          mdp,
          key: KeyType,
          eval_steps: int,
          n_steps: int,
          policy_fn: Callable[[QType, int], F["A"]],
          update_fn: Callable[[RolloutSample, QType], QType],
          sample_fn: Callable[[MDP, SamplerState, PiType, BatchKeys], RolloutSample],
          verbose: bool = True,
          ) -> Tuple[TrainMetrics, QType]:

    metrics = TrainMetrics.initialize(n_steps // eval_steps)
    value = init_value

    def print_log(data):
        print(f"Progress: {data[0]:5d}, Avg episode reward: {data[1]:.2f}")

    def log_fn(sampler_state, metrics, value_norm, index):
        metrics = metrics.write(
            index // eval_steps,
            {"avg_episode_rewards": jnp.nanmean(sampler_state.episode_reward_queue),
             "avg_episode_lengths": jnp.nanmean(sampler_state.episode_length_queue),
             "std_episode_rewards": jnp.nanstd(sampler_state.episode_reward_queue),
             "std_episode_lengths": jnp.nanstd(sampler_state.episode_length_queue),
             "max_value_diff": value_norm,
             "avg_value_eval": jnp.nan})
        if verbose:
            call(print_log,
                 (index + 1, jnp.nanmean(sampler_state.episode_reward_queue)))
        sampler_state = sampler_state.refresh_queues()
        return sampler_state, metrics

    def _step_fn(index, step_data):
        metrics, sampler_state, value, key = step_data

        key, step_key = jrd.split(key, 2)
        step_keys = jrd.split(step_key, sampler_state.last_state.shape[0])
        policy = policy_fn(value, index)
        rollout_sample, sampler_state = sample_fn(mdp, sampler_state, policy, step_keys)
        next_value = update_fn(rollout_sample, value)
        value_norm = jnp.max(jnp.abs(next_value - value))

        sampler_state, metrics, = jax.lax.cond(
            (index % eval_steps) == (eval_steps - 1),
            log_fn,
            lambda sampler_state, metrics, *_: (sampler_state, metrics),
            sampler_state, metrics, value_norm, index)

        value = next_value
        return metrics, sampler_state, value, key

    metrics, sampler_state, value, key = jax.lax.fori_loop(
        0, n_steps, _step_fn, (metrics, sampler_state, value, key))

    return metrics, value


@register_as("async")
def evaluate():
    raise NotImplementedError


@register_as("sync")
def train(init_value: QType,
          mdp: MDP,
          value_star: QType,
          key: KeyType,
          learner_state: Any,
          n_steps: int,
          eval_period: int,
          gamma: float,
          policy_fn: Callable,
          update_fn: Callable,
          ) -> Any:

    metrics = SyncTrainMetrics.initialize(n_steps)
    value = init_value

    def _eval_policy(policy):
        return (jaxdp.policy_evaluation.q(mdp, policy, gamma) * mdp.initial).sum()

    def _step_fn(index, _step_data):
        metrics, value, learner_state, key = _step_data
        key, step_key = jrd.split(key, 2)

        reward, next_state, terminal = jaxdp.sync_sample(mdp, step_key)
        sample = SyncSample(next_state, reward, terminal)

        _next_value, learner_state = update_fn(index, sample, value, learner_state, gamma)
        # next_value = jnp.einsum("as,s->as", _next_value, 1 - mdp.terminal)
        next_value = _next_value

        policy = policy_fn(next_value, index)
        expected_policy_eval = jax.lax.cond(
            (index % eval_period) == (eval_period - 1),
            _eval_policy,
            lambda _: jnp.nan,
            policy
        )
        expected_value = jnp.einsum("as,as,s->", policy, value, mdp.initial)
        max_value_diff = jnp.abs(next_value - value).max()
        bellman_error = jnp.abs(jaxdp.bellman_operator.q(
            mdp, policy, value, gamma) - value).max()
        value_error = jnp.linalg.norm(value - value_star, ord=2)

        metrics = SyncTrainMetrics(
            metrics.expected_policy_eval.at[index].set(expected_policy_eval),
            metrics.max_value_diff.at[index].set(max_value_diff),
            metrics.bellman_error.at[index].set(bellman_error),
            metrics.expected_value.at[index].set(expected_value),
            metrics.value_error.at[index].set(value_error),
        )

        return metrics, next_value, learner_state, key

    metrics, value, learner_state, key = jax.lax.fori_loop(
        0, n_steps, _step_fn, (metrics, value, learner_state, key))

    return metrics, value, learner_state


@register_as("sync")
def evaluate():
    raise NotImplementedError


def no_learner_state(update_fn: Callable):
    """ Update function wrapper to ignore learner_state in the train
    """
    def wrapper(index, sample, value, learner_state, *args, **kwargs):
        return update_fn(index, sample, value, *args, **kwargs), None

    return wrapper


def no_step_index(update_fn: Callable):
    """ Update function wrapper to ignore step index in the train
    """
    def wrapper(index, *args, **kwargs):
        return update_fn(*args, **kwargs)

    return wrapper
