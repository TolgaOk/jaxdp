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


class AsyncTrainMetrics(NamedTuple):
    mean_behavior_reward: F["N"]
    mean_behavior_length: F["N"]
    std_behavior_reward: F["N"]
    std_behavior_length: F["N"]
    bellman_error: F["N"]
    l_inf_value_norm: F["N"]
    policy_evaluation: F["N"]
    value_error: F["N"]


class SyncTrainMetrics(NamedTuple):
    policy_evaluation: F["N"]
    l_inf_value_norm: F["N"]
    bellman_error: F["N"]
    expected_value: F["N"]
    value_error: F["N"]


def initialize_metrics(metric_cls: Union[Type[AsyncTrainMetrics],
                                         Type[SyncTrainMetrics]],
                       size: int
                       ) -> "AsyncTrainMetrics":
    return metric_cls(*[jnp.full((size,), jnp.nan) for _ in metric_cls._fields])


BatchKeys: Type = NewType("BatchKeys", I["B 2"])


def eval_policy(mdp: MDP, policy: PiType, gamma: float) -> float:
    return (jaxdp.policy_evaluation.v(mdp, policy, gamma) * mdp.initial).sum()


@register_as("asynchronous")
def train(sampler_state: SamplerState,
          init_value: QType,
          mdp: MDP,
          key: KeyType,
          learner_state: Any,
          value_star: QType,
          eval_period: int,
          n_steps: int,
          gamma: float,
          behavior_policy_fn: Callable[[QType, int], PiType],
          target_policy_fn: Callable[[QType, int], PiType],
          update_fn: Callable,
          sample_fn: Callable[[MDP, SamplerState, PiType, BatchKeys], RolloutSample],
          verbose: bool = True,
          ) -> Tuple[AsyncTrainMetrics, QType, SamplerState, Any]:

    metrics = initialize_metrics(AsyncTrainMetrics, n_steps // eval_period)
    value = init_value

    def _print_fn(print_data, ):
        index, metrics = print_data
        title = "Training Metrics - Iteration"
        print("=" * 50)
        print(f"{title:^40} {index + 1}")
        print("")
        for name in metrics._fields:
            val = getattr(metrics, name)[index].item()
            formatted_name = name.replace("_", " ").title()
            print(f"{formatted_name:<30} : {val:>15.4f}")

    def _log_fn(sampler_state, metrics, step_data):
        index, value, next_value = step_data
        target_policy = target_policy_fn(value, index)

        sampled_rewards = sampler_state.episode_reward_queue
        sampled_lengths = sampler_state.episode_length_queue

        bellman_error = jnp.abs(jaxdp.bellman_operator.q(
            mdp, target_policy, value, gamma) - value).max()
        l_inf_value_norm = jnp.abs(next_value - value).max()
        policy_evaluation = eval_policy(mdp, target_policy, gamma)
        value_error = jnp.linalg.norm(value - value_star, ord=2)

        metrics = AsyncTrainMetrics(
            metrics.mean_behavior_reward.at[index].set(jnp.nanmean(sampled_rewards)),
            metrics.mean_behavior_length.at[index].set(jnp.nanmean(sampled_lengths)),
            metrics.std_behavior_reward.at[index].set(jnp.nanstd(sampled_rewards)),
            metrics.std_behavior_length.at[index].set(jnp.nanstd(sampled_lengths)),
            metrics.bellman_error.at[index].set(bellman_error),
            metrics.l_inf_value_norm.at[index].set(l_inf_value_norm),
            metrics.policy_evaluation.at[index].set(policy_evaluation),
            metrics.value_error.at[index].set(value_error),
        )

        if verbose:
            call(_print_fn, (index, metrics))
        sampler_state = sampler_state.refresh_queues()
        return sampler_state, metrics

    def _step_fn(index, step_data):
        metrics, sampler_state, value, learner_state, key = step_data

        key, step_key = jrd.split(key, 2)
        behavior_policy = behavior_policy_fn(value, index)
        rollout_sample, sampler_state = sample_fn(step_key, mdp, sampler_state, behavior_policy)
        next_value, learner_state = update_fn(index, rollout_sample, value, learner_state, gamma)

        sampler_state, metrics, = jax.lax.cond(
            (index % eval_period) == (eval_period - 1),
            _log_fn,
            lambda sampler_state, metrics, *_: (sampler_state, metrics),
            sampler_state, metrics, (index // eval_period, value, next_value))

        value = next_value
        return metrics, sampler_state, value, learner_state, key

    metrics, sampler_state, value, learner_state, key = jax.lax.fori_loop(
        0, n_steps, _step_fn, (metrics, sampler_state, value, learner_state, key))

    return metrics, value, learner_state, sampler_state


@register_as("sync")
def train(init_value: QType,
          mdp: MDP,
          key: KeyType,
          learner_state: Any,
          value_star: QType,
          n_steps: int,
          eval_period: int,
          gamma: float,
          target_policy_fn: Callable,
          update_fn: Callable,
          ) -> Any:

    metrics = initialize_metrics(SyncTrainMetrics, n_steps)
    value = init_value

    def _step_fn(index, _step_data):
        metrics, value, learner_state, key = _step_data
        key, step_key = jrd.split(key, 2)

        reward, next_state, terminal = jaxdp.sync_sample(mdp, step_key)
        sample = SyncSample(next_state, reward, terminal)
        next_value, learner_state = update_fn(index, sample, value, learner_state, gamma)

        target_policy = target_policy_fn(next_value, index)
        policy_evaluation = jax.lax.cond(
            (index % eval_period) == (eval_period - 1),
            eval_policy,
            lambda *_: jnp.nan,
            mdp, target_policy, gamma
        )
        expected_value = jnp.einsum("as,as,s->", target_policy, value, mdp.initial)
        l_inf_value_norm = jnp.abs(next_value - value).max()
        bellman_error = jnp.abs(jaxdp.bellman_operator.q(
            mdp, target_policy, value, gamma) - value).max()
        value_error = jnp.linalg.norm(value - value_star, ord=2)

        metrics = SyncTrainMetrics(
            metrics.policy_evaluation.at[index].set(policy_evaluation),
            metrics.l_inf_value_norm.at[index].set(l_inf_value_norm),
            metrics.bellman_error.at[index].set(bellman_error),
            metrics.expected_value.at[index].set(expected_value),
            metrics.value_error.at[index].set(value_error),
        )

        return metrics, next_value, learner_state, key

    metrics, value, learner_state, key = jax.lax.fori_loop(
        0, n_steps, _step_fn, (metrics, value, learner_state, key))

    return metrics, value, learner_state


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
