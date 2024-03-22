from typing import Callable, Dict, Union, List, Tuple, NamedTuple, Type, Any

import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float, Int32
from jax.typing import ArrayLike
from jax.experimental.host_callback import call

from jaxdp.learning.sampler import RolloutSample, SamplerState, SyncSample
from jaxdp.mdp.mdp import MDP
import jaxdp


class TrainMetrics(NamedTuple):
    avg_episode_rewards: Float[Array, "N"]
    avg_episode_lengths: Float[Array, "N"]
    std_episode_rewards: Float[Array, "N"]
    std_episode_lengths: Float[Array, "N"]
    max_value_diff: Float[Array, "N"]
    avg_value_eval: Float[Array, "N"]

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
    expected_policy_eval: Float[Array, "N"]
    max_value_diff: Float[Array, "N"]
    max_bellman_error: Float[Array, "N"]
    expected_value: Float[Array, "N"]
    value_error: Float[Array, "N"]

    @staticmethod
    def initialize(step_size: int) -> "SyncTrainMetrics":
        return SyncTrainMetrics(*[jnp.full((step_size,), jnp.nan)
                                  for _ in range(5)])


QValueType: Type = Float[Array, "A S"]
PolicyType: Type = Float[Array, "A S"]
BatchKeys: Type = Int32[Array, "B 2"]


def train(sampler_state: SamplerState,
          init_value: QValueType,
          mdp,
          key: ArrayLike,
          eval_steps: int,
          n_steps: int,
          policy_fn: Callable[[QValueType, int], Float[Array, "A"]],
          update_fn: Callable[[RolloutSample, QValueType], QValueType],
          sample_fn: Callable[[MDP, SamplerState, PolicyType, BatchKeys], RolloutSample],
          verbose: bool = True,
          ) -> Tuple[TrainMetrics, QValueType]:

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


def evaluate_value():
    raise NotImplementedError


def sync_train(init_value: QValueType,
               mdp: MDP,
               value_star: Float[Array, "A S"],
               key: ArrayLike,
               n_steps: int,
               eval_period: int,
               gamma: float,
               policy_fn: Callable,
               update_fn: Callable,
               learner_state: Any
               ) -> Any:

    metrics = SyncTrainMetrics.initialize(n_steps)
    value = init_value

    def _eval_policy(policy):
        return (jaxdp.policy_evaluation(mdp, policy, gamma) * mdp.initial).sum()

    def _step_fn(index, _step_data):
        metrics, value, learner_state, key = _step_data
        key, step_key = jrd.split(key, 2)

        reward, next_state, terminal = jaxdp.sync_sample(mdp, step_key)
        sample = SyncSample(next_state, reward, terminal)

        next_value, learner_state = update_fn(index, sample, value, learner_state)

        policy = policy_fn(next_value, index)
        expected_policy_eval = jax.lax.cond(
            (index % eval_period) == (eval_period - 1),
            _eval_policy,
            lambda _: jnp.nan,
            policy
        )
        expected_value = jnp.einsum("as,as,s->", policy, value, mdp.initial)
        max_value_diff = jnp.abs(next_value - value).max()
        max_bellman_error = jnp.abs(jaxdp.bellman_q_operator(
            mdp, policy, value, gamma) - value).max()
        value_error = jnp.linalg.norm(value - value_star, ord=2) / \
            jnp.linalg.norm(value_star, ord=2)

        metrics = SyncTrainMetrics(
            metrics.expected_policy_eval.at[index].set(expected_policy_eval),
            metrics.max_value_diff.at[index].set(max_value_diff),
            metrics.max_bellman_error.at[index].set(max_bellman_error),
            metrics.expected_value.at[index].set(expected_value),
            metrics.value_error.at[index].set(value_error),
        )

        return metrics, next_value, learner_state, key

    metrics, value, learner_state, key = jax.lax.fori_loop(
        0, n_steps, _step_fn, (metrics, value, learner_state, key))

    return metrics, value
