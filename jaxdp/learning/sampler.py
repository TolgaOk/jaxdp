from typing import Any, Dict, Union, List, NamedTuple, Tuple, Optional
import jax
import jax.numpy as jnp
import jax.random as jrd
from jax.typing import ArrayLike as KeyType
import distrax

import jaxdp
from jaxdp import MDP
from jaxdp.mdp import MDP
from jaxdp.typehints import F, PiType


class RolloutSample(NamedTuple):
    state: F["... T S"]
    next_state: F["... T S"]
    action: F["... T A"]
    reward: F["... T"]
    terminal: F["... T"]
    timeout: F["... T"]

    def __getitem__(self, key) -> "RolloutSample":
        return RolloutSample(
            self.state[key],
            self.next_state[key],
            self.action[key],
            self.reward[key],
            self.terminal[key],
            self.timeout[key],
        )

    @classmethod
    def full(cls, state_size: int, action_size: int, rollout_length: int
             ) -> "RolloutSample":
        return cls(
            state=jnp.full((rollout_length, state_size), jnp.nan),
            next_state=jnp.full((rollout_length, state_size), jnp.nan),
            action=jnp.full((rollout_length, action_size), jnp.nan),
            reward=jnp.full((rollout_length,), jnp.nan),
            terminal=jnp.full((rollout_length,), jnp.nan),
            timeout=jnp.full((rollout_length,), jnp.nan),
        )


class StepSample(NamedTuple):
    state: F["S"]
    next_state: F["S"]
    action: F["A"]
    reward: F[""]
    terminal: F[""]
    timeout: F[""]


class SyncSample(NamedTuple):
    next_state: F["A S S"]
    reward: F["A S"]
    terminal: F["A S"]


class EpisodeStats(NamedTuple):
    rewards: F[""]
    lengths: F[""]
    episode_reward_queue: F["K"]
    episode_length_queue: F["K"]


class SamplerState(NamedTuple):
    last_state: F["S"]
    episode_step: F[""]
    rewards: F[""]
    lengths: F[""]
    episode_reward_queue: F["K"]
    episode_length_queue: F["K"]

    @staticmethod
    def initialize_rollout_state(mdp: MDP,
                                 queue_size: int,
                                 init_state_key: KeyType
                                 ) -> "SamplerState":
        init_state = mdp.init_state(init_state_key)
        return SamplerState(
            init_state,
            jnp.array(1.0),
            jnp.array(1.0),
            jnp.array(1.0),
            jnp.full((queue_size,), jnp.nan),
            jnp.full((queue_size,), jnp.nan),
        )

    def refresh_queues(self) -> "SamplerState":
        return SamplerState(
            self.last_state,
            self.episode_step,
            self.rewards,
            self.lengths,
            jnp.full_like(self.episode_reward_queue, jnp.nan),
            jnp.full_like(self.episode_length_queue, jnp.nan),
        )


def _rollout_sample(length: int,
                    mdp: MDP,
                    policy: F["A S"],
                    state: F["S"],
                    episode_step: F[""],
                    max_episode_step: int,
                    key: KeyType
                    ) -> Tuple[RolloutSample,
                               F["S"],
                               F[""]]:
    state_size = mdp.state_size
    action_size = mdp.action_size
    rollout = dict(
        state=jnp.zeros((length, state_size)),
        next_state=jnp.zeros((length, state_size)),
        action=jnp.zeros((length, action_size)),
        reward=jnp.zeros((length,)),
        terminal=jnp.zeros((length,)),
        timeout=jnp.zeros((length,)),
    )
    keys = jrd.split(key, length)

    def step_fn(index, step_data):
        rollout, state, episode_step = step_data
        *step_data, _state, episode_step = jaxdp.async_sample_step_pi(
            mdp, policy, state, episode_step, max_episode_step, keys[index])
        names = ("action", "next_state", "reward",
                 "terminal", "timeout", "state")
        for name, data in zip(names, (*step_data, state)):
            rollout[name] = rollout[name].at[index].set(data)

        return (rollout, _state, episode_step)

    rollout, state, episode_step = jax.lax.fori_loop(
        0, length, step_fn, (rollout, state, episode_step))

    return RolloutSample(**rollout), state, episode_step


def update_sampler_records(rollout_data: RolloutSample,
                           sampler_state: SamplerState,
                           ) -> EpisodeStats:
    rewards = sampler_state.rewards
    lengths = sampler_state.lengths
    eps_rewards = sampler_state.episode_reward_queue
    eps_lengths = sampler_state.episode_length_queue

    done = jnp.logical_or(rollout_data.terminal, rollout_data.timeout)

    def queue_push(queue_array, value, condition):
        pushed_array = queue_array.at[1:].set(queue_array[:-1]).at[0].set(value)
        no_nan_queue = jnp.nan_to_num(queue_array)
        return (pushed_array * condition +
                (no_nan_queue / (1 - jnp.isnan(queue_array) * (1 - condition))) * (1 - condition))

    def step_fn(index, step_data):
        rewards, lengths, eps_rewards, eps_lengths = step_data
        rewards = rollout_data.reward[index] + rewards
        lengths = 1 + lengths
        eps_rewards = queue_push(eps_rewards, rewards, done[index])
        eps_lengths = queue_push(eps_lengths, lengths, done[index])

        return (rewards * (1 - done[index]), lengths * (1 - done[index]), eps_rewards, eps_lengths)

    rewards, lengths, eps_rewards, eps_lengths = jax.lax.fori_loop(
        0, rollout_data.reward.shape[0], step_fn, (rewards,
                                                   lengths,
                                                   eps_rewards,
                                                   eps_lengths))

    return rewards, lengths, eps_rewards, eps_lengths


def rollout_sample(mdp: MDP,
                   sampler_state: SamplerState,
                   policy: PiType,
                   key: KeyType,
                   max_episode_length: int,
                   rollout_len: int,
                   ) -> Tuple[RolloutSample, SamplerState]:
    rollout, last_state, episode_step = _rollout_sample(
        rollout_len, mdp, policy, sampler_state.last_state,
        sampler_state.episode_step, max_episode_length, key)

    return rollout, SamplerState(
        last_state,
        episode_step,
        *update_sampler_records(rollout, sampler_state)
    )
