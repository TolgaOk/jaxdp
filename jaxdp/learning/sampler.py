from typing import Dict, Union, List, NamedTuple, Tuple, Optional
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float
from jax.typing import ArrayLike
import distrax

import jaxdp
from jaxdp import MDP
from jaxdp.mdp import MDP


class RolloutSample(NamedTuple):
    state: Float[Array, "... T S"]
    next_state: Float[Array, "... T S"]
    action: Float[Array, "... T A"]
    reward: Float[Array, "... T"]
    terminal: Float[Array, "... T"]
    timeout: Float[Array, "... T"]

    def __getitem__(self, key) -> "RolloutSample":
        return RolloutSample(
            self.state[key],
            self.next_state[key],
            self.action[key],
            self.reward[key],
            self.terminal[key],
            self.timeout[key],
        )


class StepSample(NamedTuple):
    state: Float[Array, "S"]
    next_state: Float[Array, "S"]
    action: Float[Array, "A"]
    reward: Float[Array, ""]
    terminal: Float[Array, ""]
    timeout: Float[Array, ""]


class SamplerState(NamedTuple):
    last_state: Float[Array, "S"]
    episode_step: Float[Array, ""]
    rewards: Float[Array, ""]
    lengths: Float[Array, ""]
    episode_reward_queue: Float[Array, "K"]
    episode_length_queue: Float[Array, "K"]

    @staticmethod
    def initialize_rollout_state(mdp: MDP,
                                 batch_size: int,
                                 queue_size: int,
                                 init_state_key: ArrayLike
                                 ) -> "SamplerState":
        init_state = jax.vmap(mdp.init_state, (0,))(jrd.split(init_state_key, batch_size))
        return SamplerState(
            init_state,
            jnp.zeros((batch_size,)),
            jnp.zeros((batch_size,)),
            jnp.zeros((batch_size,)),
            jnp.full((batch_size, queue_size,), jnp.nan),
            jnp.full((batch_size, queue_size,), jnp.nan),
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
                    policy: Float[Array, "A S"],
                    state: Float[Array, "S"],
                    episode_step: Float[Array, ""],
                    max_episode_step: int,
                    key: jrd.KeyArray
                    ) -> Tuple[RolloutSample,
                               Float[Array, "S"],
                               Float[Array, ""]]:
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


def rollout_sample(mdp: MDP,
                   sampler_state: SamplerState,
                   policy: Float[Array, "A S"],
                   key: jrd.KeyArray,
                   max_episode_length: int,
                   rollout_len: int,
                   ) -> Tuple[RolloutSample, SamplerState]:
    rollout, last_state, episode_step = _rollout_sample(
        rollout_len, mdp, policy, sampler_state.last_state,
        sampler_state.episode_step, max_episode_length, key)

    rewards = sampler_state.rewards
    lengths = sampler_state.lengths
    eps_rewards = sampler_state.episode_reward_queue
    eps_lengths = sampler_state.episode_length_queue

    done = jnp.logical_or(rollout.terminal, rollout.timeout)

    def queue_push(queue_array, value, condition):
        pushed_array = queue_array.at[1:].set(queue_array[:-1]).at[0].set(value)
        no_nan_queue = jnp.nan_to_num(queue_array)
        return (pushed_array * condition +
                (no_nan_queue / (1 - jnp.isnan(queue_array) * (1 - condition))) * (1 - condition))

    def step_fn(index, step_data):
        rewards, lengths, eps_rewards, eps_lengths = step_data
        rewards = rollout.reward[index] + rewards
        lengths = 1 + lengths

        eps_rewards = queue_push(eps_rewards, rewards, done[index])
        eps_lengths = queue_push(eps_lengths, lengths, done[index])

        return (rewards * (1 - done[index]), lengths * (1 - done[index]), eps_rewards, eps_lengths)

    rewards, lengths, eps_rewards, eps_lengths = jax.lax.fori_loop(
        0, rollout.reward.shape[0], step_fn, (rewards,
                                              lengths,
                                              eps_rewards,
                                              eps_lengths))

    return rollout, SamplerState(
        last_state,
        episode_step,
        rewards,
        lengths,
        eps_rewards,
        eps_lengths
    )
