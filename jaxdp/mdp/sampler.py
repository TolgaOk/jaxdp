from typing import Any, NamedTuple, Tuple, Protocol
import jax
import jax.numpy as jnp
import jax.random as jrd
from jax.typing import ArrayLike as KeyType
from gymnax.environments.environment import Environment, EnvParams, EnvState

import jaxdp
from jaxdp import MDP
from jaxdp.mdp import MDP
from jaxdp.typehints import F, PiType


class RolloutSample(NamedTuple):
    obs: F["... T S"]
    next_obs: F["... T S"]
    action: F["... T A"]
    reward: F["... T"]
    terminal: F["... T"]
    timeout: F["... T"]

    def __getitem__(self, key) -> "RolloutSample":
        return RolloutSample(
            self.obs[key],
            self.next_obs[key],
            self.action[key],
            self.reward[key],
            self.terminal[key],
            self.timeout[key],
        )

    @classmethod
    def full(cls, obs_size: int, action_size: int, rollout_length: int
             ) -> "RolloutSample":
        return cls(
            obs=jnp.full((rollout_length, obs_size), jnp.nan),
            next_obs=jnp.full((rollout_length, obs_size), jnp.nan),
            action=jnp.full((rollout_length, action_size), jnp.nan),
            reward=jnp.full((rollout_length,), jnp.nan),
            terminal=jnp.full((rollout_length,), jnp.nan),
            timeout=jnp.full((rollout_length,), jnp.nan),
        )


class StepSample(NamedTuple):
    obs: F["S"]
    next_obs: F["S"]
    action: F["A"]
    reward: F[""]
    terminal: F[""]
    timeout: F[""]


class SyncSample(NamedTuple):
    next_obs: F["A S S"]
    reward: F["A S"]
    terminal: F["A S"]


class EpisodeStats(NamedTuple):
    rewards: F[""]
    lengths: F[""]
    episode_reward_queue: F["K"]
    episode_length_queue: F["K"]


class SamplerState(NamedTuple):
    last_obs: F["S"]
    episode_step: F[""]
    rewards: F[""]
    lengths: F[""]
    episode_reward_queue: F["K"]
    episode_length_queue: F["K"]

    @staticmethod
    def initialize_rollout_state(init_obs: F["S"],
                                 queue_size: int,
                                 ) -> "SamplerState":
        return SamplerState(
            init_obs,
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.full((queue_size,), jnp.nan),
            jnp.full((queue_size,), jnp.nan),
        )

    def refresh_queues(self) -> "SamplerState":
        return SamplerState(
            self.last_obs,
            self.episode_step,
            self.rewards,
            self.lengths,
            jnp.full_like(self.episode_reward_queue, jnp.nan),
            jnp.full_like(self.episode_length_queue, jnp.nan),
        )


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


def rollout_mdp_sample(key: KeyType,
                       mdp: MDP,
                       sampler_state: SamplerState,
                       policy: PiType,
                       max_episode_length: int,
                       rollout_len: int,
                       ) -> Tuple[RolloutSample, SamplerState]:
    obs = sampler_state.last_obs

    obs_size = mdp.state_size
    action_size = mdp.action_size
    rollout = dict(
        obs=jnp.zeros((rollout_len, obs_size)),
        next_obs=jnp.zeros((rollout_len, obs_size)),
        action=jnp.zeros((rollout_len, action_size)),
        reward=jnp.zeros((rollout_len,)),
        terminal=jnp.zeros((rollout_len,)),
        timeout=jnp.zeros((rollout_len,)),
    )
    keys = jrd.split(key, rollout_len)

    def step_fn(index, step_data):
        rollout, obs, episode_step = step_data
        *step_data, _obs, episode_step = jaxdp.async_sample_step_pi(
            mdp, policy, obs, episode_step, max_episode_length, keys[index])
        names = ("action", "next_obs", "reward",
                 "terminal", "timeout", "obs")
        for name, data in zip(names, (*step_data, obs)):
            rollout[name] = rollout[name].at[index].set(data)

        return (rollout, _obs, episode_step)

    rollout, obs, episode_step = jax.lax.fori_loop(
        0, rollout_len, step_fn, (rollout, obs, episode_step))

    return (
        RolloutSample(**rollout),
        SamplerState(
            obs,
            episode_step,
            *update_sampler_records(rollout, sampler_state)
        ))


class Policy(Protocol):

    @staticmethod
    def reset(key: KeyType, state: Any) -> Any: pass
    @staticmethod
    def sample(key: KeyType, state: Any, obs: F["S"]) -> Tuple[F["A"], Any]: pass


def sample_gymnax_rollout(key: KeyType,
                          sampler_state: SamplerState,
                          env_state: EnvState,
                          policy_state: Any,
                          env_param: EnvParams,
                          env: Environment,
                          policy: Policy,
                          rollout_length: int):

    rollout = RolloutSample.full(env.obs_shape[0], 1, rollout_length)

    def step(index, step_data):
        rollout, key, last_obs, env_state, policy_state = step_data
        key, env_key, policy_key, policy_reset_key = jrd.split(key, 4)
        action, policy_state = policy.sample(policy_key, policy_state, last_obs)
        next_obs, env_state, reward, done, info = env.step(env_key, env_state, action, env_param)

        rollout = RolloutSample(
            rollout.obs.at[index].set(last_obs),
            rollout.next_obs.at[index].set(next_obs),
            rollout.action.at[index].set(action),
            rollout.reward.at[index].set(reward),
            rollout.terminal.at[index].set(done),
            rollout.timeout.at[index].set(done),
        )

        policy_state = jax.lax.cond(
            done,
            policy.reset,
            lambda key, state: state,
            policy_reset_key, policy_state)

        return (rollout, key, next_obs, env_state, policy_state)

    rollout, key, last_obs, env_state, policy_state = jax.lax.fori_loop(
        0, rollout_length, step,
        (rollout, key, sampler_state.last_obs, env_state, policy_state)
    )

    sampler_state = SamplerState(
        last_obs,
        *update_sampler_records(rollout, sampler_state)
    )
    return (key, rollout, sampler_state, env_state, policy_state)
