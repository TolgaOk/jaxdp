"""Sampling from Gymnax environments."""

from dataclasses import replace
from typing import Generic, TypeVar

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
from gymnax.environments.environment import Environment, EnvParams, EnvState

from jaxdp.mdp.sampler.mdp import _update_episode

EnvStateT = TypeVar("EnvStateT", bound=EnvState)
EnvParamsT = TypeVar("EnvParamsT", bound=EnvParams)


@chex.dataclass(frozen=True)
class State(Generic[EnvStateT]):
    """Gymnax sampler state."""

    last_obs: jax.Array
    env: EnvStateT
    episode_step: jax.Array
    rewards: jax.Array
    lengths: jax.Array
    episode_reward_queue: jax.Array
    episode_length_queue: jax.Array


@chex.dataclass(frozen=True)
class RolloutData:
    """One transition or a time-major batch of transitions."""

    obs: jax.Array
    next_obs: jax.Array
    action: jax.Array
    reward: jax.Array
    terminal: jax.Array
    timeout: jax.Array


def step(
    key: chex.PRNGKey,
    action: int | float | jax.Array,
    state: State[EnvStateT],
    env_param: EnvParamsT,
    env: Environment[EnvStateT, EnvParamsT],
    max_episode_length: int,
) -> tuple[RolloutData, State[EnvStateT]]:
    """Sample one transition and advance the auto-resetting sampler state."""
    env_step_key, env_reset_key = jrd.split(key)
    next_obs, stepped_env_state, reward, terminal, _ = env.step_env(
        env_step_key,
        state.env,
        action,
        env_param,
    )

    next_episode_step = state.episode_step + 1
    timeout = next_episode_step >= max_episode_length
    episode_end = jnp.logical_or(terminal, timeout)
    reset_obs, reset_env_state = env.reset_env(env_reset_key, env_param)

    def select(reset: jax.Array, stepped: jax.Array) -> jax.Array:
        return jax.lax.select(episode_end, reset, stepped)

    next_sample_env = jax.tree.map(select, reset_env_state, stepped_env_state)
    next_sample_obs = jax.lax.select(episode_end, reset_obs, next_obs)
    episode_step = jnp.where(episode_end, 0, next_episode_step)
    rewards, lengths, reward_queue, length_queue = _update_episode(
        state.rewards,
        state.lengths,
        state.episode_reward_queue,
        state.episode_length_queue,
        reward,
        terminal,
        timeout,
    )

    step_data = RolloutData(
        obs=state.last_obs,
        next_obs=next_obs,
        action=jnp.asarray(action),
        reward=reward,
        terminal=terminal,
        timeout=timeout,
    )
    next_state = replace(
        state,
        last_obs=next_sample_obs,
        env=next_sample_env,
        episode_step=episode_step,
        rewards=rewards,
        lengths=lengths,
        episode_reward_queue=reward_queue,
        episode_length_queue=length_queue,
    )
    return step_data, next_state


def init_sampler_state(
    init_obs: jax.Array,
    env_state: EnvStateT,
    queue_size: int,
) -> State[EnvStateT]:
    """Initialize sampler state and empty episode-statistic queues."""
    return State(
        last_obs=init_obs,
        env=env_state,
        episode_step=jnp.array(0),
        rewards=jnp.array(0.0),
        lengths=jnp.array(0),
        episode_reward_queue=jnp.full(queue_size, jnp.nan),
        episode_length_queue=jnp.full(queue_size, jnp.nan),
    )


def init_rollout(obs_size: int, action_size: int, rollout_len: int) -> RolloutData:
    """Allocate an empty time-major rollout."""
    return RolloutData(
        obs=jnp.full((rollout_len, obs_size), jnp.nan),
        next_obs=jnp.full((rollout_len, obs_size), jnp.nan),
        action=jnp.full((rollout_len, action_size), jnp.nan),
        reward=jnp.full(rollout_len, jnp.nan),
        terminal=jnp.full(rollout_len, jnp.nan),
        timeout=jnp.full(rollout_len, jnp.nan),
    )


__all__ = ["State", "RolloutData", "step", "init_sampler_state", "init_rollout"]
