"""Sampling from finite MDPs."""

from dataclasses import replace

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd

import jaxdp
from jaxdp.mdp import Mdp


@chex.dataclass(frozen=True)
class State:
    """MDP sampler state."""

    last_state: jax.Array
    episode_step: jax.Array
    rewards: jax.Array
    lengths: jax.Array
    episode_reward_queue: jax.Array
    episode_length_queue: jax.Array


@chex.dataclass(frozen=True)
class RolloutData:
    """One transition or a time-major batch of transitions."""

    state: jax.Array
    next_state: jax.Array
    action: jax.Array
    reward: jax.Array
    terminal: jax.Array
    timeout: jax.Array


def rollout(
    key: chex.PRNGKey,
    sampler_state: State,
    policy: jax.Array,
    mdp: Mdp,
    rollout_len: int,
    max_episode_len: int,
) -> tuple[RolloutData, State]:
    """Sample a fixed-length rollout and return its final sampler state."""
    rollout_data = init_rollout(mdp.state_size, mdp.action_size, rollout_len)
    step_keys = jrd.split(key, rollout_len)

    def step_sample(
        index: jax.Array,
        payload: tuple[RolloutData, State],
    ) -> tuple[RolloutData, State]:
        data, state = payload
        step_data, state = step(
            step_keys[index],
            state,
            policy,
            mdp,
            max_episode_len,
        )

        def write(array: jax.Array, value: jax.Array) -> jax.Array:
            return array.at[index].set(value)

        return jax.tree.map(write, data, step_data), state

    return jax.lax.fori_loop(
        0,
        rollout_len,
        step_sample,
        (rollout_data, sampler_state),
    )


def _queue_push(queue: jax.Array, value: jax.Array, condition: jax.Array) -> jax.Array:
    pushed = queue.at[1:].set(queue[:-1]).at[0].set(value)
    return jax.lax.select(condition, pushed, queue)


def _update_episode(
    rewards: jax.Array,
    lengths: jax.Array,
    reward_queue: jax.Array,
    length_queue: jax.Array,
    reward: jax.Array,
    terminal: jax.Array,
    timeout: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    done = jnp.logical_or(terminal, timeout)
    total_reward = reward + rewards
    total_length = 1 + lengths
    return (
        jnp.where(done, 0, total_reward),
        jnp.where(done, 0, total_length),
        _queue_push(reward_queue, total_reward, done),
        _queue_push(length_queue, total_length, done),
    )


def _update_state(state: State, step_data: RolloutData) -> State:
    rewards, lengths, reward_queue, length_queue = _update_episode(
        state.rewards,
        state.lengths,
        state.episode_reward_queue,
        state.episode_length_queue,
        step_data.reward,
        step_data.terminal,
        step_data.timeout,
    )
    return replace(
        state,
        rewards=rewards,
        lengths=lengths,
        episode_reward_queue=reward_queue,
        episode_length_queue=length_queue,
    )


def step(
    key: chex.PRNGKey,
    sampler_state: State,
    policy: jax.Array,
    mdp: Mdp,
    max_episode_len: int,
) -> tuple[RolloutData, State]:
    """Sample one transition and advance the continuing sampler state."""
    _, step_key = jrd.split(key)
    (
        action,
        next_state,
        reward,
        terminal,
        timeout,
        continuation_state,
        episode_step,
    ) = jaxdp.async_sample_step_pi(
        mdp,
        policy,
        sampler_state.last_state,
        sampler_state.episode_step,
        max_episode_len,
        step_key,
    )
    step_data = RolloutData(
        state=sampler_state.last_state,
        next_state=next_state,
        action=action,
        reward=reward,
        terminal=terminal,
        timeout=timeout,
    )
    next_sampler_state = replace(
        _update_state(sampler_state, step_data),
        last_state=continuation_state,
        episode_step=episode_step,
    )
    return step_data, next_sampler_state


def init_sampler_state(key: chex.PRNGKey, mdp: Mdp, queue_size: int) -> State:
    """Initialize sampler state and empty episode-statistic queues."""
    return State(
        last_state=mdp.init_state(key),
        episode_step=jnp.array(0),
        rewards=jnp.array(0.0),
        lengths=jnp.array(0),
        episode_reward_queue=jnp.full(queue_size, jnp.nan),
        episode_length_queue=jnp.full(queue_size, jnp.nan),
    )


def init_rollout(state_size: int, action_size: int, rollout_len: int) -> RolloutData:
    """Allocate an empty time-major rollout."""
    return RolloutData(
        state=jnp.full((rollout_len, state_size), jnp.nan),
        next_state=jnp.full((rollout_len, state_size), jnp.nan),
        action=jnp.full((rollout_len, action_size), jnp.nan),
        reward=jnp.full(rollout_len, jnp.nan),
        terminal=jnp.full(rollout_len, jnp.nan),
        timeout=jnp.full(rollout_len, jnp.nan),
    )


def refresh_queues(state: State) -> State:
    """Clear completed-episode statistics."""
    return replace(
        state,
        episode_reward_queue=jnp.full_like(state.episode_reward_queue, jnp.nan),
        episode_length_queue=jnp.full_like(state.episode_length_queue, jnp.nan),
    )


__all__ = [
    "State",
    "RolloutData",
    "rollout",
    "step",
    "init_sampler_state",
    "init_rollout",
    "refresh_queues",
]
