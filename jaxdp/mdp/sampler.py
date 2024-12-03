from typing import Any, Tuple, Union
import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct
from jax.typing import ArrayLike as KeyType
from gymnax.environments.environment import Environment, EnvParams, EnvState

import jaxdp
from jaxdp import MDP
from jaxdp.mdp import MDP
from jaxdp.typehints import F, PiType
from jaxdp.utils import StaticMeta


@struct.dataclass
class State:
    last_state: F["S"]
    episode_step: F[""]
    rewards: F[""]
    lengths: F[""]
    episode_reward_queue: F["K"]
    episode_length_queue: F["K"]


@struct.dataclass
class RolloutData:
    state: Union[F["T S"], F["S"]]
    next_state: Union[F["T S"], F["S"]]
    action: Union[F["T A"], F["A"]]
    reward: Union[F["T"], F[""]]
    terminal: Union[F["T"], F[""]]
    timeout: Union[F["T"], F[""]]


def rollout(key: KeyType,
            sampler_state: State,
            policy: PiType,
            mdp: MDP,
            rollout_len: int,
            max_episode_len: int
            ) -> Tuple[RolloutData, State]:

    rollout = init_rollout(
        mdp.state_size,
        mdp.action_size,
        rollout_len)
    step_keys = jrd.split(key, rollout_len)

    def step_sample(i: int, payload):
        rollout, state = payload
        step_data, state = step(
            step_keys[i],
            state,
            policy,
            mdp,
            max_episode_len)
        rollout = jax.tree.map(lambda x, y: x.at[i].set(y), rollout, step_data)
        return rollout, state

    return jax.lax.fori_loop(0, rollout_len, step_sample, (rollout, sampler_state))


def step(key: KeyType,
         sampler_state: State,
         policy: PiType,
         mdp: MDP,
         max_episode_len: int
         ) -> Tuple[RolloutData, State]:

    key, step_key = jrd.split(key)

    *sample_data, new_last_state, episode_step = jaxdp.async_sample_step_pi(
        mdp,
        policy,
        sampler_state.last_state,
        sampler_state.episode_step,
        max_episode_len,
        step_key
    )
    names = ("action", "next_state", "reward",
             "terminal", "timeout", "state")
    step_data_dict = {name: data for name, data in
                      zip(names, (*sample_data, sampler_state.last_state))}
    step_data = RolloutData(**step_data_dict)

    def queue_push(queue_array, value, condition):
        pushed_array = queue_array.at[1:].set(queue_array[:-1]).at[0].set(value)
        no_nan_queue = jnp.nan_to_num(queue_array)
        return (pushed_array * condition +
                (no_nan_queue / (1 - jnp.isnan(queue_array) * (1 - condition))) * (1 - condition))

    def update_state(state, step_data: RolloutData) -> State:
        done = jnp.logical_or(step_data.terminal, step_data.timeout)
        rewards = step_data.reward + state.rewards
        lengths = 1 + state.lengths
        return state.replace(
            episode_reward_queue=queue_push(state.episode_reward_queue, rewards, done),
            episode_length_queue=queue_push(state.episode_length_queue, lengths, done),
            rewards=rewards * (1 - done),
            lengths=lengths * (1 - done),
        )

    sampler_state = update_state(
        sampler_state, step_data
    ).replace(
        last_state=new_last_state,
        episode_step=episode_step)

    return step_data, sampler_state


def init_sampler_state(key: KeyType, mdp: MDP, queue_size: int) -> State:
    init_state = mdp.init_state(key)
    return State(
        init_state,
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.full((queue_size,), jnp.nan),
        jnp.full((queue_size,), jnp.nan),
    )


def init_rollout(state_size: int, action_size: int, rollout_len: int) -> State:
    return RolloutData(
        state=jnp.full((rollout_len, state_size), jnp.nan),
        next_state=jnp.full((rollout_len, state_size), jnp.nan),
        action=jnp.full((rollout_len, action_size), jnp.nan),
        reward=jnp.full((rollout_len,), jnp.nan),
        terminal=jnp.full((rollout_len,), jnp.nan),
        timeout=jnp.full((rollout_len,), jnp.nan),
    )


def refresh_queues(state: State) -> State:
    return state.replace(
        episode_reward_queue=jnp.full_like(state.episode_reward_queue, jnp.nan),
        episode_length_queue=jnp.full_like(state.episode_length_queue, jnp.nan),
    )

