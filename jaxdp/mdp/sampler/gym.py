from typing import Any, Tuple, Union
import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct
from jax.typing import ArrayLike as KeyType
from gymnax.environments.environment import Environment, EnvParams, EnvState

from jaxdp.typehints import F, I
from jaxdp.mdp.sampler.mdp import _update_state, refresh_queues


@struct.dataclass
class State:
    last_obs: F["S"]
    env: EnvState
    episode_step: I[""]
    rewards: F[""]
    lengths: I[""]
    episode_reward_queue: F["K"]
    episode_length_queue: F["K"]


@struct.dataclass
class RolloutData:
    obs: Union[F["T S"], F["S"]]
    next_obs: Union[F["T S"], F["S"]]
    action: Union[F["T A"], F["A"]]
    reward: Union[F["T"], F[""]]
    terminal: Union[F["T"], F[""]]
    timeout: Union[F["T"], F[""]]


def step(key: KeyType,
         action: F["A"],
         state: State,
         env_param: EnvParams,
         env: Environment,
         max_episode_length: int
         ) -> Tuple[RolloutData, State]:

    env_step_key, env_reset_key = jrd.split(key, 2)
    next_obs_st, next_env_state_st, reward, terminal, info = env.step(
        env_step_key, state.env, action, env_param)

    # Decide end of rollout condition
    episode_step = (state.episode_step + 1)
    timeout = episode_step >= max_episode_length
    done = jnp.logical_or(terminal, timeout)
    episode_step = episode_step * (1 - done)

    # Auto-reset environment based on done
    next_obs_re, next_env_state_re = env.reset_env(env_reset_key, env_param)
    next_env_state = jax.tree_map(
        lambda x, y: jax.lax.select(done, x, y), next_env_state_re, next_env_state_st
    )
    next_obs = jax.lax.select(done, next_obs_re, next_obs_st)

    step_data = RolloutData(
        state.last_obs,
        next_obs_st,
        action,
        reward,
        terminal,
        timeout,
    )
    state = _update_state(state, step_data).replace(
        last_obs=next_obs, episode_step=episode_step, env=next_env_state)

    return step_data, state


def init_sampler_state(init_obs: F["S"], env_state: EnvState, queue_size: int) -> State:
    return State(
        init_obs,
        env_state,
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.full((queue_size,), jnp.nan),
        jnp.full((queue_size,), jnp.nan),
    )


def init_rollout(obs_size: int, action_size: int, rollout_len: int) -> State:
    return RolloutData(
        obs=jnp.full((rollout_len, obs_size), jnp.nan),
        next_obs=jnp.full((rollout_len, obs_size), jnp.nan),
        action=jnp.full((rollout_len, action_size), jnp.nan),
        reward=jnp.full((rollout_len,), jnp.nan),
        terminal=jnp.full((rollout_len,), jnp.nan),
        timeout=jnp.full((rollout_len,), jnp.nan),
    )
