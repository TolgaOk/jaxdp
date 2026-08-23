from typing import Any, Never

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments.environment import Environment, EnvParams, EnvState

from jaxdp.mdp.sampler.gym import init_sampler_state, step


@struct.dataclass
class MockState(EnvState):
    x: jax.Array


class MockEnvironment(Environment[MockState, EnvParams]):
    def step(
        self,
        key: jax.Array,
        state: MockState,
        action: int | float | jax.Array,
        params: EnvParams | None = None,
    ) -> Never:
        raise AssertionError("sampler must call step_env to retain the transition successor")

    def step_env(
        self,
        key: jax.Array,
        state: MockState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, MockState, jax.Array, jax.Array, dict[Any, Any]]:
        del key, action, params
        next_state = MockState(time=state.time + 1, x=jnp.array(1))
        return jnp.array(1), next_state, jnp.array(2.0), jnp.array(True), {}

    def reset_env(
        self,
        key: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, MockState]:
        del key, params
        return jnp.array(0), MockState(time=0, x=jnp.array(0))


def test_step_keeps_gym_successor_separate_from_reset_observation() -> None:
    env = MockEnvironment()
    state = init_sampler_state(
        jnp.array(0),
        MockState(time=0, x=jnp.array(0)),
        queue_size=2,
    )

    data, next_state = step(
        jax.random.key(0),
        jnp.array(0),
        state,
        EnvParams(max_steps_in_episode=10),
        env,
        max_episode_length=10,
    )

    assert data.obs == 0
    assert data.next_obs == 1
    assert data.reward == 2.0
    assert data.terminal
    assert next_state.last_obs == 0
    assert next_state.env.x == 0
    assert next_state.episode_step == 0
    assert next_state.rewards == 0.0
    assert next_state.lengths == 0
    assert next_state.episode_reward_queue[0] == 2.0
    assert next_state.episode_length_queue[0] == 1
