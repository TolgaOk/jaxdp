from typing import Dict, Union, List, NamedTuple, Tuple
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


@partial(jax.jit, static_argnames="length")
def _rollout_sample(length: int,
                    mdp: MDP,
                    policy: Float[Array, "A S"],
                    state: Float[Array, "... S"],
                    episode_step: Float[Array, "..."],
                    max_episode_step: int,
                    key: jrd.KeyArray
                    ) -> Tuple[RolloutSample,
                               Float[Array, "... S"],
                               Float[Array, "..."]]:

    batch_size = state.shape[0]
    state_size = mdp.state_size
    action_size = mdp.action_size
    rollout = dict(
        state=jnp.zeros((batch_size, length, state_size)),
        next_state=jnp.zeros((batch_size, length, state_size)),
        action=jnp.zeros((batch_size, length, action_size)),
        reward=jnp.zeros((batch_size, length)),
        terminal=jnp.zeros((batch_size, length)),
        timeout=jnp.zeros((batch_size, length)),
    )
    batch_async_sample = jax.vmap(
        jaxdp.async_sample_step_pi,
        (None, None, 0, 0, None, 0), 0)
    batch_split = jax.vmap(jrd.split, (0, None), 1)
    keys = batch_split(jrd.split(key, batch_size), length)

    def step_fn(index, step_data):
        rollout, state, episode_step = step_data
        *step_data, _state, episode_step = batch_async_sample(
            mdp, policy, state, episode_step, max_episode_step, keys[index])
        names = ("action", "next_state", "reward",
                 "terminal", "timeout", "state")
        for name, data in zip(names, (*step_data, state)):
            rollout[name] = rollout[name].at[:, index].set(data)

        return (rollout, _state, episode_step)

    rollout, state, episode_step = jax.lax.fori_loop(
        0, length, step_fn, (rollout, state, episode_step))

    return RolloutSample(**rollout), state, episode_step


class Sampler():

    def __init__(self,
                 mdp: MDP,
                 rollout_len: int,
                 batch_size: int,
                 max_episode_length: int,
                 init_key: ArrayLike
                 ) -> None:
        self.mdp = mdp
        self.state = None
        self.rollout_len = rollout_len
        self.batch_size = batch_size
        self.max_episode_length = max_episode_length

        self._state = distrax.OneHotCategorical(
            probs=mdp.initial).sample(seed=init_key, sample_shape=(batch_size,))
        self._episode_step = jnp.zeros((batch_size,))
        self._episodic_reward = []
        self._episode_length = []
        self._rewards = jnp.zeros(batch_size)
        self._lengths = jnp.zeros(batch_size)

    @property
    def recent_episode_rewards(self):
        rewards = self._episodic_reward
        self._episodic_reward = []
        return rewards

    @property
    def recent_episode_lengths(self):
        lengths = self._episode_length
        self._episode_length = []
        return lengths

    def rollout_sample(self, length: int, policy: Float[Array, "A S"], key: jrd.KeyArray,
                       log_rewards: bool = True
                       ) -> RolloutSample:
        rollout, self._state, self._episode_step = _rollout_sample(
            length, self.mdp, policy, self._state,
            self._episode_step, self.max_episode_length, key)
        
        if log_rewards:
            dones = jnp.logical_or(rollout.terminal, rollout.timeout)
            for index in range(length):
                done = dones[:, index]
                self._rewards = self._rewards + rollout.reward[:, index]
                self._lengths = self._lengths + 1
                self._episodic_reward += self._rewards[jnp.argwhere(done).flatten()].tolist()
                self._episode_length += self._lengths[jnp.argwhere(done).flatten()].tolist()
                self._rewards = self._rewards * (1 - done) + jnp.zeros(self.batch_size) * done
                self._lengths = self._lengths * (1 - done) + jnp.zeros(self.batch_size) * done

        return rollout
