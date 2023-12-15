from typing import Dict, Union, List, NamedTuple
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

        self.batch_async_sample = jax.jit(
            jax.vmap(
                jaxdp.async_sample_step_pi,
                (None, None, 0, 0, None, 0), 0)
        )

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

    def step_sample(self, policy: Float[Array, "A S"], key: ArrayLike) -> RolloutSample:
        keys = jrd.split(key, num=self.batch_size)
        state = self._state
        (action,
         next_state,
         reward,
         terminal,
         timeout,
         self._state,
         self._episode_step
         ) = self.batch_async_sample(
             self.mdp,
            policy,
            self._state,
            self._episode_step,
            self.max_episode_length,
            keys)
        self._rewards = self._rewards + reward
        self._lengths = self._lengths + 1
        done = jnp.logical_or(terminal, timeout)
        self._episodic_reward += self._rewards[jnp.argwhere(
            done).flatten()].tolist()
        self._episode_length += self._lengths[jnp.argwhere(
            done).flatten()].tolist()

        self._rewards = self._rewards * \
            (1 - done) + jnp.zeros(self.batch_size) * done
        self._lengths = self._lengths * \
            (1 - done) + jnp.zeros(self.batch_size) * done

        return RolloutSample(
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            terminal=terminal,
            timeout=timeout,
        )

    def rollout_sample(self, policy: Float[Array, "A S"], key: ArrayLike) -> RolloutSample:

        step_samples = []

        for _ in range(self.rollout_len):
            key, step_key = jrd.split(key, 2)
            step_samples.append(
                self.step_sample(policy, step_key)
            )

        return jax.tree_util.tree_map(
            lambda *steps: jnp.stack(steps, axis=1), *step_samples)
