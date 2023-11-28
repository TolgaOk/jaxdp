from typing import Tuple, Any, Union, Type
from gym.spaces import Box, Discrete
import jax.numpy as jnp
from jax import random as jrandom
import jax.nn
import distrax


class GarnetMDP():
    # TODO: Add test
    # TODO: Add documentation

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 temperature: float,
                 episode_length: int = 10,
                 seed: int = 42) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.temperature = temperature
        self.episode_length = episode_length
        self.init_seed = seed

        self.observation_space = Discrete(n=state_size, seed=seed+1)
        self.action_space = Discrete(n=action_size, seed=seed+2)

        if temperature < 0:
            raise ValueError(
                f"Temperature: <{temperature}> must be non-negative!")

        p_subkey, rho_subkey, r_subkey = jrandom.split(
            jrandom.PRNGKey(seed), num=3)

        self.transition_p = self._stochastic_array(
            logits=jrandom.uniform(
                p_subkey, (action_size, state_size, state_size)),
            temperature=temperature,
            axis=-1)
        self.initial_p = self._stochastic_array(
            logits=jrandom.uniform(rho_subkey, (state_size,)),
            temperature=temperature,
            axis=-1)
        self.rewards = jrandom.uniform(r_subkey, (state_size, action_size))

        self._step = None
        self.state = None
        self.reset_key = None

    @staticmethod
    def _stochastic_array(logits, temperature: float, axis: int) -> jnp.array:
        raw_values = jnp.exp(logits * temperature)
        return raw_values / raw_values.sum(axis=axis, keepdims=True)

    def _preprocess_action(self, action: jnp.array) -> jnp.array:
        return jax.nn.one_hot(action, self.action_size, axis=-1).flatten()

    def _post_process_state(self, state: jnp.array) -> jnp.array:
        return jnp.argmax(state, axis=-1)

    def step(self, action: jnp.array) -> Tuple[Union[jnp.array, jnp.float32, bool, Any]]:
        if self._step is None:
            raise RuntimeError("First call reset!")
        self._step += 1

        action = self._preprocess_action(action)

        next_state_p, reward = self.transition(self.state, action)
        self.reset_key, subkey = jrandom.split(self.reset_key)
        self.state = self.next_state_sample(next_state_p, subkey)
        timeout = self._step == self.episode_length
        terminal = False

        self._step = self._step if not timeout else None
        if timeout:
            self.reset_key = None
        return self._post_process_state(self.state), reward, terminal, timeout, {}

    def transition(self, state: jnp.array, action: jnp.array) -> Tuple[jnp.array]:
        p_pi = jnp.einsum("a,axy->xy", action, self.transition_p)
        next_state_p = jnp.einsum("yx,x", p_pi, state)
        reward = jnp.einsum("xa,a,x->", self.rewards, action, state)
        return next_state_p, reward

    def next_state_sample(self, dist_probs: jnp.array, key: jrandom.KeyArray, shape: Tuple[int] = ()) -> jnp.array:
        next_state_dist = distrax.OneHotCategorical(probs=dist_probs)
        return next_state_dist.sample(seed=key, sample_shape=shape)

    def reset(self, seed: int, shape: Tuple[int] = ()) -> jnp.array:
        self.reset_key = jrandom.PRNGKey(seed)
        self.reset_key, subkey = jrandom.split(self.reset_key)
        self._step = 0
        self.state = distrax.OneHotCategorical(
            probs=self.initial_p).sample(seed=subkey, sample_shape=shape)
        return self._post_process_state(self.state)


class OneHotGarnetMDP(GarnetMDP):

    def __init__(self, state_size: int, action_size: int, temperature: float, episode_length: int = 10, seed: int = 42) -> None:
        super().__init__(state_size, action_size, temperature, episode_length, seed)

        self.observation_space = Box(low=0, high=1, shape=(self.state_size,))
        self.action_space = Box(low=0, high=1, shape=(self.action_size,))

    def _post_process_state(self, state: jnp.array) -> jnp.array:
        return state

    def _preprocess_action(self, action: jnp.array) -> jnp.array:
        return action
