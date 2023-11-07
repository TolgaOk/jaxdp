from typing import Optional, Callable
import jax.numpy as jnp
import jax.random as jrd
import jax
import distrax
from jaxtyping import Array, Float


class BaseDP():

    def __init__(self,
                 transition: Float[Array, "A S S"],
                 reward: Float[Array, "A S"],
                 initial: Float[Array, "S"],
                 terminal: Float[Array, "S"],
                 episode_length: int
                 ) -> None:
        self.transition = transition
        self.reward = reward
        self.initial = initial
        self.terminal = terminal
        self.episode_length = episode_length

        self.state_size = self.transition.shape[1]
        self.action_size = self.transition.shape[0]

        if not jnp.allclose(transition.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Transition matrix must be column stochastic.")

    def greedy_policy(self, values: Float[Array, "A S"]) -> Float[Array, "A S"]:
        return jax.nn.one_hot(jnp.argmax(values, axis=0, keepdims=False),
                              num_classes=self.action_size,
                              axis=0)
        return (jnp.argmax(values, axis=0, keepdims=True) ==
                jnp.arange(self.action_size).reshape(-1, 1)).astype("float32")

    def soft_policy(self, values: Float[Array, "A S"], key: jrd.KeyArray) -> Float[Array, "A S"]:
        return distrax.OneHotCategorical(logits=values.T).sample(seed=key).T

    def e_greedy_policy(self,
                        values: Float[Array, "A S"],
                        epsilon: float,
                        key: jrd.KeyArray
                        ) -> Float[Array, "A S"]:
        random_act = distrax.OneHotCategorical(logits=jnp.zeros(values.shape).T).sample(seed=key).T
        greedy_act = self.greedy_policy(values)
        mask = jrd.bernoulli(p=epsilon, shape=self.state_size).reshape(1, -1)
        return random_act * mask + (1 - mask) * greedy_act

    def async_sample(self,
                     policy_fn: Callable[[jrd.KeyArray], Float[Array, "A S"]],
                     length: int,
                     n_sample: int,
                     key: jrd.KeyArray,
                     initial_state: Optional[Float[Array, "S"]] = None
                     ) -> Float[Array, "S N"]:
        trajectory = {
            "state": jnp.empty((n_sample, length, self.state_size)),
            "next_state": jnp.empty((n_sample, length, self.state_size)),
            "action": jnp.empty((n_sample, length, self.action_size)),
            "reward": jnp.empty((n_sample, length)),
            "terminal": jnp.empty((n_sample, length)),
            "timeout": jnp.empty((n_sample, length)),
        }

        def _sample_init_state(subkey: jrd.KeyArray, n_sample: int) -> Float[Array, "S"]:
            return distrax.OneHotCategorical(
                probs=self.initial).sample(
                    seed=subkey, sample_shape=(n_sample,))

        key, subkey = jrd.split(key, num=2)
        if initial_state is None:
            state = _sample_init_state(subkey, n_sample)
        else:
            state = initial_state

        # actions = self.greedy_policy(value)
        actions = policy_fn()
        episode_step = jnp.zeros((n_sample,))
        for step in range(length):
            action = jnp.einsum("as,bs->ba", actions, state)
            reward = jnp.einsum("as,as,bs->b", self.reward, actions, state)
            next_state_p = jnp.einsum("as,axs,bs->bx", actions, self.transition, state)
            key, subkey = jrd.split(key, num=2)
            next_state = distrax.OneHotCategorical(probs=next_state_p).sample(seed=subkey)

            episode_step = episode_step + 1
            timeout = episode_step >= self.episode_length
            terminal = jnp.einsum("s,bs->b", self.terminal, next_state)
            done = jnp.logical_or(terminal, timeout)
            episode_step = episode_step * (1 - done)

            key, subkey = jrd.split(key, num=2)
            init_state = _sample_init_state(subkey, n_sample)

            for array, name in ([state, "state"],
                                [next_state, "next_state"],
                                [action, "action"],
                                [reward, "reward"],
                                [timeout, "timeout"],
                                [terminal, "terminal"]):
                trajectory[name] = trajectory[name].at[:, step].set(array)

            state = next_state * (1 - done.reshape(-1, 1)) + init_state * (done.reshape(-1, 1))
        return trajectory

    def evaluate(self, value: Float[Array, "A S"]) -> float:
        return (jnp.max(value, axis=0) * self.initial).sum()


class ValueIteration(BaseDP):

    def sync_learn(self, steps: int, gamma: float = 0.99) -> Float[Array, "S A"]:
        values = jnp.zeros((self.action_size, self.state_size))
        info = {
            "performance": [],
            "values": []
        }

        for step in range(steps):
            target_values = (gamma * jnp.einsum(
                "ux,axs,x->uas", values, self.transition, (1 - self.terminal)))
            values = self.reward * (1 - self.terminal).reshape(1, -1) + \
                jnp.max(target_values, axis=0)
            info["performance"].append(self.evaluate(values))
            info["values"].append(values)

        return values, info

    def async_learn(self,
                    n_trajectories: int,
                    trajectory_length: int,
                    reset_trajectory_at_each_iteration: bool,
                    n_steps: int,
                    gamma: float):
        values = jnp.zeros((self.action_size, self.state_size))
        info = {
            "performance": [],
            "values": []
        }

        trajectories = self.async_sample(
            value=values,
        )


class PolicyIteration(BaseDP):
    pass


class QuasiPolicyIteration(BaseDP):
    pass
