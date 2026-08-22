"""Policy components for finite Markov decision processes."""

import math
from typing import Protocol

import chex
import jax
import jax.numpy as jnp


class Mdp(Protocol):
    """Tabular dynamics consumed by value-based policies."""

    transition: jax.Array
    reward: jax.Array


class Policy(Protocol):
    """Policy constructed from action or state values."""

    def q(self, value: jax.Array) -> jax.Array: ...
    def v(self, mdp: Mdp, value: jax.Array, gamma: float) -> jax.Array: ...


def _state_action_value(mdp: Mdp, value: jax.Array, gamma: float) -> jax.Array:
    reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
    continuation = jnp.einsum("axs,x->as", mdp.transition, value)
    return reward + gamma * continuation


def _greedy(value: jax.Array) -> jax.Array:
    return jax.nn.one_hot(
        jnp.argmax(value, axis=0),
        value.shape[0],
        axis=0,
    )


@chex.dataclass(frozen=True)
class Greedy:
    """Greedy value-based policy."""

    def q(self, value: jax.Array) -> jax.Array:
        """Return a greedy policy for an ``(A, S)`` action-value array."""
        return _greedy(value)

    def v(self, mdp: Mdp, value: jax.Array, gamma: float) -> jax.Array:
        """Return a greedy policy after one-step state-value lookahead."""
        return self.q(_state_action_value(mdp, value, gamma))


@chex.dataclass(frozen=True)
class Soft:
    """Softmax value-based policy.

    Attributes:
        temperature: Positive softmax temperature.
    """

    temperature: float

    def __post_init__(self) -> None:
        """Validate the temperature."""
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

    def q(self, value: jax.Array) -> jax.Array:
        """Return a softmax policy for an ``(A, S)`` action-value array."""
        return jax.nn.softmax(value / self.temperature, axis=0)

    def v(self, mdp: Mdp, value: jax.Array, gamma: float) -> jax.Array:
        """Return a softmax policy after one-step state-value lookahead."""
        return self.q(_state_action_value(mdp, value, gamma))


@chex.dataclass(frozen=True)
class EpsilonGreedy:
    """Epsilon-greedy value-based policy.

    Attributes:
        epsilon: Uniform exploration probability in the closed interval ``[0, 1]``.
    """

    epsilon: float

    def __post_init__(self) -> None:
        """Validate epsilon."""
        if not math.isfinite(self.epsilon) or not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be finite and between zero and one")

    def q(self, value: jax.Array) -> jax.Array:
        """Return an epsilon-greedy policy for an ``(A, S)`` action-value array."""
        greedy = _greedy(value)
        return (1 - self.epsilon) * greedy + self.epsilon / value.shape[0]

    def v(self, mdp: Mdp, value: jax.Array, gamma: float) -> jax.Array:
        """Return an epsilon-greedy policy after one-step state-value lookahead."""
        return self.q(_state_action_value(mdp, value, gamma))


__all__ = ["Policy", "Greedy", "Soft", "EpsilonGreedy"]
