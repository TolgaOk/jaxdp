"""Policy components for finite Markov decision processes."""

from typing import Protocol

import chex
import jax
import jax.numpy as jnp

from jaxdp.mdp.mdp import MDP
from jaxdp.operator import state_action_value


class Policy(Protocol):
    """Policy constructed from action or state values."""

    def q(self, value: jax.Array) -> jax.Array: ...
    def v(self, mdp: MDP, value: jax.Array, gamma: float | jax.Array) -> jax.Array: ...


@chex.dataclass(frozen=True)
class Greedy:
    """Greedy value-based policy."""

    def q(self, value: jax.Array) -> jax.Array:
        """Return a greedy policy for an ``(A, S)`` action-value array."""
        return _greedy(value)

    def v(self, mdp: MDP, value: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Return a greedy policy after one-step state-value lookahead."""
        return self.q(state_action_value(mdp, value, gamma))


@chex.dataclass(frozen=True)
class Soft:
    """Softmax value-based policy.

    Attributes:
        temperature: Positive softmax temperature.
    """

    temperature: float

    def __post_init__(self) -> None:
        """Validate the temperature."""
        temperature = jnp.asarray(self.temperature)
        chex.assert_shape(temperature, (), custom_message="temperature must be scalar")
        chex.assert_tree_all_finite(temperature, custom_message="temperature must be finite")
        chex.assert_trees_all_equal(
            temperature > 0,
            jnp.asarray(True),
            custom_message="temperature must be positive",
        )

    def q(self, value: jax.Array) -> jax.Array:
        """Return a softmax policy for an ``(A, S)`` action-value array."""
        chex.assert_rank(value, 2)
        return jax.nn.softmax(value / self.temperature, axis=0)

    def v(self, mdp: MDP, value: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Return a softmax policy after one-step state-value lookahead."""
        return self.q(state_action_value(mdp, value, gamma))


@chex.dataclass(frozen=True)
class EpsilonGreedy:
    """Epsilon-greedy value-based policy.

    Attributes:
        epsilon: Uniform exploration probability in the closed interval ``[0, 1]``.
    """

    epsilon: float

    def __post_init__(self) -> None:
        """Validate epsilon."""
        epsilon = jnp.asarray(self.epsilon)
        chex.assert_shape(epsilon, (), custom_message="epsilon must be scalar")
        chex.assert_tree_all_finite(epsilon, custom_message="epsilon must be finite")
        chex.assert_trees_all_equal(
            (epsilon >= 0) & (epsilon <= 1),
            jnp.asarray(True),
            custom_message="epsilon must be in [0, 1]",
        )

    def q(self, value: jax.Array) -> jax.Array:
        """Return an epsilon-greedy policy for an ``(A, S)`` action-value array."""
        greedy = _greedy(value)
        return (1 - self.epsilon) * greedy + self.epsilon / value.shape[0]

    def v(self, mdp: MDP, value: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Return an epsilon-greedy policy after one-step state-value lookahead."""
        return self.q(state_action_value(mdp, value, gamma))


def _greedy(value: jax.Array) -> jax.Array:
    chex.assert_rank(value, 2)
    return jax.nn.one_hot(
        jnp.argmax(value, axis=0),
        value.shape[0],
        axis=0,
    )


__all__ = ["Policy", "Greedy", "Soft", "EpsilonGreedy"]
