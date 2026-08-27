"""Policy components for finite Markov decision processes."""

import chex
import jax
import jax.numpy as jnp

from jaxdp.mdp.mdp import MDP
from jaxdp.operator import TransOp, _assert_gamma


@chex.dataclass(frozen=True)
class Greedy:
    r"""Namespace for greedy policy selection.

    The ``q`` and ``v`` methods apply

    .. math::

        \mathcal{G}:Q\to\Pi,
        \qquad
        \mathcal{G}\mathcal{B}_\gamma:V\to\Pi.

    Methods:
        q: Select a greedy policy from action values.
        v: Select a greedy policy from state values.
    """

    def q(self, q_val: jax.Array) -> jax.Array:
        """Select a greedy policy from action values.

        Args:
            q_val: Action values with shape ``(A, S)``.

        Returns:
            Action probabilities with shape ``(A, S)``.
        """
        chex.assert_rank(q_val, 2)
        return jax.nn.one_hot(
            jnp.argmax(q_val, axis=0),
            q_val.shape[0],
            axis=0,
        )

    def v(self, mdp: MDP, v_val: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Select a greedy policy from state values.

        Args:
            mdp: Finite Markov decision process.
            v_val: State values with shape ``(S,)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Action probabilities with shape ``(A, S)``.
        """
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        q_val = reward + _assert_gamma(gamma) * TransOp().sa(mdp, v_val)
        return self.q(q_val)


@chex.dataclass(frozen=True)
class Soft:
    r"""Namespace for temperature-scaled softmax policy selection.

    For a positive temperature, the ``q`` and ``v`` methods apply

    .. math::

        \mathcal{S}_\eta:Q\to\Pi,
        \qquad
        \mathcal{S}_\eta\mathcal{B}_\gamma:V\to\Pi.

    Attributes:
        temperature: Positive softmax temperature.

    Methods:
        q: Select a softmax policy from action values.
        v: Select a softmax policy from state values.
    """

    temperature: float

    def q(self, q_val: jax.Array) -> jax.Array:
        """Select a softmax policy from action values.

        Args:
            q_val: Action values with shape ``(A, S)``.

        Returns:
            Action probabilities with shape ``(A, S)``.
        """
        chex.assert_rank(q_val, 2)
        temperature = jnp.asarray(self.temperature)
        chex.assert_shape(temperature, (), custom_message="temperature must be scalar")
        chex.assert_tree_all_finite(temperature, custom_message="temperature must be finite")
        chex.assert_trees_all_equal(
            temperature > 0,
            jnp.asarray(True),
            custom_message="temperature must be positive",
        )
        return jax.nn.softmax(q_val / temperature, axis=0)

    def v(self, mdp: MDP, v_val: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Select a softmax policy from state values.

        Args:
            mdp: Finite Markov decision process.
            v_val: State values with shape ``(S,)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Action probabilities with shape ``(A, S)``.
        """
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        q_val = reward + _assert_gamma(gamma) * TransOp().sa(mdp, v_val)
        return self.q(q_val)


@chex.dataclass(frozen=True)
class EpsilonGreedy:
    r"""Namespace for epsilon-greedy policy selection.

    For an exploration probability in the closed unit interval, the ``q`` and ``v`` methods apply

    .. math::

        \mathcal{G}_\epsilon:Q\to\Pi,
        \qquad
        \mathcal{G}_\epsilon\mathcal{B}_\gamma:V\to\Pi.

    Attributes:
        epsilon: Uniform exploration probability in the closed interval ``[0, 1]``.

    Methods:
        q: Select an epsilon-greedy policy from action values.
        v: Select an epsilon-greedy policy from state values.
    """

    epsilon: float

    def q(self, q_val: jax.Array) -> jax.Array:
        """Select an epsilon-greedy policy from action values.

        Args:
            q_val: Action values with shape ``(A, S)``.

        Returns:
            Action probabilities with shape ``(A, S)``.
        """
        epsilon = jnp.asarray(self.epsilon)
        chex.assert_shape(epsilon, (), custom_message="epsilon must be scalar")
        chex.assert_tree_all_finite(epsilon, custom_message="epsilon must be finite")
        chex.assert_trees_all_equal(
            (epsilon >= 0) & (epsilon <= 1),
            jnp.asarray(True),
            custom_message="epsilon must be in [0, 1]",
        )
        greedy = Greedy().q(q_val)
        return (1 - epsilon) * greedy + epsilon / q_val.shape[0]

    def v(self, mdp: MDP, v_val: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Select an epsilon-greedy policy from state values.

        Args:
            mdp: Finite Markov decision process.
            v_val: State values with shape ``(S,)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Action probabilities with shape ``(A, S)``.
        """
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        q_val = reward + _assert_gamma(gamma) * TransOp().sa(mdp, v_val)
        return self.q(q_val)


__all__ = ["Greedy", "Soft", "EpsilonGreedy"]
