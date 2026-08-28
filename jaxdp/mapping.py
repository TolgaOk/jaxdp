"""Mappings for finite Markov decision processes."""

import chex
import jax
import jax.numpy as jnp

from jaxdp.mdp.mdp import MDP
from jaxdp.operator import TransOp, _assert_gamma, _assert_policy

_ATOL = 1e-5


@chex.dataclass(frozen=True)
class GreedyMap:
    r"""Namespace for greedy policy mapping.

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
class SoftGreedyMap:
    r"""Namespace for temperature-scaled soft-greedy policy mapping.

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
    r"""Namespace for epsilon-greedy policy mapping.

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
        greedy = GreedyMap().q(q_val)
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


@chex.dataclass(frozen=True)
class Expectation:
    r"""Namespace for expectations over finite distributions.

    The ``s`` and ``sa`` methods evaluate

    .. math::

        \mathbb{E}_{\rho}[v]
        = \sum_{s\in\mathcal{S}}\rho(s)v(s),
        \qquad
        \mathbb{E}_{\xi}[q]
        = \sum_{s\in\mathcal{S}}\sum_{a\in\mathcal{A}}\xi(s,a)q(s,a).

    Methods:
        s: Evaluate a state-value expectation.
        sa: Evaluate a state-action-value expectation.
    """

    def s(self, v_val: jax.Array, dist: jax.Array) -> jax.Array:
        """Return the expectation of state values under a state distribution.

        Args:
            v_val: State values with shape ``(S,)``.
            dist: State distribution with shape ``(S,)``.

        Returns:
            Scalar expectation with shape ``()``.
        """
        chex.assert_rank(v_val, 1)
        chex.assert_equal_shape((v_val, dist))
        self._assert_dist(dist)
        return jnp.sum(dist * v_val)

    def sa(self, q_val: jax.Array, dist: jax.Array) -> jax.Array:
        """Return the expectation of action values under a state-action distribution.

        Args:
            q_val: Action values with shape ``(A, S)``.
            dist: State-action distribution with shape ``(A, S)``.

        Returns:
            Scalar expectation with shape ``()``.
        """
        chex.assert_rank(q_val, 2)
        chex.assert_equal_shape((q_val, dist))
        self._assert_dist(dist)
        return jnp.sum(dist * q_val)

    @staticmethod
    def _assert_dist(dist: jax.Array) -> None:
        chex.assert_tree_all_finite(dist, custom_message="distribution must be finite")
        chex.assert_trees_all_equal(
            jnp.all(dist >= 0),
            jnp.asarray(True),
            custom_message="distribution probabilities must be nonnegative",
        )
        chex.assert_trees_all_close(
            jnp.sum(dist),
            jnp.ones((), dtype=dist.dtype),
            atol=_ATOL,
            rtol=0.0,
            custom_message="distribution probabilities must sum to one",
        )


@chex.dataclass(frozen=True)
class Occupancy:
    r"""Namespace for finite-step state and state-action distribution mappings.

    Starting from the MDP initial distribution, the configured number of steps applies

    .. math::

        \rho_{k+1}(s')
        = \sum_{s\in\mathcal{S}}P^\pi(s'\mid s)\rho_k(s),
        \qquad
        \xi_k(s,a)=\rho_k(s)\pi(a\mid s).

    Attributes:
        steps: Nonnegative number of transitions from the initial distribution.

    Methods:
        q: Return the finite-step state-action distribution.
        v: Return the finite-step state distribution.
    """

    steps: int

    def q(self, mdp: MDP, policy: jax.Array) -> jax.Array:
        """Return the state-action distribution after the configured number of steps.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.

        Returns:
            State-action distribution with shape ``(A, S)``.
        """
        return policy * self.v(mdp, policy)

    def v(self, mdp: MDP, policy: jax.Array) -> jax.Array:
        """Return the state distribution after the configured number of steps.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.

        Returns:
            State distribution with shape ``(S,)``.
        """
        chex.assert_type(self.steps, int, custom_message="steps must be an integer")
        chex.assert_scalar_non_negative(self.steps, custom_message="steps must be nonnegative")
        p_s = _policy_transition(mdp, policy)
        return jax.lax.fori_loop(
            0,
            self.steps,
            lambda _, dist: p_s @ dist,
            mdp.initial,
        )


@chex.dataclass(frozen=True)
class Stationary:
    r"""Namespace for invariant distribution mappings.

    ``v`` returns the normalized minimum-norm solution of

    .. math::

        \rho_\infty(s')
        = \sum_{s\in\mathcal{S}}P^\pi(s'\mid s)\rho_\infty(s),
        \qquad
        \sum_{s\in\mathcal{S}}\rho_\infty(s)=1.

    The ``q`` method combines the invariant state distribution with the policy:

    .. math::

        \xi_\infty(s,a)=\rho_\infty(s)\pi(a\mid s).

    Methods:
        q: Return the invariant state-action distribution.
        v: Return the invariant state distribution.
    """

    def q(self, mdp: MDP, policy: jax.Array) -> jax.Array:
        """Return the invariant state-action distribution for the induced chain.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.

        Returns:
            Invariant state-action distribution with shape ``(A, S)``.
        """
        return policy * self.v(mdp, policy)

    def v(self, mdp: MDP, policy: jax.Array) -> jax.Array:
        """Return the invariant state distribution for the induced chain.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.

        Returns:
            Normalized minimum-norm invariant state distribution with shape ``(S,)``.
        """
        p_s = _policy_transition(mdp, policy)
        state_size = mdp.state_size
        system = jnp.concatenate(
            (
                p_s - jnp.eye(state_size, dtype=p_s.dtype),
                jnp.ones((1, state_size), dtype=p_s.dtype),
            ),
            axis=0,
        )
        target = jnp.concatenate(
            (
                jnp.zeros(state_size, dtype=p_s.dtype),
                jnp.ones(1, dtype=p_s.dtype),
            )
        )
        dist = jnp.linalg.lstsq(system, target, rcond=None)[0]
        dist = jnp.maximum(dist, 0)
        return dist / jnp.sum(dist)


def eigenvalues(mdp: MDP, policy: jax.Array) -> jax.Array:
    """Return the eigenvalues of the policy-induced transition matrix.

    Args:
        mdp: Finite Markov decision process.
        policy: Action probabilities with shape ``(A, S)``.

    Returns:
        Complex eigenvalues with shape ``(S,)``.
    """
    return jnp.linalg.eigvals(_policy_transition(mdp, policy))


def _policy_transition(mdp: MDP, policy: jax.Array) -> jax.Array:
    _assert_policy(mdp, policy)
    return jnp.einsum("as,axs->xs", policy, mdp.transition)


__all__ = [
    "GreedyMap",
    "SoftGreedyMap",
    "EpsilonGreedy",
    "Expectation",
    "Occupancy",
    "Stationary",
    "eigenvalues",
]
