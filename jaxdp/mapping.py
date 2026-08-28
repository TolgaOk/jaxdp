"""Mappings for finite Markov decision processes."""

import chex
import jax
import jax.numpy as jnp

from jaxdp.mdp.mdp import MDP
from jaxdp.operator import _assert_gamma, _assert_policy, trans_op

_ATOL = 1e-5


class Reward:
    r"""Namespace for expected immediate reward mappings.

    The ``sa`` and ``s`` methods apply

    .. math::

        r(s,a)=\sum_{s'\in\mathcal{S}}P(s'\mid s,a)R(s,a,s'),
        \qquad
        r^{\pi}(s)=\sum_{a\in\mathcal{A}}\pi(a\mid s)r(s,a).

    Methods:
        s: Return policy-expected state rewards.
        sa: Return expected state-action rewards.
    """

    @staticmethod
    def s(mdp: MDP, policy: jax.Array) -> jax.Array:
        """Return expected immediate rewards under a policy.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.

        Returns:
            Expected state rewards with shape ``(S,)``.
        """
        _assert_policy(mdp, policy)
        return jnp.sum(policy * reward.sa(mdp), axis=0)

    @staticmethod
    def sa(mdp: MDP) -> jax.Array:
        """Return expected immediate rewards for each state-action pair.

        Args:
            mdp: Finite Markov decision process.

        Returns:
            Expected state-action rewards with shape ``(A, S)``.
        """
        return jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)


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

    @staticmethod
    def q(q_val: jax.Array) -> jax.Array:
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

    @staticmethod
    def v(mdp: MDP, v_val: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Select a greedy policy from state values.

        Args:
            mdp: Finite Markov decision process.
            v_val: State values with shape ``(S,)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Action probabilities with shape ``(A, S)``.
        """
        reward_sa = reward.sa(mdp)
        q_val = reward_sa + _assert_gamma(gamma) * trans_op.sa(mdp, v_val)
        return greedy_map.q(q_val)


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
        reward_sa = reward.sa(mdp)
        q_val = reward_sa + _assert_gamma(gamma) * trans_op.sa(mdp, v_val)
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
        greedy = greedy_map.q(q_val)
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
        reward_sa = reward.sa(mdp)
        q_val = reward_sa + _assert_gamma(gamma) * trans_op.sa(mdp, v_val)
        return self.q(q_val)


class ProjSimplex:
    r"""Namespace for Euclidean projection onto the action simplex.

    The ``q`` method applies

    .. math::

        \operatorname{proj}_{\Delta_A}(q)(\cdot\mid s)
        = \mathop{\arg\min}_{p\in\Delta_A}
          \frac{1}{2}\lVert p-q(\cdot,s)\rVert_2^2.

    Methods:
        q: Project each state column onto the action simplex.
    """

    @staticmethod
    def q(q_val: jax.Array) -> jax.Array:
        """Project action values onto the action simplex for every state.

        Args:
            q_val: Action values with shape ``(A, S)``.

        Returns:
            Action probabilities with shape ``(A, S)``.
        """
        chex.assert_rank(q_val, 2)
        chex.assert_axis_dimension_gt(q_val, 0, 0)
        sorted_val = jnp.flip(jnp.sort(q_val, axis=0), axis=0)
        cumulative = jnp.cumsum(sorted_val, axis=0) - 1
        rank = jnp.arange(1, q_val.shape[0] + 1, dtype=q_val.dtype)[:, None]
        support_size = jnp.sum(sorted_val - cumulative / rank > 0, axis=0)
        threshold = jnp.take_along_axis(
            cumulative,
            support_size[None, :] - 1,
            axis=0,
        )[0] / support_size
        return jnp.maximum(q_val - threshold, 0)


@chex.dataclass(frozen=True)
class MellowMax:
    r"""Namespace for the Mellowmax action-value reduction.

    For a positive temperature, the ``q`` method applies

    .. math::

        \operatorname{mm}_{\tau}q(s)
        = \tau\log\left(
          \frac{1}{|\mathcal{A}|}\sum_{a\in\mathcal{A}}
          \exp\left(\frac{q(s,a)}{\tau}\right)\right).

    Attributes:
        temperature: Positive Mellowmax temperature.

    Methods:
        q: Reduce action values to state values.
    """

    temperature: float

    def q(self, q_val: jax.Array) -> jax.Array:
        """Reduce action values with normalized log-mean-exp.

        Args:
            q_val: Action values with shape ``(A, S)``.

        Returns:
            State values with shape ``(S,)``.
        """
        chex.assert_rank(q_val, 2)
        chex.assert_axis_dimension_gt(q_val, 0, 0)
        temperature = jnp.asarray(self.temperature)
        chex.assert_shape(temperature, (), custom_message="temperature must be scalar")
        chex.assert_tree_all_finite(temperature, custom_message="temperature must be finite")
        chex.assert_trees_all_equal(
            temperature > 0,
            jnp.asarray(True),
            custom_message="temperature must be positive",
        )
        action_size = jnp.asarray(q_val.shape[0], dtype=temperature.dtype)
        return temperature * (
            jax.nn.logsumexp(q_val / temperature, axis=0) - jnp.log(action_size)
        )


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

    @staticmethod
    def s(v_val: jax.Array, dist: jax.Array) -> jax.Array:
        """Return the expectation of state values under a state distribution.

        Args:
            v_val: State values with shape ``(S,)``.
            dist: State distribution with shape ``(S,)``.

        Returns:
            Scalar expectation with shape ``()``.
        """
        chex.assert_rank(v_val, 1)
        chex.assert_equal_shape((v_val, dist))
        Expectation._assert_dist(dist)
        return jnp.sum(dist * v_val)

    @staticmethod
    def sa(q_val: jax.Array, dist: jax.Array) -> jax.Array:
        """Return the expectation of action values under a state-action distribution.

        Args:
            q_val: Action values with shape ``(A, S)``.
            dist: State-action distribution with shape ``(A, S)``.

        Returns:
            Scalar expectation with shape ``()``.
        """
        chex.assert_rank(q_val, 2)
        chex.assert_equal_shape((q_val, dist))
        Expectation._assert_dist(dist)
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
    r"""Namespace for normalized state and state-action occupancy mappings.

    Starting from the MDP initial distribution, each configured iteration applies

    .. math::

        d_{k+1}^{\gamma}
        = (1-\gamma)\mu
          + \gamma(\mathcal{P}^{\pi})^*d_k^{\gamma},
        \qquad
        d_0^{\gamma}=\mu,
        \qquad
        \xi_k^{\gamma}(s,a)=d_k^{\gamma}(s)\pi(a\mid s).

    At ``gamma=1``, iteration ``k`` is the ordinary time-``k`` marginal. For ``gamma<1``, the
    iteration converges to the normalized discounted occupancy

    .. math::

        d_{\gamma}^{\pi}
        = (1-\gamma)\sum_{t=0}^{\infty}
          \gamma^t\bigl((\mathcal{P}^{\pi})^*\bigr)^t\mu.

    Attributes:
        step: Nonnegative number of forward iterations.

    Methods:
        q: Return the normalized state-action occupancy.
        v: Return the normalized state occupancy.
    """

    step: int = 1

    def q(
        self,
        mdp: MDP,
        policy: jax.Array,
        gamma: float | jax.Array = 1.0,
    ) -> jax.Array:
        """Return the state-action occupancy after the configured iterations.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.
            gamma: Scalar discount in the closed interval ``[0, 1]``.

        Returns:
            Normalized state-action occupancy with shape ``(A, S)``.
        """
        return policy * self.v(mdp, policy, gamma)

    def v(
        self,
        mdp: MDP,
        policy: jax.Array,
        gamma: float | jax.Array = 1.0,
    ) -> jax.Array:
        """Return the state occupancy after the configured iterations.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.
            gamma: Scalar discount in the closed interval ``[0, 1]``.

        Returns:
            Normalized state occupancy with shape ``(S,)``.
        """
        chex.assert_type(self.step, int, custom_message="step must be an integer")
        chex.assert_scalar_non_negative(self.step, custom_message="step must be nonnegative")
        gamma_array = jnp.asarray(gamma)
        chex.assert_shape(gamma_array, (), custom_message="gamma must be scalar")
        chex.assert_tree_all_finite(gamma_array, custom_message="gamma must be finite")
        chex.assert_trees_all_equal(
            (gamma_array >= 0) & (gamma_array <= 1),
            jnp.asarray(True),
            custom_message="gamma must be in [0, 1]",
        )
        p_s = _policy_transition(mdp, policy)
        return jax.lax.fori_loop(
            0,
            self.step,
            lambda _, dist: (1 - gamma_array) * mdp.initial + gamma_array * (p_s @ dist),
            mdp.initial,
        )


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

    @staticmethod
    def q(mdp: MDP, policy: jax.Array) -> jax.Array:
        """Return the invariant state-action distribution for the induced chain.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.

        Returns:
            Invariant state-action distribution with shape ``(A, S)``.
        """
        return policy * stationary.v(mdp, policy)

    @staticmethod
    def v(mdp: MDP, policy: jax.Array) -> jax.Array:
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


reward = Reward
greedy_map = GreedyMap
proj_simplex = ProjSimplex
expectation = Expectation
stationary = Stationary


__all__ = [
    "greedy_map",
    "SoftGreedyMap",
    "EpsilonGreedy",
    "proj_simplex",
    "MellowMax",
    "reward",
    "expectation",
    "Occupancy",
    "stationary",
    "eigenvalues",
]
