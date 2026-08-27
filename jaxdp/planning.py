"""Planning solvers for finite Markov decision processes."""

import chex
import jax
import jax.numpy as jnp

from jaxdp.mdp import MDP, make_mrp
from jaxdp.operator import Resolvent, _assert_gamma, _assert_policy
from jaxdp.policy import Greedy


@chex.dataclass(frozen=True)
class PolicyEvaluation:
    r"""Namespace for exact discounted policy evaluation.

    For a policy, ``v`` and ``q`` apply

    .. math::

        v^\pi
        = \mathcal{R}_{S,\gamma}(\bar{\mathcal{P}}^{\pi}_{S})r^\pi,
        \qquad
        q^\pi = \mathcal{R}^{\pi}_{SA,\gamma}r,

    Here the right-hand sides are the expected state and state-action rewards.

    Methods:
        q: Return exact action values for a policy.
        v: Return exact state values for a policy.
    """

    def q(self, mdp: MDP, policy: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Return exact action values for a policy.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Policy action values with shape ``(A, S)``.
        """
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        return Resolvent().sa(mdp, policy, reward, gamma)

    def v(self, mdp: MDP, policy: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Return exact state values for a policy.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Policy state values with shape ``(S,)``.
        """
        mrp = make_mrp(mdp, policy)
        p_s = mrp.transition * (1 - mrp.terminal)[:, None]
        return Resolvent().s(p_s, mrp.reward, gamma)


@chex.dataclass(frozen=True)
class ValueIteration:
    r"""Namespace for fixed-step value iteration.

    The ``q`` and ``v`` methods apply the corresponding Bellman optimality operator ``step``
    times:

    .. math::

        q_n=(\mathcal{T}^{*}_{Q})^nq_0,
        \qquad
        v_n=(\mathcal{T}^{*}_{V})^nv_0.

    Attributes:
        step: Number of Bellman optimality updates.

    Methods:
        q: Iterate action values.
        v: Iterate state values.
    """

    step: int = 1

    def q(
        self,
        mdp: MDP,
        q_val: jax.Array,
        gamma: float | jax.Array,
    ) -> jax.Array:
        """Apply fixed-step Bellman optimality updates to action values.

        Args:
            mdp: Finite Markov decision process.
            q_val: Initial action values with shape ``(A, S)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Action values after ``step`` updates with shape ``(A, S)``.
        """
        chex.assert_type(self.step, int, custom_message="step must be an integer")
        chex.assert_scalar_non_negative(self.step, custom_message="step must be nonnegative")
        gamma_array = _assert_gamma(gamma)
        chex.assert_shape(q_val, (mdp.action_size, mdp.state_size))
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)

        def update(val: jax.Array, _: None) -> tuple[jax.Array, None]:
            next_val = reward + gamma_array * jnp.einsum(
                "axs,x,x->as",
                mdp.transition,
                jnp.max(val, axis=0),
                1 - mdp.terminal,
            )
            return next_val, None

        q_val, _ = jax.lax.scan(update, q_val, xs=None, length=self.step)
        return q_val

    def v(
        self,
        mdp: MDP,
        v_val: jax.Array,
        gamma: float | jax.Array,
    ) -> jax.Array:
        """Apply fixed-step Bellman optimality updates to state values.

        Args:
            mdp: Finite Markov decision process.
            v_val: Initial state values with shape ``(S,)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            State values after ``step`` updates with shape ``(S,)``.
        """
        chex.assert_type(self.step, int, custom_message="step must be an integer")
        chex.assert_scalar_non_negative(self.step, custom_message="step must be nonnegative")
        gamma_array = _assert_gamma(gamma)
        chex.assert_shape(v_val, (mdp.state_size,))
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)

        def update(val: jax.Array, _: None) -> tuple[jax.Array, None]:
            next_val = jnp.max(
                reward
                + gamma_array
                * jnp.einsum(
                    "axs,x,x->as",
                    mdp.transition,
                    val,
                    1 - mdp.terminal,
                ),
                axis=0,
            )
            return next_val, None

        v_val, _ = jax.lax.scan(update, v_val, xs=None, length=self.step)
        return v_val


@chex.dataclass(frozen=True)
class PolicyIteration:
    r"""Namespace for fixed-step exact policy iteration.

    Starting from an initial policy, each update applies

    .. math::

        q^{\pi_k} = \mathcal{R}^{\pi_k}_{SA,\gamma}r,
        \qquad
        \pi_{k+1}=\mathcal{G}(q^{\pi_k}).

    Attributes:
        step: Number of policy-improvement updates.

    Methods:
        q: Return action values for the final policy.
        v: Return state values for the final policy.
        policy: Return the final policy.
    """

    step: int = 1

    def q(
        self,
        mdp: MDP,
        policy: jax.Array,
        gamma: float | jax.Array,
    ) -> jax.Array:
        """Return action values after fixed-step policy improvement.

        Args:
            mdp: Finite Markov decision process.
            policy: Initial action probabilities with shape ``(A, S)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Final-policy action values with shape ``(A, S)``.
        """
        policy = self.policy(mdp, policy, gamma)
        return PolicyEvaluation().q(mdp, policy, gamma)

    def v(
        self,
        mdp: MDP,
        policy: jax.Array,
        gamma: float | jax.Array,
    ) -> jax.Array:
        """Return state values after fixed-step policy improvement.

        Args:
            mdp: Finite Markov decision process.
            policy: Initial action probabilities with shape ``(A, S)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Final-policy state values with shape ``(S,)``.
        """
        policy = self.policy(mdp, policy, gamma)
        return PolicyEvaluation().v(mdp, policy, gamma)

    def policy(
        self,
        mdp: MDP,
        policy: jax.Array,
        gamma: float | jax.Array,
    ) -> jax.Array:
        """Return the policy after fixed-step policy improvement.

        Args:
            mdp: Finite Markov decision process.
            policy: Initial action probabilities with shape ``(A, S)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Improved action probabilities with shape ``(A, S)``.
        """
        chex.assert_type(self.step, int, custom_message="step must be an integer")
        chex.assert_scalar_non_negative(self.step, custom_message="step must be nonnegative")
        gamma_array = _assert_gamma(gamma)
        _assert_policy(mdp, policy)
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        not_terminal = 1 - mdp.terminal
        identity = jnp.eye(mdp.state_size, dtype=mdp.transition.dtype)
        greedy = Greedy()

        def improve(pol: jax.Array, _: None) -> tuple[jax.Array, None]:
            p_s = jnp.einsum(
                "as,axs,x->xs",
                pol,
                mdp.transition,
                not_terminal,
            )
            reward_s = jnp.einsum("as,as->s", pol, reward)
            v_val = jnp.linalg.solve(identity - gamma_array * p_s.T, reward_s)
            q_val = reward + gamma_array * jnp.einsum(
                "axs,x,x->as",
                mdp.transition,
                v_val,
                not_terminal,
            )
            return greedy.q(q_val), None

        policy, _ = jax.lax.scan(improve, policy, xs=None, length=self.step)
        return policy


__all__ = ["PolicyEvaluation", "ValueIteration", "PolicyIteration"]
