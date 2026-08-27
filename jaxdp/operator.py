"""Value-space maps and Bellman operators for finite MDPs."""

import chex
import jax
import jax.numpy as jnp

from jaxdp.mdp import MDP

_ATOL = 1e-5


@chex.dataclass(frozen=True)
class ValueMap:
    r"""Namespace for maps between state- and action-value spaces.

    ``to_q`` applies the one-step state-to-action backup

    .. math::

        (\mathcal{B}_\gamma v)(s,a)
        = r(s,a)
        + \gamma \sum_{s'\in\mathcal{S}}\bar P(s'\mid s,a)v(s'),

    The barred transition kernel masks values after terminal successors. ``to_v`` applies the
    greedy action reduction

    .. math::

        (\mathcal{M}q)(s) = \max_{a\in\mathcal{A}}q(s,a).

    Methods:
        to_q: Map state values to one-step action values.
        to_v: Map action values to greedy state values.
    """

    def to_q(self, mdp: MDP, v_val: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Map state values to one-step action values.

        Args:
            mdp: Finite Markov decision process.
            v_val: State values with shape ``(S,)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Action values with shape ``(A, S)``.
        """
        gamma_array = _assert_gamma(gamma)
        chex.assert_shape(v_val, (mdp.state_size,))
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        expected_next = jnp.einsum(
            "axs,x,x->as",
            mdp.transition,
            v_val,
            1 - mdp.terminal,
        )
        return reward + gamma_array * expected_next

    def to_v(self, q_val: jax.Array) -> jax.Array:
        """Map action values to greedy state values.

        Args:
            q_val: Action values with shape ``(A, S)``.

        Returns:
            Greedy state values with shape ``(S,)``.
        """
        chex.assert_rank(q_val, 2)
        return jnp.max(q_val, axis=0)


@chex.dataclass(frozen=True)
class Resolvent:
    r"""Namespace for discounted transition resolvents.

    For the policy-induced terminal-masked transition operators on state and state-action vectors,
    ``s`` and ``sa`` apply

    .. math::

        \mathcal{R}_{S,\gamma}(\mathcal{P}_{S})
        = \left(I-\gamma\mathcal{P}_{S}\right)^{-1},
        \qquad
        \mathcal{R}^{\pi}_{SA,\gamma}
        = \left(I-\gamma\bar{\mathcal{P}}\Pi_\pi\right)^{-1}.

    Their inputs are arbitrary vectors on the corresponding finite spaces. The state-action
    resolvent uses the identity

    .. math::

        \mathcal{R}^{\pi}_{SA,\gamma}x
        = x + \gamma\bar{\mathcal{P}}
          \mathcal{R}_{S,\gamma}(\bar{\mathcal{P}}^{\pi}_{S})\Pi_\pi x,

    which requires only a state-sized linear solve.

    Methods:
        s: Apply a state-space resolvent.
        sa: Apply a policy-induced state-action-space resolvent.
    """

    def s(self, p_s: jax.Array, vec: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Apply the state-space resolvent to a vector.

        Args:
            p_s: State transition operator with shape ``(S, S)`` and storage order
                ``p_s[s_next, s]``. Include terminal masking in this operator when required.
            vec: State-space vector with shape ``(S,)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Resolved state-space vector with shape ``(S,)``.
        """
        gamma_array = _assert_gamma(gamma)
        chex.assert_rank(p_s, 2)
        state_size = p_s.shape[-1]
        chex.assert_shape(p_s, (state_size, state_size))
        chex.assert_shape(vec, (state_size,))
        return jnp.linalg.solve(
            jnp.eye(state_size, dtype=p_s.dtype) - gamma_array * p_s.T,
            vec,
        )

    def sa(
        self,
        mdp: MDP,
        policy: jax.Array,
        vec: jax.Array,
        gamma: float | jax.Array,
    ) -> jax.Array:
        """Apply the policy-induced state-action resolvent to a vector.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.
            vec: State-action-space vector with shape ``(A, S)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Resolved state-action-space vector with shape ``(A, S)``.
        """
        gamma_array = jnp.asarray(gamma)
        _assert_policy(mdp, policy)
        chex.assert_shape(vec, (mdp.action_size, mdp.state_size))
        p_s = jnp.einsum("as,axs,x->xs", policy, mdp.transition, 1 - mdp.terminal)
        s_vec = jnp.einsum("as,as->s", policy, vec)
        resolved_s_vec = self.s(p_s, s_vec, gamma_array)
        return vec + gamma_array * jnp.einsum(
            "axs,x,x->as",
            mdp.transition,
            resolved_s_vec,
            1 - mdp.terminal,
        )


@chex.dataclass(frozen=True)
class BellmanOp:
    r"""Namespace for discounted Bellman policy operators.

    For a policy, ``v`` and ``q`` apply

    .. math::

        \mathcal{T}^{\pi}_{V} = \Pi_\pi \mathcal{B}_\gamma,
        \qquad
        \mathcal{T}^{\pi}_{Q} = \mathcal{B}_\gamma \Pi_\pi,

    Here the backup maps state values to action values and the policy map averages actions.

    Methods:
        q: Apply the action-value Bellman policy operator.
        v: Apply the state-value Bellman policy operator.
    """

    def q(
        self,
        mdp: MDP,
        policy: jax.Array,
        q_val: jax.Array,
        gamma: float | jax.Array,
    ) -> jax.Array:
        """Apply the Bellman policy operator to action values.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.
            q_val: Action values with shape ``(A, S)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Updated action values with shape ``(A, S)``.
        """
        gamma_array = _assert_gamma(gamma)
        _assert_policy(mdp, policy)
        chex.assert_shape(q_val, (mdp.action_size, mdp.state_size))
        next_v_val = jnp.einsum("as,as->s", policy, q_val)
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        expected_next = jnp.einsum(
            "axs,x,x->as",
            mdp.transition,
            next_v_val,
            1 - mdp.terminal,
        )
        return reward + gamma_array * expected_next

    def v(
        self,
        mdp: MDP,
        policy: jax.Array,
        v_val: jax.Array,
        gamma: float | jax.Array,
    ) -> jax.Array:
        """Apply the Bellman policy operator to state values.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.
            v_val: State values with shape ``(S,)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Updated state values with shape ``(S,)``.
        """
        gamma_array = _assert_gamma(gamma)
        _assert_policy(mdp, policy)
        chex.assert_shape(v_val, (mdp.state_size,))
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        expected_next = jnp.einsum(
            "axs,x,x->as",
            mdp.transition,
            v_val,
            1 - mdp.terminal,
        )
        q_val = reward + gamma_array * expected_next
        return jnp.einsum("as,as->s", policy, q_val)


@chex.dataclass(frozen=True)
class BellmanOptOp:
    r"""Namespace for discounted Bellman optimality operators.

    The ``v`` and ``q`` methods apply

    .. math::

        \mathcal{T}^{*}_{V} = \mathcal{M}\mathcal{B}_\gamma,
        \qquad
        \mathcal{T}^{*}_{Q} = \mathcal{B}_\gamma\mathcal{M},

    Here the backup maps state values to action values and the greedy map takes their maximum.

    Methods:
        q: Apply the action-value Bellman optimality operator.
        v: Apply the state-value Bellman optimality operator.
    """

    def q(self, mdp: MDP, q_val: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Apply the Bellman optimality operator to action values.

        Args:
            mdp: Finite Markov decision process.
            q_val: Action values with shape ``(A, S)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Updated action values with shape ``(A, S)``.
        """
        value_map = ValueMap()
        return value_map.to_q(mdp, value_map.to_v(q_val), gamma)

    def v(self, mdp: MDP, v_val: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Apply the Bellman optimality operator to state values.

        Args:
            mdp: Finite Markov decision process.
            v_val: State values with shape ``(S,)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Updated state values with shape ``(S,)``.
        """
        value_map = ValueMap()
        return value_map.to_v(value_map.to_q(mdp, v_val, gamma))


@chex.dataclass(frozen=True)
class SoftBellmanOptOp:
    r"""Namespace for entropy-regularized Bellman optimality operators.

    For a positive temperature, the action reduction and resulting operators are

    .. math::

        \mathcal{L}_{\tau}q(s)
        = \tau\log\sum_{a\in\mathcal{A}}\exp\left(\frac{q(s,a)}{\tau}\right),
        \qquad
        \mathcal{T}^{\mathrm{soft}}_{V,\tau}
        = \mathcal{L}_{\tau}\mathcal{B}_{\gamma},
        \qquad
        \mathcal{T}^{\mathrm{soft}}_{Q,\tau}
        = \mathcal{B}_{\gamma}\mathcal{L}_{\tau}.

    Attributes:
        temperature: Positive entropy temperature.

    Methods:
        q: Apply the action-value soft Bellman optimality operator.
        v: Apply the state-value soft Bellman optimality operator.
    """

    temperature: float

    def q(self, mdp: MDP, q_val: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Apply the soft Bellman optimality operator to action values.

        Args:
            mdp: Finite Markov decision process.
            q_val: Action values with shape ``(A, S)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Updated action values with shape ``(A, S)``.
        """
        return ValueMap().to_q(mdp, self._reduce(q_val), gamma)

    def v(self, mdp: MDP, v_val: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Apply the soft Bellman optimality operator to state values.

        Args:
            mdp: Finite Markov decision process.
            v_val: State values with shape ``(S,)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Updated state values with shape ``(S,)``.
        """
        return self._reduce(ValueMap().to_q(mdp, v_val, gamma))

    def _reduce(self, q_val: jax.Array) -> jax.Array:
        chex.assert_rank(q_val, 2)
        temperature = _assert_temperature(self.temperature)
        return temperature * jax.nn.logsumexp(q_val / temperature, axis=0)


@chex.dataclass(frozen=True)
class MellowmaxBellmanOptOp:
    r"""Namespace for Mellowmax Bellman optimality operators.

    For a positive temperature, the action reduction and resulting operators are

    .. math::

        \operatorname{mm}_{\tau}q(s)
        = \tau\log\left(
          \frac{1}{|\mathcal{A}|}\sum_{a\in\mathcal{A}}
          \exp\left(\frac{q(s,a)}{\tau}\right)\right),
        \qquad
        \mathcal{T}^{\mathrm{mm}}_{V,\tau}
        = \operatorname{mm}_{\tau}\mathcal{B}_{\gamma},
        \qquad
        \mathcal{T}^{\mathrm{mm}}_{Q,\tau}
        = \mathcal{B}_{\gamma}\operatorname{mm}_{\tau}.

    Attributes:
        temperature: Positive Mellowmax temperature.

    Methods:
        q: Apply the action-value Mellowmax Bellman optimality operator.
        v: Apply the state-value Mellowmax Bellman optimality operator.
    """

    temperature: float

    def q(self, mdp: MDP, q_val: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Apply the Mellowmax Bellman optimality operator to action values.

        Args:
            mdp: Finite Markov decision process.
            q_val: Action values with shape ``(A, S)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Updated action values with shape ``(A, S)``.
        """
        return ValueMap().to_q(mdp, self._reduce(q_val), gamma)

    def v(self, mdp: MDP, v_val: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Apply the Mellowmax Bellman optimality operator to state values.

        Args:
            mdp: Finite Markov decision process.
            v_val: State values with shape ``(S,)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Updated state values with shape ``(S,)``.
        """
        return self._reduce(ValueMap().to_q(mdp, v_val, gamma))

    def _reduce(self, q_val: jax.Array) -> jax.Array:
        chex.assert_rank(q_val, 2)
        temperature = _assert_temperature(self.temperature)
        action_size = jnp.asarray(q_val.shape[0], dtype=temperature.dtype)
        return temperature * (jax.nn.logsumexp(q_val / temperature, axis=0) - jnp.log(action_size))


@chex.dataclass(frozen=True)
class BoltzmannBellmanOp:
    r"""Namespace for Boltzmann Bellman operators.

    For a positive temperature, the action reduction and resulting operators are

    .. math::

        \operatorname{boltz}_{\tau}q(s)
        = \sum_{a\in\mathcal{A}}
          \frac{\exp(q(s,a)/\tau)}{\sum_b\exp(q(s,b)/\tau)}q(s,a),
        \qquad
        \mathcal{T}^{\mathrm{boltz}}_{V,\tau}
        = \operatorname{boltz}_{\tau}\mathcal{B}_{\gamma},
        \qquad
        \mathcal{T}^{\mathrm{boltz}}_{Q,\tau}
        = \mathcal{B}_{\gamma}\operatorname{boltz}_{\tau}.

    Unlike the regularized optimality reductions, fixed-temperature Boltzmann expectation is not
    generally a sup-norm non-expansion.

    Attributes:
        temperature: Positive Boltzmann temperature.

    Methods:
        q: Apply the action-value Boltzmann Bellman operator.
        v: Apply the state-value Boltzmann Bellman operator.
    """

    temperature: float

    def q(self, mdp: MDP, q_val: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Apply the Boltzmann Bellman operator to action values.

        Args:
            mdp: Finite Markov decision process.
            q_val: Action values with shape ``(A, S)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Updated action values with shape ``(A, S)``.
        """
        return ValueMap().to_q(mdp, self._reduce(q_val), gamma)

    def v(self, mdp: MDP, v_val: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Apply the Boltzmann Bellman operator to state values.

        Args:
            mdp: Finite Markov decision process.
            v_val: State values with shape ``(S,)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Updated state values with shape ``(S,)``.
        """
        return self._reduce(ValueMap().to_q(mdp, v_val, gamma))

    def _reduce(self, q_val: jax.Array) -> jax.Array:
        chex.assert_rank(q_val, 2)
        temperature = _assert_temperature(self.temperature)
        policy = jax.nn.softmax(q_val / temperature, axis=0)
        return jnp.sum(policy * q_val, axis=0)


def _assert_gamma(gamma: float | jax.Array) -> jax.Array:
    gamma_array = jnp.asarray(gamma)
    chex.assert_shape(gamma_array, (), custom_message="gamma must be scalar")
    chex.assert_tree_all_finite(gamma_array, custom_message="gamma must be finite")
    chex.assert_trees_all_equal(
        (gamma_array >= 0) & (gamma_array < 1),
        jnp.asarray(True),
        custom_message="gamma must be in [0, 1)",
    )
    return gamma_array


def _assert_temperature(temperature: float | jax.Array) -> jax.Array:
    temperature_array = jnp.asarray(temperature)
    chex.assert_shape(temperature_array, (), custom_message="temperature must be scalar")
    chex.assert_tree_all_finite(
        temperature_array,
        custom_message="temperature must be finite",
    )
    chex.assert_trees_all_equal(
        temperature_array > 0,
        jnp.asarray(True),
        custom_message="temperature must be positive",
    )
    return temperature_array


def _assert_policy(mdp: MDP, policy: jax.Array) -> None:
    chex.assert_shape(
        policy,
        (mdp.action_size, mdp.state_size),
        custom_message="policy shape must be (A, S)",
    )
    chex.assert_tree_all_finite(policy, custom_message="policy must be finite")
    chex.assert_trees_all_equal(
        jnp.all(policy >= 0),
        jnp.asarray(True),
        custom_message="policy probabilities must be nonnegative",
    )
    policy_mass = policy.sum(axis=0)
    chex.assert_trees_all_close(
        policy_mass,
        jnp.ones_like(policy_mass),
        atol=_ATOL,
        rtol=0.0,
        custom_message="policy probabilities must sum to one for each state",
    )


__all__ = [
    "ValueMap",
    "Resolvent",
    "BellmanOp",
    "BellmanOptOp",
    "SoftBellmanOptOp",
    "MellowmaxBellmanOptOp",
    "BoltzmannBellmanOp",
]
