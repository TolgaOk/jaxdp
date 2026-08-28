"""Planning solvers for finite Markov decision processes."""

from __future__ import annotations

from dataclasses import replace

import chex
import jax
import jax.numpy as jnp

from jaxdp.mapping import greedy_map, reward
from jaxdp.mdp import MDP, make_mrp
from jaxdp.operator import bellman_opt_op, resolvent


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

    @staticmethod
    def q(mdp: MDP, policy: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Return exact action values for a policy.

        Args:
            mdp: Finite Markov decision process.
            policy: Action probabilities with shape ``(A, S)``.
            gamma: Scalar discount in the interval ``[0, 1)``.

        Returns:
            Policy action values with shape ``(A, S)``.
        """
        reward_sa = reward.sa(mdp)
        return resolvent.sa(mdp, policy, reward_sa, gamma)

    @staticmethod
    def v(mdp: MDP, policy: jax.Array, gamma: float | jax.Array) -> jax.Array:
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
        return resolvent.s(p_s, mrp.reward, gamma)


@chex.dataclass(frozen=True)
class ValueIteration:
    r"""Perform one state-value iteration update at a time.

    Each update applies

    .. math::

        v_{k+1}=\mathcal{T}^{*}_{V}v_k.

    Attributes:
        gamma: Scalar discount in the interval ``[0, 1)``.

    Public dataclasses:
        State: Current state-value iterate.

    Public methods:
        init: Initialize zero state values.
        update: Apply one Bellman optimality update.
    """

    gamma: float

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic state-value iterate.

        Attributes:
            v_val: State values with shape ``(S,)``.
        """

        v_val: jax.Array

    def init(self, mdp: MDP) -> ValueIteration.State:
        """Initialize zero state values for an MDP."""
        return self.State(
            v_val=jnp.zeros((mdp.state_size,), dtype=mdp.reward.dtype),
        )

    def update(self, mdp: MDP, state: ValueIteration.State) -> ValueIteration.State:
        """Apply one state-value Bellman optimality update."""
        v_val = bellman_opt_op.v(mdp, state.v_val, self.gamma)
        return replace(state, v_val=v_val)


@chex.dataclass(frozen=True)
class QValueIteration:
    r"""Perform one action-value iteration update at a time.

    Each update applies

    .. math::

        q_{k+1}=\mathcal{T}^{*}_{Q}q_k.

    Attributes:
        gamma: Scalar discount in the interval ``[0, 1)``.

    Public dataclasses:
        State: Current action-value iterate.

    Public methods:
        init: Initialize zero action values.
        update: Apply one Bellman optimality update.
    """

    gamma: float

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic action-value iterate.

        Attributes:
            q_val: Action values with shape ``(A, S)``.
        """

        q_val: jax.Array

    def init(self, mdp: MDP) -> QValueIteration.State:
        """Initialize zero action values for an MDP."""
        return self.State(
            q_val=jnp.zeros(
                (mdp.action_size, mdp.state_size),
                dtype=mdp.reward.dtype,
            ),
        )

    def update(self, mdp: MDP, state: QValueIteration.State) -> QValueIteration.State:
        """Apply one action-value Bellman optimality update."""
        q_val = bellman_opt_op.q(mdp, state.q_val, self.gamma)
        return replace(state, q_val=q_val)


@chex.dataclass(frozen=True)
class AnchoredValueIteration:
    r"""Perform one anchored state-value iteration update at a time.

    Each update applies the Anchored Value Iteration recurrence

    .. math::

        v_k=\beta_k v_0+(1-\beta_k)\mathcal{T}^{*}_{V}v_{k-1},
        \qquad
        \beta_k=\left(\sum_{i=0}^{k}\gamma^{-2i}\right)^{-1}.

    Attributes:
        gamma: Scalar discount in the interval ``[0, 1)``.

    Public dataclasses:
        State: Current iterate, anchor, and anchor coefficient.

    Public methods:
        init: Initialize the state from an optional state-value anchor.
        update: Apply one anchored Bellman optimality update.
    """

    gamma: float

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic anchored state-value iterate.

        Attributes:
            v_val: Current state values with shape ``(S,)``.
            v_anchor: Initial state values with shape ``(S,)``.
            beta: Scalar coefficient corresponding to the current iterate.
        """

        v_val: jax.Array
        v_anchor: jax.Array
        beta: jax.Array

    def init(
        self,
        mdp: MDP,
        v_val: jax.Array | None = None,
    ) -> AnchoredValueIteration.State:
        """Initialize the iterate and anchor from state values."""
        if v_val is None:
            v_val = jnp.zeros((mdp.state_size,), dtype=mdp.reward.dtype)
        chex.assert_shape(v_val, (mdp.state_size,))
        return self.State(
            v_val=v_val,
            v_anchor=v_val,
            beta=jnp.ones((), dtype=v_val.dtype),
        )

    def update(
        self,
        mdp: MDP,
        state: AnchoredValueIteration.State,
    ) -> AnchoredValueIteration.State:
        """Apply one anchored state-value Bellman optimality update."""
        gamma_sq = jnp.asarray(self.gamma) ** 2
        beta = gamma_sq * state.beta / (1 + gamma_sq * state.beta)
        bellman_v = bellman_opt_op.v(mdp, state.v_val, self.gamma)
        v_val = beta * state.v_anchor + (1 - beta) * bellman_v
        return replace(state, v_val=v_val, beta=beta)


@chex.dataclass(frozen=True)
class AnchoredQValueIteration:
    r"""Perform one anchored action-value iteration update at a time.

    Each update applies the Anchored Value Iteration recurrence

    .. math::

        q_k=\beta_k q_0+(1-\beta_k)\mathcal{T}^{*}_{Q}q_{k-1},
        \qquad
        \beta_k=\left(\sum_{i=0}^{k}\gamma^{-2i}\right)^{-1}.

    Attributes:
        gamma: Scalar discount in the interval ``[0, 1)``.

    Public dataclasses:
        State: Current iterate, anchor, and anchor coefficient.

    Public methods:
        init: Initialize the state from an optional action-value anchor.
        update: Apply one anchored Bellman optimality update.
    """

    gamma: float

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic anchored action-value iterate.

        Attributes:
            q_val: Current action values with shape ``(A, S)``.
            q_anchor: Initial action values with shape ``(A, S)``.
            beta: Scalar coefficient corresponding to the current iterate.
        """

        q_val: jax.Array
        q_anchor: jax.Array
        beta: jax.Array

    def init(
        self,
        mdp: MDP,
        q_val: jax.Array | None = None,
    ) -> AnchoredQValueIteration.State:
        """Initialize the iterate and anchor from action values."""
        if q_val is None:
            q_val = jnp.zeros(
                (mdp.action_size, mdp.state_size),
                dtype=mdp.reward.dtype,
            )
        chex.assert_shape(q_val, (mdp.action_size, mdp.state_size))
        return self.State(
            q_val=q_val,
            q_anchor=q_val,
            beta=jnp.ones((), dtype=q_val.dtype),
        )

    def update(
        self,
        mdp: MDP,
        state: AnchoredQValueIteration.State,
    ) -> AnchoredQValueIteration.State:
        """Apply one anchored action-value Bellman optimality update."""
        gamma_sq = jnp.asarray(self.gamma) ** 2
        beta = gamma_sq * state.beta / (1 + gamma_sq * state.beta)
        bellman_q = bellman_opt_op.q(mdp, state.q_val, self.gamma)
        q_val = beta * state.q_anchor + (1 - beta) * bellman_q
        return replace(state, q_val=q_val, beta=beta)


@chex.dataclass(frozen=True)
class PolicyIteration:
    r"""Perform one exact policy iteration update at a time.

    Every state satisfies ``v_val = v^policy``. Each update applies

    .. math::

        \pi_{k+1}=\mathcal{G}\mathcal{B}_{\gamma}v^{\pi_k},
        \qquad
        v^{\pi_{k+1}}
        =\mathcal{R}_{S,\gamma}
          (\bar{\mathcal{P}}^{\pi_{k+1}}_{S})r^{\pi_{k+1}}.

    Attributes:
        gamma: Scalar discount in the interval ``[0, 1)``.

    Public dataclasses:
        State: Current policy and its exact state values.

    Public methods:
        init: Evaluate an initial policy exactly.
        update: Improve and evaluate the policy once.
    """

    gamma: float

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic exact policy-iteration state.

        Attributes:
            policy: Action probabilities with shape ``(A, S)``.
            v_val: Exact policy state values with shape ``(S,)``.
        """

        policy: jax.Array
        v_val: jax.Array

    def init(self, mdp: MDP, policy: jax.Array) -> PolicyIteration.State:
        """Initialize from an exactly evaluated policy."""
        v_val = policy_eval.v(mdp, policy, self.gamma)
        return self.State(policy=policy, v_val=v_val)

    def update(self, mdp: MDP, state: PolicyIteration.State) -> PolicyIteration.State:
        """Apply one greedy improvement and exact policy evaluation."""
        policy = greedy_map.v(mdp, state.v_val, self.gamma)
        v_val = policy_eval.v(mdp, policy, self.gamma)
        return replace(state, policy=policy, v_val=v_val)


policy_eval = PolicyEvaluation


__all__ = [
    "policy_eval",
    "ValueIteration",
    "QValueIteration",
    "AnchoredValueIteration",
    "AnchoredQValueIteration",
    "PolicyIteration",
]
