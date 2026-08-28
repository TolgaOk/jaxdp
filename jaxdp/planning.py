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


__all__ = ["policy_eval", "ValueIteration", "QValueIteration", "PolicyIteration"]
