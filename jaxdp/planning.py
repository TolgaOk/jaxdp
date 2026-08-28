"""Planning solvers for finite Markov decision processes."""

from __future__ import annotations

from dataclasses import replace

import chex
import jax
import jax.numpy as jnp

from jaxdp.mapping import greedy_map, reward
from jaxdp.mdp import MDP, make_mrp
from jaxdp.operator import adj_trans_op, bellman_opt_op, resolvent, trans_op


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
class SafeAcceleratedValueIteration:
    r"""Perform one safeguarded Accelerated Value Iteration update at a time.

    Each update applies the S-AVI recurrence

    .. math::

        h_s
        = v_s + \beta(v_s-v_{s-1}),
        \qquad
        \tilde{v}_{s+1}
        = h_s-\alpha(h_s-\mathcal{T}^{*}_{V}h_s),

    and accepts the accelerated proposal only when

    .. math::

        \lVert\tilde{v}_{s+1}-\mathcal{T}^{*}_{V}\tilde{v}_{s+1}\rVert_\infty
        \leq \lambda'^{s+1}
        \lVert v_0-\mathcal{T}^{*}_{V}v_0\rVert_\infty.

    Otherwise, the update returns the ordinary value-iteration step
    ``bellman_opt_op.v(mdp, v_val, gamma)``. By default, ``step_size`` and ``momentum`` use the
    paper's A-VI tuning, while ``rate`` uses its experimental choice ``(1 + gamma) / 2``.

    Attributes:
        gamma: Scalar discount in the interval ``[0, 1)``.
        rate: Safeguard rate in ``[gamma, 1)``.
        step_size: Positive A-VI residual step size.
        momentum: Nonnegative A-VI extrapolation coefficient.

    Public dataclasses:
        State: Current and preceding values, residual bound, and acceptance indicator.

    Public methods:
        init: Initialize with one ordinary value-iteration step.
        update: Propose and safeguard one accelerated update.
    """

    gamma: float
    rate: float | None = None
    step_size: float | None = None
    momentum: float | None = None

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic safeguarded A-VI state.

        Attributes:
            v_val: Current state values with shape ``(S,)``.
            prev_v_val: Preceding state values with shape ``(S,)``.
            bound: Scalar residual bound for ``v_val``.
            accepted: Whether the latest accelerated proposal was accepted.
        """

        v_val: jax.Array
        prev_v_val: jax.Array
        bound: jax.Array
        accepted: jax.Array

    def init(
        self,
        mdp: MDP,
        v_val: jax.Array | None = None,
    ) -> SafeAcceleratedValueIteration.State:
        """Initialize from state values and take the paper's initial VI step."""
        if v_val is None:
            v_val = jnp.zeros((mdp.state_size,), dtype=mdp.reward.dtype)
        chex.assert_shape(v_val, (mdp.state_size,))
        gamma = jnp.asarray(self.gamma)
        rate = (1 + gamma) / 2 if self.rate is None else jnp.asarray(self.rate)
        chex.assert_shape(rate, (), custom_message="rate must be scalar")
        chex.assert_tree_all_finite(rate, custom_message="rate must be finite")
        chex.assert_trees_all_equal(
            (rate >= gamma) & (rate < 1),
            jnp.asarray(True),
            custom_message="rate must be in [gamma, 1)",
        )

        next_v_val = bellman_opt_op.v(mdp, v_val, gamma)
        residual = jnp.max(jnp.abs(v_val - next_v_val))
        return self.State(
            v_val=next_v_val,
            prev_v_val=v_val,
            bound=rate * residual,
            accepted=jnp.asarray(False),
        )

    def update(
        self,
        mdp: MDP,
        state: SafeAcceleratedValueIteration.State,
    ) -> SafeAcceleratedValueIteration.State:
        """Propose and safeguard one accelerated value-iteration update."""
        gamma = jnp.asarray(self.gamma)
        rate = (1 + gamma) / 2 if self.rate is None else jnp.asarray(self.rate)
        step_size = 1 / (1 + gamma) if self.step_size is None else jnp.asarray(self.step_size)
        momentum = (
            gamma / (1 + jnp.sqrt(1 - gamma**2))
            if self.momentum is None
            else jnp.asarray(self.momentum)
        )
        chex.assert_shape(state.v_val, (mdp.state_size,))
        chex.assert_shape(state.prev_v_val, (mdp.state_size,))
        chex.assert_shape(state.bound, ())
        chex.assert_shape(state.accepted, ())
        chex.assert_shape(rate, (), custom_message="rate must be scalar")
        chex.assert_shape(step_size, (), custom_message="step_size must be scalar")
        chex.assert_shape(momentum, (), custom_message="momentum must be scalar")
        chex.assert_tree_all_finite(state, custom_message="state arrays must be finite")
        chex.assert_tree_all_finite(
            (rate, step_size, momentum),
            custom_message="planner parameters must be finite",
        )
        chex.assert_trees_all_equal(
            (rate >= gamma) & (rate < 1),
            jnp.asarray(True),
            custom_message="rate must be in [gamma, 1)",
        )
        chex.assert_trees_all_equal(
            step_size > 0,
            jnp.asarray(True),
            custom_message="step_size must be positive",
        )
        chex.assert_trees_all_equal(
            momentum >= 0,
            jnp.asarray(True),
            custom_message="momentum must be nonnegative",
        )
        chex.assert_trees_all_equal(
            state.bound >= 0,
            jnp.asarray(True),
            custom_message="bound must be nonnegative",
        )

        extrapolated = state.v_val + momentum * (state.v_val - state.prev_v_val)
        bellman_extrapolated = bellman_opt_op.v(mdp, extrapolated, gamma)
        proposal = extrapolated - step_size * (extrapolated - bellman_extrapolated)
        proposal_bellman = bellman_opt_op.v(mdp, proposal, gamma)
        proposal_residual = jnp.max(jnp.abs(proposal - proposal_bellman))
        bound = rate * state.bound
        accepted = proposal_residual <= bound
        fallback = bellman_opt_op.v(mdp, state.v_val, gamma)
        v_val = jnp.where(accepted, proposal, fallback)
        return replace(
            state,
            v_val=v_val,
            prev_v_val=state.v_val,
            bound=bound,
            accepted=accepted,
        )


@chex.dataclass(frozen=True)
class MomentumValueIteration:
    r"""Perform one Momentum Value Iteration update at a time.

    Each update applies the M-VI recurrence

    .. math::

        v_{s+1}
        = v_s
          - \alpha(v_s-\mathcal{T}^{*}_{V}v_s)
          + \beta(v_s-v_{s-1}).

    By default, ``step_size`` and ``momentum`` use the paper's constant heavy-ball tuning. The paper
    establishes acceleration for reversible policy evaluation, but not general optimal control;
    unsuitable coefficients or transition structure can make M-VI diverge.

    Attributes:
        gamma: Scalar discount in the interval ``[0, 1)``.
        step_size: Positive Bellman-residual step size.
        momentum: Nonnegative value-difference coefficient.

    Public dataclasses:
        State: Current and preceding state-value iterates.

    Public methods:
        init: Initialize with one ordinary value-iteration step.
        update: Apply one momentum value-iteration update.
    """

    gamma: float
    step_size: float | None = None
    momentum: float | None = None

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic M-VI state.

        Attributes:
            v_val: Current state values with shape ``(S,)``.
            prev_v_val: Preceding state values with shape ``(S,)``.
        """

        v_val: jax.Array
        prev_v_val: jax.Array

    def init(
        self,
        mdp: MDP,
        v_val: jax.Array | None = None,
    ) -> MomentumValueIteration.State:
        """Initialize from state values and take one value-iteration step."""
        if v_val is None:
            v_val = jnp.zeros((mdp.state_size,), dtype=mdp.reward.dtype)
        chex.assert_shape(v_val, (mdp.state_size,))
        next_v_val = bellman_opt_op.v(mdp, v_val, self.gamma)
        return self.State(v_val=next_v_val, prev_v_val=v_val)

    def update(
        self,
        mdp: MDP,
        state: MomentumValueIteration.State,
    ) -> MomentumValueIteration.State:
        """Apply one momentum value-iteration update."""
        gamma = jnp.asarray(self.gamma)
        root = jnp.sqrt(1 - gamma**2)
        step_size = 2 / (1 + root) if self.step_size is None else jnp.asarray(self.step_size)
        momentum = (1 - root) / (1 + root) if self.momentum is None else jnp.asarray(self.momentum)
        chex.assert_shape(state.v_val, (mdp.state_size,))
        chex.assert_shape(state.prev_v_val, (mdp.state_size,))
        chex.assert_shape(step_size, (), custom_message="step_size must be scalar")
        chex.assert_shape(momentum, (), custom_message="momentum must be scalar")
        chex.assert_tree_all_finite(state, custom_message="state arrays must be finite")
        chex.assert_tree_all_finite(
            (step_size, momentum),
            custom_message="planner parameters must be finite",
        )
        chex.assert_trees_all_equal(
            step_size > 0,
            jnp.asarray(True),
            custom_message="step_size must be positive",
        )
        chex.assert_trees_all_equal(
            momentum >= 0,
            jnp.asarray(True),
            custom_message="momentum must be nonnegative",
        )

        bellman_v = bellman_opt_op.v(mdp, state.v_val, gamma)
        v_val = (
            state.v_val
            - step_size * (state.v_val - bellman_v)
            + momentum * (state.v_val - state.prev_v_val)
        )
        return replace(state, v_val=v_val, prev_v_val=state.v_val)


@chex.dataclass(frozen=True)
class RankOneValueIteration:
    r"""Perform one Rank-One Value Iteration update at a time.

    Each update applies Algorithm 1 of Rank-One Value Iteration:

    .. math::

        d_k
        = \frac{(P^{\pi_k})^*d_{k-1}}
               {\lVert(P^{\pi_k})^*d_{k-1}\rVert_1},
        \qquad
        v_{k+1}
        = \mathcal{T}^*_Vv_k
          + \frac{\gamma}{1-\gamma}
            \langle d_k,\mathcal{T}^*_Vv_k-v_k\rangle\mathbf{1},

    where ``policy`` is greedy with respect to ``v_val``. The paper's shift argument requires an
    unmasked stochastic transition, so every terminal indicator must be zero. Absorbing states can
    instead be represented directly by their transitions and rewards.

    Attributes:
        gamma: Scalar discount in the interval ``(0, 1)``.

    Public dataclasses:
        State: Current state values and warm-started stationary estimate.

    Public methods:
        init: Initialize values and the stationary estimate.
        update: Apply one rank-one value iteration update.
    """

    gamma: float

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic Rank-One Value Iteration state.

        Attributes:
            v_val: Current state values with shape ``(S,)``.
            dist: Stationary-distribution estimate with shape ``(S,)``.
        """

        v_val: jax.Array
        dist: jax.Array

    def init(
        self,
        mdp: MDP,
        v_val: jax.Array | None = None,
        dist: jax.Array | None = None,
    ) -> RankOneValueIteration.State:
        """Initialize values and the stationary estimate."""
        if v_val is None:
            v_val = jnp.zeros((mdp.state_size,), dtype=mdp.reward.dtype)
        if dist is None:
            dist = jnp.full(
                (mdp.state_size,),
                1 / mdp.state_size,
                dtype=mdp.transition.dtype,
            )
        chex.assert_shape(v_val, (mdp.state_size,))
        chex.assert_shape(dist, (mdp.state_size,))
        return self.State(v_val=v_val, dist=dist)

    def update(
        self,
        mdp: MDP,
        state: RankOneValueIteration.State,
    ) -> RankOneValueIteration.State:
        """Apply one warm-started rank-one value iteration update."""
        gamma = jnp.asarray(self.gamma)
        chex.assert_shape(gamma, (), custom_message="gamma must be scalar")
        chex.assert_shape(state.v_val, (mdp.state_size,))
        chex.assert_shape(state.dist, (mdp.state_size,))
        chex.assert_tree_all_finite(state, custom_message="state arrays must be finite")
        chex.assert_tree_all_finite(gamma, custom_message="gamma must be finite")
        chex.assert_trees_all_equal(
            (gamma > 0) & (gamma < 1),
            jnp.asarray(True),
            custom_message="gamma must be in (0, 1)",
        )
        chex.assert_trees_all_equal(
            jnp.all(mdp.terminal == 0),
            jnp.asarray(True),
            custom_message="Rank-One Value Iteration requires unmasked transitions",
        )
        chex.assert_trees_all_equal(
            jnp.all(state.dist >= 0),
            jnp.asarray(True),
            custom_message="dist must be nonnegative",
        )
        chex.assert_trees_all_close(
            jnp.sum(state.dist),
            jnp.asarray(1, dtype=state.dist.dtype),
            custom_message="dist must sum to one",
        )

        q_val = reward.sa(mdp) + gamma * trans_op.sa(mdp, state.v_val)
        policy = greedy_map.q(q_val)
        mrp = make_mrp(mdp, policy)
        dist = adj_trans_op.s(mrp, state.dist)
        dist = dist / jnp.linalg.norm(dist, ord=1)
        bellman_v = jnp.max(q_val, axis=0)
        correction = gamma / (1 - gamma) * jnp.sum(dist * (bellman_v - state.v_val))
        v_val = bellman_v + correction
        return replace(state, v_val=v_val, dist=dist)


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
    "SafeAcceleratedValueIteration",
    "MomentumValueIteration",
    "RankOneValueIteration",
    "PolicyIteration",
]
