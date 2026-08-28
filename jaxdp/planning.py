"""Planning solvers for finite Markov decision processes."""

from __future__ import annotations

import math
from dataclasses import replace

import chex
import jax
import jax.numpy as jnp

from jaxdp.mapping import greedy_map, reward
from jaxdp.mdp import MDP, make_mrp
from jaxdp.operator import adj_trans_op, bellman_op, bellman_opt_op, resolvent, trans_op


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
class PIDValueIteration:
    r"""Perform one fixed-gain PID Value Iteration update at a time.

    Each update applies the PID VI recurrence

    .. math::

        z_{k+1}
        = \beta z_k+\alpha(\mathcal{T}^{*}_{V}v_k-v_k),
        \qquad
        v_{k+1}
        = v_k
          +\kappa_p(\mathcal{T}^{*}_{V}v_k-v_k)
          +\kappa_I z_{k+1}
          +\kappa_d(v_k-v_{k-1}).

    The default gains ``(kp, ki, kd) = (1, 0, 0)`` recover ordinary Value Iteration. Other gains
    may accelerate or destabilize the recurrence depending on the MDP.

    Attributes:
        gamma: Scalar discount in the interval ``[0, 1)``.
        kp: Proportional gain ``kappa_p``.
        ki: Integral gain ``kappa_I``.
        kd: Derivative gain ``kappa_d``.
        alpha: Bellman-residual input coefficient for the integrator.
        beta: Previous-integrator coefficient.

    Public dataclasses:
        State: Current and previous values and the integrator state.

    Public methods:
        init: Initialize values and a zero integrator state.
        update: Apply one fixed-gain PID VI update.
    """

    gamma: float
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    alpha: float = 0.05
    beta: float = 0.95

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic PID Value Iteration state.

        Attributes:
            v_val: Current state values with shape ``(S,)``.
            prev_v_val: Previous state values with shape ``(S,)``.
            z_val: Integral state with shape ``(S,)``.
        """

        v_val: jax.Array
        prev_v_val: jax.Array
        z_val: jax.Array

    def init(
        self,
        mdp: MDP,
        v_val: jax.Array | None = None,
    ) -> PIDValueIteration.State:
        """Initialize current and previous values with a zero integrator state."""
        if v_val is None:
            v_val = jnp.zeros((mdp.state_size,), dtype=mdp.reward.dtype)
        chex.assert_shape(v_val, (mdp.state_size,))
        return self.State(
            v_val=v_val,
            prev_v_val=v_val,
            z_val=jnp.zeros_like(v_val),
        )

    def update(
        self,
        mdp: MDP,
        state: PIDValueIteration.State,
    ) -> PIDValueIteration.State:
        """Apply one proportional-integral-derivative Bellman update."""
        gamma = jnp.asarray(self.gamma)
        kp = jnp.asarray(self.kp)
        ki = jnp.asarray(self.ki)
        kd = jnp.asarray(self.kd)
        alpha = jnp.asarray(self.alpha)
        beta = jnp.asarray(self.beta)
        chex.assert_shape(state.v_val, (mdp.state_size,))
        chex.assert_shape(state.prev_v_val, (mdp.state_size,))
        chex.assert_shape(state.z_val, (mdp.state_size,))
        chex.assert_shape(gamma, (), custom_message="gamma must be scalar")
        chex.assert_shape(kp, (), custom_message="kp must be scalar")
        chex.assert_shape(ki, (), custom_message="ki must be scalar")
        chex.assert_shape(kd, (), custom_message="kd must be scalar")
        chex.assert_shape(alpha, (), custom_message="alpha must be scalar")
        chex.assert_shape(beta, (), custom_message="beta must be scalar")
        chex.assert_tree_all_finite(state, custom_message="state arrays must be finite")
        chex.assert_tree_all_finite(
            (gamma, kp, ki, kd, alpha, beta),
            custom_message="planner parameters must be finite",
        )
        chex.assert_trees_all_equal(
            (gamma >= 0) & (gamma < 1),
            jnp.asarray(True),
            custom_message="gamma must be in [0, 1)",
        )

        bellman_v = bellman_opt_op.v(mdp, state.v_val, gamma)
        residual = bellman_v - state.v_val
        z_val = beta * state.z_val + alpha * residual
        v_val = state.v_val + kp * residual + ki * z_val + kd * (state.v_val - state.prev_v_val)
        return replace(
            state,
            v_val=v_val,
            prev_v_val=state.v_val,
            z_val=z_val,
        )


@chex.dataclass(frozen=True)
class AndersonValueIteration:
    r"""Perform one Anderson-accelerated value iteration update at a time.

    Each update solves the regularized residual-mixing problem from Anderson Value Iteration:

    .. math::

        \delta_i = \mathcal{T}^{*}_{V}v_i-v_i,
        \qquad
        \alpha
        = \frac{(\Delta_k^\top\Delta_k+\lambda I)^{-1}\mathbf{1}}
               {\mathbf{1}^\top
                (\Delta_k^\top\Delta_k+\lambda I)^{-1}\mathbf{1}},
        \qquad
        v_{k+1}=\sum_i\alpha_i\mathcal{T}^{*}_{V}v_i,

    where ``Delta`` contains at most ``memory + 1`` recent residuals. The linear system is solved
    directly; ``regularization`` keeps it nonsingular when residuals are dependent.

    Attributes:
        gamma: Scalar discount in the interval ``[0, 1)``.
        memory: Number of preceding iterates retained in addition to the current iterate.
        regularization: Positive diagonal regularization for the residual Gram matrix.

    Public dataclasses:
        State: Recent iterates, Bellman images, and latest mixing coefficients.

    Public methods:
        init: Initialize with one ordinary value-iteration step.
        update: Apply one Anderson-accelerated value-iteration update.
    """

    gamma: float
    memory: int = 5
    regularization: float = 1e-6

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic Anderson Value Iteration state.

        Attributes:
            v_val: Current state values with shape ``(S,)``.
            v_hist: Recent state values with shape ``(memory + 1, S)``, newest first.
            bellman_hist: Corresponding Bellman images with shape ``(memory + 1, S)``.
            count: Number of active rows in each history.
            coeff: Mixing coefficients used by the latest update, newest input first.
        """

        v_val: jax.Array
        v_hist: jax.Array
        bellman_hist: jax.Array
        count: jax.Array
        coeff: jax.Array

    def init(
        self,
        mdp: MDP,
        v_val: jax.Array | None = None,
    ) -> AndersonValueIteration.State:
        """Initialize from state values and take the paper's initial VI step."""
        if isinstance(self.memory, bool) or not isinstance(self.memory, int) or self.memory < 1:
            raise ValueError("memory must be a positive integer")
        if v_val is None:
            v_val = jnp.zeros((mdp.state_size,), dtype=mdp.reward.dtype)
        regularization = jnp.asarray(self.regularization)
        chex.assert_shape(v_val, (mdp.state_size,))
        chex.assert_shape(regularization, (), custom_message="regularization must be scalar")
        chex.assert_tree_all_finite(
            regularization,
            custom_message="regularization must be finite",
        )
        chex.assert_trees_all_equal(
            regularization > 0,
            jnp.asarray(True),
            custom_message="regularization must be positive",
        )

        next_v_val = bellman_opt_op.v(mdp, v_val, self.gamma)
        v_val = v_val.astype(next_v_val.dtype)
        next_bellman = bellman_opt_op.v(mdp, next_v_val, self.gamma)
        size = self.memory + 1
        v_hist = jnp.zeros((size, mdp.state_size), dtype=next_v_val.dtype)
        v_hist = v_hist.at[0].set(next_v_val).at[1].set(v_val)
        bellman_hist = jnp.zeros_like(v_hist)
        bellman_hist = bellman_hist.at[0].set(next_bellman).at[1].set(next_v_val)
        coeff = jnp.zeros((size,), dtype=next_v_val.dtype).at[0].set(1)
        return self.State(
            v_val=next_v_val,
            v_hist=v_hist,
            bellman_hist=bellman_hist,
            count=jnp.asarray(2, dtype=jnp.int32),
            coeff=coeff,
        )

    def update(
        self,
        mdp: MDP,
        state: AndersonValueIteration.State,
    ) -> AndersonValueIteration.State:
        """Apply one regularized Anderson value-iteration update."""
        if isinstance(self.memory, bool) or not isinstance(self.memory, int) or self.memory < 1:
            raise ValueError("memory must be a positive integer")
        regularization = jnp.asarray(self.regularization)
        size = self.memory + 1
        chex.assert_shape(state.v_val, (mdp.state_size,))
        chex.assert_shape(state.v_hist, (size, mdp.state_size))
        chex.assert_shape(state.bellman_hist, (size, mdp.state_size))
        chex.assert_shape(state.count, ())
        chex.assert_shape(state.coeff, (size,))
        chex.assert_shape(regularization, (), custom_message="regularization must be scalar")
        chex.assert_tree_all_finite(state, custom_message="state arrays must be finite")
        chex.assert_tree_all_finite(
            regularization,
            custom_message="regularization must be finite",
        )
        chex.assert_trees_all_equal(
            (state.count >= 1) & (state.count <= size),
            jnp.asarray(True),
            custom_message="count must index the history",
        )
        chex.assert_trees_all_equal(
            regularization > 0,
            jnp.asarray(True),
            custom_message="regularization must be positive",
        )

        active = (jnp.arange(size) < state.count).astype(state.v_val.dtype)
        residual = (state.bellman_hist - state.v_hist) * active[:, None]
        gram = residual @ residual.T
        gram += jnp.diag(regularization * active + 1 - active)
        solved = jnp.linalg.solve(gram, active)
        coeff = active * solved / jnp.sum(active * solved)
        v_val = jnp.einsum("i,is->s", coeff, state.bellman_hist)
        bellman_v = bellman_opt_op.v(mdp, v_val, self.gamma)
        v_hist = jnp.concatenate((v_val[None], state.v_hist[:-1]), axis=0)
        bellman_hist = jnp.concatenate(
            (bellman_v[None], state.bellman_hist[:-1]),
            axis=0,
        )
        return replace(
            state,
            v_val=v_val,
            v_hist=v_hist,
            bellman_hist=bellman_hist,
            count=jnp.minimum(state.count + 1, size),
            coeff=coeff,
        )


@chex.dataclass(frozen=True)
class SafeAndersonValueIteration:
    r"""Perform one safeguarded Type-I Anderson value-iteration update at a time.

    For the Bellman residual

    .. math::

        g(v)=v-\mathcal{T}^{*}_{V}v,

    each update applies Algorithm 3 of Globally Convergent Type-I Anderson Acceleration:

    .. math::

        \tilde{v}_{k+1}=v_k-H_kg_k,
        \qquad
        H_k=H_{k-1}
        +\frac{(s_{k-1}-H_{k-1}\tilde{y}_{k-1})
        \hat{s}_{k-1}^{\top}H_{k-1}}
        {\hat{s}_{k-1}^{\top}H_{k-1}\tilde{y}_{k-1}},

    where Powell regularization constructs ``tilde_y`` and restart checking keeps the recent
    directions strongly independent. The trial is accepted only when

    .. math::

        \lVert g_k\rVert_2
        \leq D\lVert g_0\rVert_2(n_{AA}+1)^{-(1+\epsilon)};

    otherwise, the update takes an ordinary Bellman step. The inverse-Jacobian approximation is
    stored as fixed-shape rank-one factors rather than a dense matrix. For discounted value
    iteration, the paper's global convergence result uses an unrelaxed Bellman fallback.

    Attributes:
        gamma: Scalar discount in the interval ``(0, 1)``.
        memory: Maximum number of inverse-Jacobian rank-one factors.
        theta: Powell regularization threshold in ``(0, 1)``.
        tau: Restart threshold in ``(0, 1)``.
        safeguard: Positive safeguard scale ``D``.
        decay: Positive safeguard exponent offset ``epsilon``.

    Public dataclasses:
        State: Current values, residual, matrix-free factors, and safeguard counters.

    Public methods:
        init: Initialize with the first safeguarded Bellman step.
        update: Apply one stabilized Type-I Anderson update.
    """

    gamma: float
    memory: int = 5
    theta: float = 0.01
    tau: float = 0.001
    safeguard: float = 1e6
    decay: float = 1e-6

    @chex.dataclass(frozen=True)
    class State:
        r"""Dynamic safeguarded Type-I Anderson state.

        The factors represent

        .. math::

            Hx=x+\sum_i h_i^{\mathrm{left}}
            \langle h_i^{\mathrm{right}},x\rangle.

        Attributes:
            v_val: Current state values with shape ``(S,)``.
            residual: Current Bellman residual with shape ``(S,)``.
            s_hist: Normalized restart directions with shape ``(memory, S)``.
            h_left: Left inverse-Jacobian factors with shape ``(memory, S)``.
            h_right: Right inverse-Jacobian factors with shape ``(memory, S)``.
            count: Number of active history rows.
            initial_residual: Initial residual norm.
            accepted_count: Number of accepted Anderson trials.
            step: Number of completed updates.
            accepted: Whether the latest trial was accepted.
            restarted: Whether the latest inverse-Jacobian update restarted.
        """

        v_val: jax.Array
        residual: jax.Array
        s_hist: jax.Array
        h_left: jax.Array
        h_right: jax.Array
        count: jax.Array
        initial_residual: jax.Array
        accepted_count: jax.Array
        step: jax.Array
        accepted: jax.Array
        restarted: jax.Array

    def init(
        self,
        mdp: MDP,
        v_val: jax.Array | None = None,
    ) -> SafeAndersonValueIteration.State:
        """Initialize from state values and take the first Bellman step."""
        if isinstance(self.memory, bool) or not isinstance(self.memory, int) or self.memory < 1:
            raise ValueError("memory must be a positive integer")
        if v_val is None:
            v_val = jnp.zeros((mdp.state_size,), dtype=mdp.reward.dtype)
        chex.assert_shape(v_val, (mdp.state_size,))
        bellman_v = bellman_opt_op.v(mdp, v_val, self.gamma)
        v_val = v_val.astype(bellman_v.dtype)
        residual = v_val - bellman_v
        state = self.State(
            v_val=v_val,
            residual=residual,
            s_hist=jnp.zeros((self.memory, mdp.state_size), dtype=bellman_v.dtype),
            h_left=jnp.zeros((self.memory, mdp.state_size), dtype=bellman_v.dtype),
            h_right=jnp.zeros((self.memory, mdp.state_size), dtype=bellman_v.dtype),
            count=jnp.asarray(0, dtype=jnp.int32),
            initial_residual=jnp.linalg.norm(residual),
            accepted_count=jnp.asarray(0, dtype=jnp.int32),
            step=jnp.asarray(0, dtype=jnp.int32),
            accepted=jnp.asarray(False),
            restarted=jnp.asarray(False),
        )
        return self.update(mdp, state)

    def update(
        self,
        mdp: MDP,
        state: SafeAndersonValueIteration.State,
    ) -> SafeAndersonValueIteration.State:
        """Apply one Powell-regularized, restarted, and safeguarded Type-I update."""
        if isinstance(self.memory, bool) or not isinstance(self.memory, int) or self.memory < 1:
            raise ValueError("memory must be a positive integer")
        gamma = jnp.asarray(self.gamma)
        theta = jnp.asarray(self.theta)
        tau = jnp.asarray(self.tau)
        safeguard = jnp.asarray(self.safeguard)
        decay = jnp.asarray(self.decay)
        chex.assert_shape(state.v_val, (mdp.state_size,))
        chex.assert_shape(state.residual, (mdp.state_size,))
        chex.assert_shape(state.s_hist, (self.memory, mdp.state_size))
        chex.assert_shape(state.h_left, (self.memory, mdp.state_size))
        chex.assert_shape(state.h_right, (self.memory, mdp.state_size))
        chex.assert_shape(state.count, ())
        chex.assert_shape(state.initial_residual, ())
        chex.assert_shape(state.accepted_count, ())
        chex.assert_shape(state.step, ())
        chex.assert_shape(state.accepted, ())
        chex.assert_shape(state.restarted, ())
        chex.assert_shape(gamma, (), custom_message="gamma must be scalar")
        chex.assert_shape(theta, (), custom_message="theta must be scalar")
        chex.assert_shape(tau, (), custom_message="tau must be scalar")
        chex.assert_shape(safeguard, (), custom_message="safeguard must be scalar")
        chex.assert_shape(decay, (), custom_message="decay must be scalar")
        chex.assert_tree_all_finite(state, custom_message="state arrays must be finite")
        chex.assert_tree_all_finite(
            (gamma, theta, tau, safeguard, decay),
            custom_message="planner parameters must be finite",
        )
        chex.assert_trees_all_equal(
            (gamma > 0) & (gamma < 1),
            jnp.asarray(True),
            custom_message="gamma must be in (0, 1)",
        )
        chex.assert_trees_all_equal(
            (theta > 0) & (theta < 1),
            jnp.asarray(True),
            custom_message="theta must be in (0, 1)",
        )
        chex.assert_trees_all_equal(
            (tau > 0) & (tau < 1),
            jnp.asarray(True),
            custom_message="tau must be in (0, 1)",
        )
        chex.assert_trees_all_equal(
            (safeguard > 0) & (decay > 0),
            jnp.asarray(True),
            custom_message="safeguard and decay must be positive",
        )
        chex.assert_trees_all_equal(
            (state.count >= 0) & (state.count <= self.memory),
            jnp.asarray(True),
            custom_message="count must index the history",
        )
        chex.assert_trees_all_equal(
            (state.initial_residual >= 0)
            & (state.accepted_count >= 0)
            & (state.step >= state.accepted_count),
            jnp.asarray(True),
            custom_message="safeguard counters must be nonnegative and ordered",
        )

        active = (jnp.arange(self.memory) < state.count).astype(state.v_val.dtype)
        h_residual = state.residual + jnp.einsum(
            "is,i->s",
            state.h_left,
            jnp.einsum("is,s->i", state.h_right, state.residual) * active,
        )
        trial = state.v_val - h_residual
        trial_bellman = bellman_opt_op.v(mdp, trial, gamma)
        trial_residual = trial - trial_bellman
        step_vec = trial - state.v_val
        residual_diff = trial_residual - state.residual
        orthogonal = step_vec - jnp.einsum(
            "is,i->s",
            state.s_hist,
            jnp.einsum("is,s->i", state.s_hist, step_vec) * active,
        )
        step_norm = jnp.linalg.norm(step_vec)
        orthogonal_norm = jnp.linalg.norm(orthogonal)
        restarted = (
            (state.count >= self.memory) | (orthogonal_norm < tau * step_norm) | (step_norm == 0)
        )
        base_active = active * (~restarted).astype(state.v_val.dtype)
        base_count = jnp.where(restarted, 0, state.count)
        s_hist = jnp.where(restarted, jnp.zeros_like(state.s_hist), state.s_hist)
        h_left = jnp.where(restarted, jnp.zeros_like(state.h_left), state.h_left)
        h_right = jnp.where(restarted, jnp.zeros_like(state.h_right), state.h_right)
        s_hat = jnp.where(restarted, step_vec, orthogonal)

        h_residual_diff = residual_diff + jnp.einsum(
            "is,i->s",
            h_left,
            jnp.einsum("is,s->i", h_right, residual_diff) * base_active,
        )
        s_hat_sq = jnp.sum(s_hat**2)
        safe_s_hat_sq = jnp.where(s_hat_sq > 0, s_hat_sq, 1)
        eta = jnp.sum(s_hat * h_residual_diff) / safe_s_hat_sq
        sign = jnp.where(eta >= 0, 1, -1)
        powell = jnp.where(
            jnp.abs(eta) >= theta,
            1,
            (1 - sign * theta) / (1 - eta),
        )
        y_tilde = powell * residual_diff - (1 - powell) * state.residual
        h_y_tilde = y_tilde + jnp.einsum(
            "is,i->s",
            h_left,
            jnp.einsum("is,s->i", h_right, y_tilde) * base_active,
        )
        h_transpose_s = s_hat + jnp.einsum(
            "is,i->s",
            h_right,
            jnp.einsum("is,s->i", h_left, s_hat) * base_active,
        )
        denominator = jnp.sum(s_hat * h_y_tilde)
        safe_denominator = jnp.where(denominator != 0, denominator, 1)
        new_h_left = step_vec - h_y_tilde
        new_h_right = h_transpose_s / safe_denominator
        s_hat_norm = jnp.linalg.norm(s_hat)
        safe_s_hat_norm = jnp.where(s_hat_norm > 0, s_hat_norm, 1)
        s_hist = s_hist.at[base_count].set(s_hat / safe_s_hat_norm)
        h_left = h_left.at[base_count].set(new_h_left)
        h_right = h_right.at[base_count].set(new_h_right)

        accepted_scale = (state.accepted_count + 1).astype(state.v_val.dtype)
        threshold = safeguard * state.initial_residual * accepted_scale ** (-1 - decay)
        accepted = (state.step == 0) | (jnp.linalg.norm(state.residual) <= threshold)
        fallback = state.v_val - state.residual
        fallback_bellman = bellman_opt_op.v(mdp, fallback, gamma)
        fallback_residual = fallback - fallback_bellman
        v_val = jnp.where(accepted, trial, fallback)
        residual = jnp.where(accepted, trial_residual, fallback_residual)
        return replace(
            state,
            v_val=v_val,
            residual=residual,
            s_hist=s_hist,
            h_left=h_left,
            h_right=h_right,
            count=base_count + 1,
            accepted_count=state.accepted_count + accepted.astype(jnp.int32),
            step=state.step + 1,
            accepted=accepted,
            restarted=restarted,
        )


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
class DeflatedValueIteration:
    r"""Perform one rank-one Deflated Dynamics Value Iteration update at a time.

    This is the paper's control form of DDVI. For a fixed state distribution ``dist``, each update
    applies

    .. math::

        w_{k+1}
        = \mathcal{T}^{*}_{V}w_k
          -\gamma\langle\rho,w_k\rangle\mathbf{1},
        \qquad
        v_{k+1}
        = w_{k+1}
          +\frac{\gamma}{1-\gamma}
           \langle\rho,w_{k+1}\rangle\mathbf{1}.

    The rank-one correction only shifts state values by a constant, so it preserves the greedy
    policy sequence. The theorem requires an unmasked stochastic transition; represent terminal
    behavior through absorbing transitions and rewards instead of terminal indicators.

    Attributes:
        gamma: Scalar discount in the interval ``(0, 1)``.

    Public dataclasses:
        State: Deflated iterate, reconstructed values, and the deflation distribution.

    Public methods:
        init: Initialize a consistent deflated iterate from state values.
        update: Apply one rank-one control DDVI update.
    """

    gamma: float

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic rank-one DDVI state.

        Attributes:
            w: Deflated state-space iterate with shape ``(S,)``.
            v_val: Reconstructed state values with shape ``(S,)``.
            dist: Fixed deflation distribution with shape ``(S,)``.
        """

        w: jax.Array
        v_val: jax.Array
        dist: jax.Array

    def init(
        self,
        mdp: MDP,
        v_val: jax.Array | None = None,
        dist: jax.Array | None = None,
    ) -> DeflatedValueIteration.State:
        """Initialize a deflated iterate that reconstructs the supplied values."""
        if v_val is None:
            v_val = jnp.zeros((mdp.state_size,), dtype=mdp.reward.dtype)
        if dist is None:
            dist = jnp.full(
                (mdp.state_size,),
                1 / mdp.state_size,
                dtype=mdp.transition.dtype,
            )
        gamma = jnp.asarray(self.gamma)
        chex.assert_shape(v_val, (mdp.state_size,))
        chex.assert_shape(dist, (mdp.state_size,))
        w = v_val - gamma * jnp.sum(dist * v_val)
        return self.State(w=w, v_val=v_val, dist=dist)

    def update(
        self,
        mdp: MDP,
        state: DeflatedValueIteration.State,
    ) -> DeflatedValueIteration.State:
        """Apply one rank-one Deflated Dynamics Value Iteration update."""
        gamma = jnp.asarray(self.gamma)
        chex.assert_shape(gamma, (), custom_message="gamma must be scalar")
        chex.assert_shape(state.w, (mdp.state_size,))
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
            custom_message="Deflated Value Iteration requires unmasked transitions",
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

        bellman_w = bellman_opt_op.v(mdp, state.w, gamma)
        w = bellman_w - gamma * jnp.sum(state.dist * state.w)
        v_val = w + gamma / (1 - gamma) * jnp.sum(state.dist * w)
        return replace(state, w=w, v_val=v_val)


@chex.dataclass(frozen=True)
class QuasiPolicyIteration:
    r"""Perform one safeguarded Quasi-Policy Iteration update at a time.

    For a fixed stochastic prior transition ``prior``, each update applies the rank-one
    quasi-Newton construction from Theorem 3.1 of Quasi-Policy Iteration:

    .. math::

        G^{\mathrm{pr}}
        &= (I-\gamma P^{\mathrm{pr}})^{-1}, \\
        w_k
        &= \mathcal{T}^{*}_{V}v_k-r^{\pi_k}-\gamma P^{\mathrm{pr}}v_k,
        &\check{w}_k&=G^{\mathrm{pr}}w_k, \\
        u_k
        &=v_k-\frac{\mathbf{1}^{\top}v_k}{S}\mathbf{1},
        &\check{u}_k&=(G^{\mathrm{pr}})^{\top}u_k, \\
        \eta_k
        &=\begin{cases}
          0, & u_k^{\top}v_k=0, \\
          \left[u_k^{\top}(v_k-\check{w}_k)\right]^{-1}, & \text{otherwise},
        \end{cases} \\
        \widetilde{G}_k
        &=G^{\mathrm{pr}}+\eta_k\check{w}_k\check{u}_k^{\top},
        &\widetilde{v}_{k+1}
        &=v_k-\widetilde{G}_k(v_k-\mathcal{T}^{*}_{V}v_k).

    The proposal is accepted when its Bellman residual is at most ``gamma * bound``; otherwise,
    the update returns one ordinary value-iteration step. The default prior is the uniform state
    transition. Terminal behavior must be represented with absorbing transitions and rewards.

    Attributes:
        gamma: Scalar discount in the interval ``(0, 1)``.

    Public dataclasses:
        State: Values, fixed-prior data, safeguard bound, and latest update diagnostics.

    Public methods:
        init: Initialize values, the prior resolvent, and the residual bound.
        update: Propose and safeguard one QPI update.
    """

    gamma: float

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic Quasi-Policy Iteration state.

        Attributes:
            v_val: Current state values with shape ``(S,)``.
            prior: Fixed prior transition with storage shape ``(S_next, S)``.
            prior_resolvent: Fixed prior resolvent with shape ``(S, S)``.
            bound: Scalar Bellman-residual bound for the current iterate.
            gain: Scalar rank-one gain used by the latest proposal.
            accepted: Whether the latest QPI proposal was accepted.
        """

        v_val: jax.Array
        prior: jax.Array
        prior_resolvent: jax.Array
        bound: jax.Array
        gain: jax.Array
        accepted: jax.Array

    def init(
        self,
        mdp: MDP,
        v_val: jax.Array | None = None,
        prior: jax.Array | None = None,
    ) -> QuasiPolicyIteration.State:
        """Initialize QPI from state values and an optional prior transition."""
        if v_val is None:
            v_val = jnp.zeros((mdp.state_size,), dtype=mdp.reward.dtype)
        if prior is None:
            prior = jnp.full(
                (mdp.state_size, mdp.state_size),
                1 / mdp.state_size,
                dtype=mdp.transition.dtype,
            )
        gamma = jnp.asarray(self.gamma)
        chex.assert_shape(gamma, (), custom_message="gamma must be scalar")
        chex.assert_shape(v_val, (mdp.state_size,))
        chex.assert_shape(prior, (mdp.state_size, mdp.state_size))
        chex.assert_tree_all_finite(
            (gamma, v_val, prior),
            custom_message="planner inputs must be finite",
        )
        chex.assert_trees_all_equal(
            (gamma > 0) & (gamma < 1),
            jnp.asarray(True),
            custom_message="gamma must be in (0, 1)",
        )
        chex.assert_trees_all_equal(
            jnp.all(mdp.terminal == 0),
            jnp.asarray(True),
            custom_message="Quasi-Policy Iteration requires unmasked transitions",
        )
        chex.assert_trees_all_equal(
            jnp.all(prior >= 0),
            jnp.asarray(True),
            custom_message="prior must be nonnegative",
        )
        chex.assert_trees_all_close(
            jnp.sum(prior, axis=0),
            jnp.ones((mdp.state_size,), dtype=prior.dtype),
            custom_message="prior columns must sum to one",
        )

        identity = jnp.eye(mdp.state_size, dtype=prior.dtype)
        prior_resolvent = jnp.linalg.solve(identity - gamma * prior.T, identity)
        bellman_v = bellman_opt_op.v(mdp, v_val, gamma)
        bound = jnp.max(jnp.abs(v_val - bellman_v))
        return self.State(
            v_val=v_val,
            prior=prior,
            prior_resolvent=prior_resolvent,
            bound=bound,
            gain=jnp.zeros((), dtype=v_val.dtype),
            accepted=jnp.asarray(False),
        )

    def update(
        self,
        mdp: MDP,
        state: QuasiPolicyIteration.State,
    ) -> QuasiPolicyIteration.State:
        """Propose and safeguard one Quasi-Policy Iteration update."""
        gamma = jnp.asarray(self.gamma)
        chex.assert_shape(gamma, (), custom_message="gamma must be scalar")
        chex.assert_shape(state.v_val, (mdp.state_size,))
        chex.assert_shape(state.prior, (mdp.state_size, mdp.state_size))
        chex.assert_shape(state.prior_resolvent, (mdp.state_size, mdp.state_size))
        chex.assert_shape(state.bound, ())
        chex.assert_shape(state.gain, ())
        chex.assert_shape(state.accepted, ())
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
            custom_message="Quasi-Policy Iteration requires unmasked transitions",
        )
        chex.assert_trees_all_equal(
            state.bound >= 0,
            jnp.asarray(True),
            custom_message="bound must be nonnegative",
        )

        bellman_v = bellman_opt_op.v(mdp, state.v_val, gamma)
        policy = greedy_map.v(mdp, state.v_val, gamma)
        policy_reward = reward.s(mdp, policy)
        prior_v = jnp.einsum("xs,x->s", state.prior, state.v_val)
        w = bellman_v - policy_reward - gamma * prior_v
        checked_w = state.prior_resolvent @ w
        u = state.v_val - jnp.mean(state.v_val)
        checked_u = state.prior_resolvent.T @ u
        constraint = jnp.dot(u, state.v_val)
        gain_denominator = jnp.dot(u, state.v_val - checked_w)
        safe_denominator = jnp.where(
            constraint == 0,
            jnp.ones_like(gain_denominator),
            gain_denominator,
        )
        gain = jnp.where(
            constraint == 0,
            jnp.zeros_like(gain_denominator),
            1 / safe_denominator,
        )
        residual = state.v_val - bellman_v
        proposal = (
            state.v_val
            - state.prior_resolvent @ residual
            - gain * checked_w * jnp.dot(checked_u, residual)
        )
        proposal_bellman = bellman_opt_op.v(mdp, proposal, gamma)
        proposal_residual = jnp.max(jnp.abs(proposal - proposal_bellman))
        bound = gamma * state.bound
        accepted = proposal_residual <= bound
        v_val = jnp.where(accepted, proposal, bellman_v)
        return replace(
            state,
            v_val=v_val,
            bound=bound,
            gain=gain,
            accepted=accepted,
        )


@chex.dataclass(frozen=True)
class DynamicBoltzmannValueIteration:
    r"""Perform one Dynamic Boltzmann Value Iteration update at a time.

    The planner uses the power schedule from Theorem 2 of Dynamic Boltzmann Softmax Updates. With
    ``step`` equal to the number of completed updates, the next update applies

    .. math::

        q_{t+1}(s,a)
        &= r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)v_t(s'), \\
        \beta_{t+1}
        &= (t+1)^p, \\
        \pi_{t+1}(a\mid s)
        &= \frac{\exp(\beta_{t+1}q_{t+1}(s,a))}
                 {\sum_b\exp(\beta_{t+1}q_{t+1}(s,b))}, \\
        v_{t+1}(s)
        &= \sum_a\pi_{t+1}(a\mid s)q_{t+1}(s,a).

    The inverse temperature tends to infinity for every positive ``power``, as required by the
    paper's convergence theorem. The default quadratic schedule is its primary empirical choice.

    Attributes:
        gamma: Scalar discount in the interval ``[0, 1)``.
        power: Positive exponent ``p`` of the inverse-temperature schedule.

    Public dataclasses:
        State: Current values and policy, completed-update count, and latest inverse temperature.

    Public methods:
        init: Initialize values before the first update.
        update: Apply one dynamic Boltzmann update.
    """

    gamma: float
    power: float = 2.0

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic Boltzmann Value Iteration state.

        Attributes:
            v_val: Current state values with shape ``(S,)``.
            policy: Latest Boltzmann policy with shape ``(A, S)``.
            step: Number of completed updates.
            beta: Scalar inverse temperature used by the latest update.
        """

        v_val: jax.Array
        policy: jax.Array
        step: jax.Array
        beta: jax.Array

    def init(
        self,
        mdp: MDP,
        v_val: jax.Array | None = None,
    ) -> DynamicBoltzmannValueIteration.State:
        """Initialize values before the first dynamic Boltzmann update."""
        if v_val is None:
            v_val = jnp.zeros((mdp.state_size,), dtype=mdp.reward.dtype)
        gamma = jnp.asarray(self.gamma)
        power = jnp.asarray(self.power)
        chex.assert_shape(v_val, (mdp.state_size,))
        chex.assert_shape(gamma, (), custom_message="gamma must be scalar")
        chex.assert_shape(power, (), custom_message="power must be scalar")
        chex.assert_tree_all_finite(
            (v_val, gamma, power),
            custom_message="planner inputs must be finite",
        )
        chex.assert_trees_all_equal(
            (gamma >= 0) & (gamma < 1),
            jnp.asarray(True),
            custom_message="gamma must be in [0, 1)",
        )
        chex.assert_trees_all_equal(
            power > 0,
            jnp.asarray(True),
            custom_message="power must be positive",
        )
        return self.State(
            v_val=v_val,
            policy=jnp.full(
                (mdp.action_size, mdp.state_size),
                1 / mdp.action_size,
                dtype=mdp.transition.dtype,
            ),
            step=jnp.zeros((), dtype=jnp.int32),
            beta=jnp.zeros((), dtype=v_val.dtype),
        )

    def update(
        self,
        mdp: MDP,
        state: DynamicBoltzmannValueIteration.State,
    ) -> DynamicBoltzmannValueIteration.State:
        """Apply one dynamic Boltzmann value-iteration update."""
        gamma = jnp.asarray(self.gamma)
        power = jnp.asarray(self.power)
        chex.assert_shape(state.v_val, (mdp.state_size,))
        chex.assert_shape(state.policy, (mdp.action_size, mdp.state_size))
        chex.assert_shape(state.step, ())
        chex.assert_shape(state.beta, ())
        chex.assert_shape(gamma, (), custom_message="gamma must be scalar")
        chex.assert_shape(power, (), custom_message="power must be scalar")
        chex.assert_tree_all_finite(state, custom_message="state arrays must be finite")
        chex.assert_tree_all_finite(
            (gamma, power),
            custom_message="planner parameters must be finite",
        )
        chex.assert_trees_all_equal(
            (gamma >= 0) & (gamma < 1),
            jnp.asarray(True),
            custom_message="gamma must be in [0, 1)",
        )
        chex.assert_trees_all_equal(
            (power > 0) & (state.step >= 0) & (state.beta >= 0),
            jnp.asarray(True),
            custom_message="power must be positive and schedule state nonnegative",
        )

        q_val = reward.sa(mdp) + gamma * trans_op.sa(mdp, state.v_val)
        step = state.step + 1
        beta = step.astype(q_val.dtype) ** power
        centered_q = q_val - jnp.max(q_val, axis=0, keepdims=True)
        policy = jax.nn.softmax(beta * centered_q, axis=0)
        v_val = jnp.sum(policy * q_val, axis=0)
        return replace(state, v_val=v_val, policy=policy, step=step, beta=beta)


@chex.dataclass(frozen=True)
class AcceleratedPolicyIteration:
    r"""Perform one degree-``d`` Accelerated Policy Iteration micro-step.

    While the current policy residual exceeds ``tolerance``, ``update`` applies one inner
    accelerated evaluation step from Algorithm 4.1:

    .. math::

        x_{\ell+1}=\mathcal{T}^{\pi_k}_{V}y_\ell,
        \qquad
        y_{\ell+1}
        = \left(1+\sum_{i=0}^{d-2}\alpha_i\right)x_{\ell+1}
          -\alpha_{d-2}x_\ell-\cdots-\alpha_0x_{\ell-d+2},

    .. math::

        \alpha_i
        = \binom{d}{i}
          \frac{\left((1-\gamma)^{1/d}-1\right)^{d-i}}{\gamma}.

    Once the residual is within tolerance, ``update`` performs one exact greedy policy
    improvement and preserves the evaluation history for the next policy. Convergence requires
    the policy-transition spectra to satisfy the paper's degree-``d`` accelerability condition.

    Attributes:
        gamma: Scalar discount in the interval ``(0, 1)``.
        degree: Acceleration degree, at least two.
        tolerance: Nonnegative policy-evaluation residual tolerance.

    Public dataclasses:
        State: Policy, approximate values, evaluation history, and phase indicators.

    Public methods:
        init: Initialize a policy and its accelerated evaluation history.
        update: Apply one evaluation or policy-improvement micro-step.
    """

    gamma: float
    degree: int = 2
    tolerance: float = 1e-6

    @chex.dataclass(frozen=True)
    class State:
        """Dynamic degree-``d`` Accelerated Policy Iteration state.

        Attributes:
            policy: Current action probabilities with shape ``(A, S)``.
            v_val: Current approximate policy values with shape ``(S,)``.
            history: Most recent ``d - 1`` intermediate values with shape ``(d - 1, S)``.
            improved: Whether the latest micro-step improved the policy.
            stable: Whether the latest improvement retained the policy.
        """

        policy: jax.Array
        v_val: jax.Array
        history: jax.Array
        improved: jax.Array
        stable: jax.Array

    def init(
        self,
        mdp: MDP,
        policy: jax.Array,
        v_val: jax.Array | None = None,
    ) -> AcceleratedPolicyIteration.State:
        """Initialize a policy and its evaluation history."""
        if isinstance(self.degree, bool) or not isinstance(self.degree, int) or self.degree < 2:
            raise ValueError("degree must be an integer of at least two")
        if v_val is None:
            v_val = jnp.zeros((mdp.state_size,), dtype=mdp.reward.dtype)
        chex.assert_shape(policy, (mdp.action_size, mdp.state_size))
        chex.assert_shape(v_val, (mdp.state_size,))
        history = jnp.broadcast_to(v_val, (self.degree - 1, mdp.state_size))
        return self.State(
            policy=policy,
            v_val=v_val,
            history=history,
            improved=jnp.asarray(False),
            stable=jnp.asarray(False),
        )

    def update(
        self,
        mdp: MDP,
        state: AcceleratedPolicyIteration.State,
    ) -> AcceleratedPolicyIteration.State:
        """Apply one accelerated evaluation or exact improvement micro-step."""
        if isinstance(self.degree, bool) or not isinstance(self.degree, int) or self.degree < 2:
            raise ValueError("degree must be an integer of at least two")
        gamma = jnp.asarray(self.gamma)
        tolerance = jnp.asarray(self.tolerance)
        chex.assert_shape(gamma, (), custom_message="gamma must be scalar")
        chex.assert_shape(tolerance, (), custom_message="tolerance must be scalar")
        chex.assert_shape(state.policy, (mdp.action_size, mdp.state_size))
        chex.assert_shape(state.v_val, (mdp.state_size,))
        chex.assert_shape(state.history, (self.degree - 1, mdp.state_size))
        chex.assert_shape(state.improved, ())
        chex.assert_shape(state.stable, ())
        chex.assert_tree_all_finite(state, custom_message="state arrays must be finite")
        chex.assert_tree_all_finite(
            (gamma, tolerance),
            custom_message="planner parameters must be finite",
        )
        chex.assert_trees_all_equal(
            (gamma > 0) & (gamma < 1),
            jnp.asarray(True),
            custom_message="gamma must be in (0, 1)",
        )
        chex.assert_trees_all_equal(
            tolerance >= 0,
            jnp.asarray(True),
            custom_message="tolerance must be nonnegative",
        )

        policy_bellman = bellman_op.v(mdp, state.policy, state.v_val, gamma)
        residual = jnp.max(jnp.abs(state.v_val - policy_bellman))
        epsilon = 1 - gamma
        index = jnp.arange(self.degree - 1)
        binomial = jnp.asarray(
            [math.comb(self.degree, i) for i in range(self.degree - 1)],
            dtype=state.v_val.dtype,
        )
        alpha = binomial * (epsilon ** (1 / self.degree) - 1) ** (self.degree - index) / gamma
        eval_v_val = (1 + jnp.sum(alpha)) * policy_bellman - jnp.einsum(
            "i,is->s",
            jnp.flip(alpha),
            state.history,
        )
        eval_history = jnp.concatenate(
            (policy_bellman[None], state.history[:-1]),
            axis=0,
        )
        greedy_policy = greedy_map.v(mdp, state.v_val, gamma)
        improved = residual <= tolerance
        stable = improved & jnp.all(greedy_policy == state.policy)
        policy = jnp.where(improved, greedy_policy, state.policy)
        v_val = jnp.where(improved, state.v_val, eval_v_val)
        history = jnp.where(improved, state.history, eval_history)
        return replace(
            state,
            policy=policy,
            v_val=v_val,
            history=history,
            improved=improved,
            stable=stable,
        )


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
    "PIDValueIteration",
    "AndersonValueIteration",
    "SafeAndersonValueIteration",
    "RankOneValueIteration",
    "DeflatedValueIteration",
    "QuasiPolicyIteration",
    "DynamicBoltzmannValueIteration",
    "AcceleratedPolicyIteration",
    "PolicyIteration",
]
