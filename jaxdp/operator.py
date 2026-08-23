"""Backward value operators for finite MDPs."""

import math

import chex
import jax
import jax.numpy as jnp

from jaxdp.mdp import Mdp


def _validate_gamma(gamma: float) -> None:
    if not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite and in the interval [0, 1)")


def _reward(mdp: Mdp) -> jax.Array:
    return jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)


def _policy_dynamics(mdp: Mdp, policy: jax.Array) -> tuple[jax.Array, jax.Array]:
    transition = jnp.einsum("as,axs->xs", policy, mdp.transition)
    reward = jnp.einsum("as,asx->sx", policy, mdp.reward)
    return transition, reward


def greedy_state_value(value: jax.Array) -> jax.Array:
    """Return greedy state values from an ``(A, S)`` action-value array."""
    return jnp.max(value, axis=0)


def state_action_value(mdp: Mdp, value: jax.Array, gamma: float) -> jax.Array:
    """Return one-step action values from an ``(S,)`` state-value array."""
    continuation = jnp.einsum(
        "axs,x,x->as",
        mdp.transition,
        value,
        1 - mdp.terminal,
    )
    return _reward(mdp) + gamma * continuation


@chex.dataclass(frozen=True)
class Expected:
    """Initial-distribution expectation."""

    def q(self, mdp: Mdp, value: jax.Array) -> jax.Array:
        """Return the expected greedy action value."""
        return self.v(mdp, greedy_state_value(value))

    def v(self, mdp: Mdp, value: jax.Array) -> jax.Array:
        """Return the expected state value."""
        return jnp.sum(mdp.initial * value)


@chex.dataclass(frozen=True)
class PolicyEvaluation:
    """Exact discounted policy evaluation."""

    gamma: float

    def __post_init__(self) -> None:
        _validate_gamma(self.gamma)

    def q(self, mdp: Mdp, policy: jax.Array) -> jax.Array:
        """Return exact action values for a policy."""
        return state_action_value(mdp, self.v(mdp, policy), self.gamma)

    def v(self, mdp: Mdp, policy: jax.Array) -> jax.Array:
        """Return exact state values for a policy using a linear solve."""
        transition, reward = _policy_dynamics(mdp, policy)
        return jnp.linalg.solve(
            jnp.eye(mdp.state_size) - self.gamma * transition.T,
            jnp.einsum("xs,sx->s", transition, reward),
        )


@chex.dataclass(frozen=True)
class Bellman:
    """Discounted Bellman policy operator."""

    gamma: float

    def __post_init__(self) -> None:
        _validate_gamma(self.gamma)

    def q(self, mdp: Mdp, policy: jax.Array, value: jax.Array) -> jax.Array:
        """Apply the Bellman policy operator to action values."""
        next_value = jnp.einsum("as,as->s", policy, value)
        return state_action_value(mdp, next_value, self.gamma)

    def v(self, mdp: Mdp, policy: jax.Array, value: jax.Array) -> jax.Array:
        """Apply the Bellman policy operator to state values."""
        action_value = state_action_value(mdp, value, self.gamma)
        return jnp.einsum("as,as->s", policy, action_value)


@chex.dataclass(frozen=True)
class Optimality:
    """Discounted Bellman optimality operator."""

    gamma: float

    def __post_init__(self) -> None:
        _validate_gamma(self.gamma)

    def q(self, mdp: Mdp, value: jax.Array) -> jax.Array:
        """Apply optimality to action values."""
        return state_action_value(mdp, greedy_state_value(value), self.gamma)

    def v(self, mdp: Mdp, value: jax.Array) -> jax.Array:
        """Apply optimality to state values."""
        return greedy_state_value(state_action_value(mdp, value, self.gamma))


__all__ = [
    "Expected",
    "PolicyEvaluation",
    "Bellman",
    "Optimality",
    "greedy_state_value",
    "state_action_value",
]
