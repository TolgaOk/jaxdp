"""Backward value operators for finite MDPs."""

import math

import chex
import jax
import jax.numpy as jnp
from jax import core

from jaxdp.mdp import Mdp


def _validate_gamma(gamma: float | jax.Array) -> None:
    gamma_array = jnp.asarray(gamma)
    if gamma_array.shape != ():
        raise ValueError("gamma must be scalar")
    if isinstance(gamma_array, core.Tracer):
        return
    if not math.isfinite(float(gamma_array)) or not 0 <= float(gamma_array) < 1:
        raise ValueError("gamma must be finite and in the interval [0, 1)")


def _reward(mdp: Mdp) -> jax.Array:
    return jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)


def _policy_dynamics(mdp: Mdp, policy: jax.Array) -> tuple[jax.Array, jax.Array]:
    transition = jnp.einsum("as,axs->xs", policy, mdp.transition)
    reward = jnp.einsum("as,asx,axs->s", policy, mdp.reward, mdp.transition)
    continuation = transition * (1 - mdp.terminal[:, None])
    return continuation, reward


def greedy_state_value(value: jax.Array) -> jax.Array:
    """Return greedy state values from an ``(A, S)`` action-value array."""
    return jnp.max(value, axis=0)


def state_action_value(mdp: Mdp, value: jax.Array, gamma: float | jax.Array) -> jax.Array:
    """Return one-step action values from an ``(S,)`` state-value array."""
    _validate_gamma(gamma)
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

    def q(self, mdp: Mdp, policy: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Return exact action values for a policy."""
        return state_action_value(mdp, self.v(mdp, policy, gamma), gamma)

    def v(self, mdp: Mdp, policy: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Return exact state values for a policy using a linear solve."""
        _validate_gamma(gamma)
        transition, reward = _policy_dynamics(mdp, policy)
        return jnp.linalg.solve(
            jnp.eye(mdp.state_size, dtype=mdp.transition.dtype) - gamma * transition.T,
            reward,
        )


@chex.dataclass(frozen=True)
class Bellman:
    """Discounted Bellman policy operator."""

    def q(
        self,
        mdp: Mdp,
        policy: jax.Array,
        value: jax.Array,
        gamma: float | jax.Array,
    ) -> jax.Array:
        """Apply the Bellman policy operator to action values."""
        next_value = jnp.einsum("as,as->s", policy, value)
        return state_action_value(mdp, next_value, gamma)

    def v(
        self,
        mdp: Mdp,
        policy: jax.Array,
        value: jax.Array,
        gamma: float | jax.Array,
    ) -> jax.Array:
        """Apply the Bellman policy operator to state values."""
        action_value = state_action_value(mdp, value, gamma)
        return jnp.einsum("as,as->s", policy, action_value)


@chex.dataclass(frozen=True)
class BellmanOptimality:
    """Discounted Bellman optimality operator."""

    def q(self, mdp: Mdp, value: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Apply Bellman optimality to action values."""
        return state_action_value(mdp, greedy_state_value(value), gamma)

    def v(self, mdp: Mdp, value: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Apply Bellman optimality to state values."""
        return greedy_state_value(state_action_value(mdp, value, gamma))


__all__ = [
    "Expected",
    "PolicyEvaluation",
    "Bellman",
    "BellmanOptimality",
    "greedy_state_value",
    "state_action_value",
]
