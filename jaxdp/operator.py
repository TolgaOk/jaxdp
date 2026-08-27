"""Backward value operators for finite MDPs and MRPs."""

import chex
import jax
import jax.numpy as jnp

from jaxdp.mdp import MDP, MRP

_ATOL = 1e-5


def greedy_state_value(value: jax.Array) -> jax.Array:
    """Return greedy state values from an ``(A, S)`` action-value array."""
    chex.assert_rank(value, 2)
    return jnp.max(value, axis=0)


def state_action_value(mdp: MDP, value: jax.Array, gamma: float | jax.Array) -> jax.Array:
    """Return one-step action values from an ``(S,)`` state-value array."""
    return _state_action_value(mdp, value, _assert_gamma(gamma))


@chex.dataclass(frozen=True)
class Expected:
    """Initial-distribution expectation."""

    def q(self, mdp: MDP, value: jax.Array) -> jax.Array:
        """Return the expected greedy action value."""
        return self.v(mdp, greedy_state_value(value))

    def v(self, mdp: MDP, value: jax.Array) -> jax.Array:
        """Return the expected state value."""
        chex.assert_shape(value, (mdp.state_size,))
        return jnp.sum(mdp.initial * value)


@chex.dataclass(frozen=True)
class PolicyEvaluation:
    """Exact discounted MRP evaluation."""

    def q(self, mdp: MDP, mrp: MRP, gamma: float | jax.Array) -> jax.Array:
        """Return exact action values using MDP actions and MRP state values."""
        gamma_array = _assert_gamma(gamma)
        value = _policy_value(mrp, gamma_array)
        return _state_action_value(mdp, value, gamma_array)

    def v(self, mrp: MRP, gamma: float | jax.Array) -> jax.Array:
        """Return exact MRP state values using a linear solve."""
        gamma_array = _assert_gamma(gamma)
        return _policy_value(mrp, gamma_array)


@chex.dataclass(frozen=True)
class Bellman:
    """Discounted Bellman policy operator."""

    def q(
        self,
        mdp: MDP,
        policy: jax.Array,
        value: jax.Array,
        gamma: float | jax.Array,
    ) -> jax.Array:
        """Apply the Bellman policy operator to action values."""
        gamma_array = _assert_gamma(gamma)
        _assert_policy(mdp, policy)
        chex.assert_shape(value, (mdp.action_size, mdp.state_size))
        next_value = jnp.einsum("as,as->s", policy, value)
        return _state_action_value(mdp, next_value, gamma_array)

    def v(
        self,
        mdp: MDP,
        policy: jax.Array,
        value: jax.Array,
        gamma: float | jax.Array,
    ) -> jax.Array:
        """Apply the Bellman policy operator to state values."""
        gamma_array = _assert_gamma(gamma)
        _assert_policy(mdp, policy)
        action_value = _state_action_value(mdp, value, gamma_array)
        return jnp.einsum("as,as->s", policy, action_value)


@chex.dataclass(frozen=True)
class BellmanOptimality:
    """Discounted Bellman optimality operator."""

    def q(self, mdp: MDP, value: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Apply Bellman optimality to action values."""
        return state_action_value(mdp, greedy_state_value(value), gamma)

    def v(self, mdp: MDP, value: jax.Array, gamma: float | jax.Array) -> jax.Array:
        """Apply Bellman optimality to state values."""
        return greedy_state_value(state_action_value(mdp, value, gamma))


def _reward(mdp: MDP) -> jax.Array:
    return jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)


def _state_action_value(mdp: MDP, value: jax.Array, gamma: jax.Array) -> jax.Array:
    chex.assert_shape(value, (mdp.state_size,))
    continuation = jnp.einsum(
        "axs,x,x->as",
        mdp.transition,
        value,
        1 - mdp.terminal,
    )
    return _reward(mdp) + gamma * continuation


def _policy_value(mrp: MRP, gamma: jax.Array) -> jax.Array:
    continuation = mrp.transition * (1 - mrp.terminal)[..., :, None]
    return jnp.linalg.solve(
        jnp.eye(mrp.state_size, dtype=mrp.transition.dtype)
        - gamma * jnp.swapaxes(continuation, -1, -2),
        mrp.reward,
    )


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
    "Expected",
    "PolicyEvaluation",
    "Bellman",
    "BellmanOptimality",
    "greedy_state_value",
    "state_action_value",
]
