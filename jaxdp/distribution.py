"""Forward state and action-state distributions for finite MDPs."""

import chex
import jax
import jax.numpy as jnp

from jaxdp.mdp import MDP
from jaxdp.operator import _assert_policy


@chex.dataclass(frozen=True)
class Occupancy:
    """Finite-step distribution propagation from the MDP initial distribution."""

    steps: int

    def __post_init__(self) -> None:
        """Validate the step count."""
        steps = jnp.asarray(self.steps)
        chex.assert_shape(steps, (), custom_message="steps must be scalar")
        chex.assert_type(steps, int, custom_message="steps must be an integer")
        chex.assert_trees_all_equal(
            steps >= 0,
            jnp.asarray(True),
            custom_message="steps must be nonnegative",
        )

    def q(self, mdp: MDP, policy: jax.Array) -> jax.Array:
        """Return the action-state distribution after the configured number of steps."""
        return policy * self.v(mdp, policy)

    def v(self, mdp: MDP, policy: jax.Array) -> jax.Array:
        """Return the state distribution after the configured number of steps."""
        transition = _policy_transition(mdp, policy)
        return jax.lax.fori_loop(
            0,
            self.steps,
            lambda _, distribution: transition @ distribution,
            mdp.initial,
        )


@chex.dataclass(frozen=True)
class Stationary:
    """Invariant distribution of the policy-induced Markov chain."""

    def q(self, mdp: MDP, policy: jax.Array) -> jax.Array:
        """Return the invariant action-state distribution."""
        return policy * self.v(mdp, policy)

    def v(self, mdp: MDP, policy: jax.Array) -> jax.Array:
        """Return the normalized minimum-norm invariant state distribution."""
        transition = _policy_transition(mdp, policy)
        state_size = mdp.state_size
        system = jnp.concatenate(
            (
                transition - jnp.eye(state_size, dtype=transition.dtype),
                jnp.ones((1, state_size), dtype=transition.dtype),
            ),
            axis=0,
        )
        target = jnp.concatenate(
            (
                jnp.zeros(state_size, dtype=transition.dtype),
                jnp.ones(1, dtype=transition.dtype),
            )
        )
        distribution = jnp.linalg.lstsq(system, target, rcond=None)[0]
        distribution = jnp.maximum(distribution, 0)
        return distribution / jnp.sum(distribution)


def eigenvalues(mdp: MDP, policy: jax.Array) -> jax.Array:
    """Return eigenvalues of the policy-induced transition matrix."""
    return jnp.linalg.eigvals(_policy_transition(mdp, policy))


def _policy_transition(mdp: MDP, policy: jax.Array) -> jax.Array:
    _assert_policy(mdp, policy)
    return jnp.einsum("as,axs->xs", policy, mdp.transition)


__all__ = ["Occupancy", "Stationary", "eigenvalues"]
