"""Finite Markov reward processes."""

import chex
import jax
import jax.numpy as jnp
from chex import dataclass

from jaxdp.mdp.mdp import MDP

_ATOL = 1e-5


@dataclass(frozen=True)
class MRP:
    r"""Finite Markov reward process.

    .. math::

        \mathcal{R}=(P,r,\mu,\tau)

    The transition convention is ``transition[..., s_next, s]``. Rewards are expected immediate
    state rewards, and every array uses the same leading batch shape. Terminal states are absorbing
    with zero outgoing reward; truncation is external to this model.

    Attributes:
        transition: Column-stochastic transition probabilities with shape ``(..., S, S)``.
        reward: Expected immediate state rewards with shape ``(..., S)``.
        initial: Initial state distribution with shape ``(..., S)``.
        terminal: Terminal-state indicators with shape ``(..., S)``.

    Methods:
        validate: Validate shapes, probabilities, and terminal semantics.
    """

    transition: jax.Array
    reward: jax.Array
    initial: jax.Array
    terminal: jax.Array

    @property
    def state_size(self) -> int:
        """Number of states."""
        return self.transition.shape[-1]

    def validate(self) -> None:
        """Validate shapes, values, and terminal semantics with Chex."""
        chex.assert_shape(
            self.transition,
            (..., None, None),
            custom_message="transition shape must be (..., S, S)",
        )
        chex.assert_axis_dimension_gt(self.transition, -1, 0)
        chex.assert_axis_dimension(
            self.transition,
            -2,
            self.state_size,
            custom_message="transition shape must be (..., S, S)",
        )

        state_shape = (*self.transition.shape[:-2], self.state_size)
        chex.assert_shape(
            self.reward,
            state_shape,
            custom_message="reward shape must match the MRP batch and state dimensions",
        )
        chex.assert_shape(
            self.initial,
            state_shape,
            custom_message="initial shape must match the MRP batch and state dimensions",
        )
        chex.assert_shape(
            self.terminal,
            state_shape,
            custom_message="terminal shape must match the MRP batch and state dimensions",
        )

        chex.assert_tree_all_finite(self, custom_message="MRP arrays must be finite")

        chex.assert_trees_all_equal(
            jnp.all(self.transition >= 0),
            jnp.asarray(True),
            custom_message="transition probabilities must be nonnegative",
        )
        transition_mass = self.transition.sum(axis=-2)
        chex.assert_trees_all_close(
            transition_mass,
            jnp.ones_like(transition_mass),
            atol=_ATOL,
            rtol=0.0,
            custom_message="transition must be column stochastic",
        )

        chex.assert_trees_all_equal(
            jnp.all(self.initial >= 0),
            jnp.asarray(True),
            custom_message="initial probabilities must be nonnegative",
        )
        initial_mass = self.initial.sum(axis=-1)
        chex.assert_trees_all_close(
            initial_mass,
            jnp.ones_like(initial_mass),
            atol=_ATOL,
            rtol=0.0,
            custom_message="initial probabilities must sum to one",
        )

        chex.assert_trees_all_equal(
            jnp.all((self.terminal == 0) | (self.terminal == 1)),
            jnp.asarray(True),
            custom_message="terminal indicators must be zero or one",
        )

        identity = jnp.eye(self.state_size, dtype=self.transition.dtype)
        terminal_transition = (self.transition - identity) * self.terminal[..., None, :]
        chex.assert_trees_all_close(
            terminal_transition,
            jnp.zeros_like(terminal_transition),
            atol=_ATOL,
            rtol=0.0,
            custom_message="terminal states must be absorbing",
        )

        terminal_reward = self.reward * self.terminal
        chex.assert_trees_all_close(
            terminal_reward,
            jnp.zeros_like(terminal_reward),
            atol=_ATOL,
            rtol=0.0,
            custom_message="rewards originating from terminal states must be zero",
        )


def make_mrp(mdp: MDP, policy: jax.Array) -> MRP:
    """Induce a finite MRP by fixing a policy in an MDP.

    Args:
        mdp: Finite Markov decision process.
        policy: Action probabilities with shape ``(..., A, S)``.

    Returns:
        Policy-induced MRP.
    """
    policy_shape = (*mdp.transition.shape[:-3], mdp.action_size, mdp.state_size)
    chex.assert_shape(
        policy,
        policy_shape,
        custom_message="policy shape must match the MDP batch, action, and state dimensions",
    )
    chex.assert_tree_all_finite(policy, custom_message="policy must be finite")
    chex.assert_trees_all_equal(
        jnp.all(policy >= 0),
        jnp.asarray(True),
        custom_message="policy probabilities must be nonnegative",
    )
    policy_mass = policy.sum(axis=-2)
    chex.assert_trees_all_close(
        policy_mass,
        jnp.ones_like(policy_mass),
        atol=_ATOL,
        rtol=0.0,
        custom_message="policy probabilities must sum to one for each state",
    )

    transition = jnp.einsum("...as,...axs->...xs", policy, mdp.transition)
    reward = jnp.einsum("...as,...asx,...axs->...s", policy, mdp.reward, mdp.transition)
    mrp = MRP(
        transition=transition,
        reward=reward,
        initial=mdp.initial,
        terminal=mdp.terminal,
    )
    mrp.validate()
    return mrp


__all__ = ["MRP", "make_mrp"]
