"""Finite Markov decision processes."""

import chex
import jax
import jax.numpy as jnp
from chex import dataclass

_ATOL = 1e-5


@dataclass(frozen=True)
class MDP:
    r"""Finite Markov decision process.

    .. math::

        \mathcal{M}=(P,R,\mu,\tau)

    The transition convention is ``transition[..., a, s_next, s]`` and the reward convention is
    ``reward[..., a, s, s_next]``. Every array uses the same leading batch shape. Terminal states
    are absorbing with zero outgoing reward; truncation is external to this model.

    Attributes:
        transition: Column-stochastic transition probabilities with shape ``(..., A, S, S)``.
        reward: Transition rewards with shape ``(..., A, S, S)``.
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

    @property
    def action_size(self) -> int:
        """Number of actions."""
        return self.transition.shape[-3]

    def validate(self) -> None:
        """Validate shapes, values, and terminal semantics with Chex."""
        chex.assert_shape(self.transition, (..., None, None, None))
        chex.assert_axis_dimension_gt(self.transition, -3, 0)
        chex.assert_axis_dimension_gt(self.transition, -1, 0)
        chex.assert_axis_dimension(self.transition, -2, self.state_size)
        chex.assert_shape(
            self.reward,
            self.transition.shape,
            custom_message="reward shape must match transition",
        )

        state_shape = (*self.transition.shape[:-3], self.state_size)
        chex.assert_shape(
            self.initial,
            state_shape,
            custom_message="initial shape must match the MDP batch and state dimensions",
        )
        chex.assert_shape(
            self.terminal,
            state_shape,
            custom_message="terminal shape must match the MDP batch and state dimensions",
        )

        chex.assert_tree_all_finite(self, custom_message="MDP arrays must be finite")

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
        terminal_transition = (self.transition - identity) * self.terminal[..., None, None, :]
        chex.assert_trees_all_close(
            terminal_transition,
            jnp.zeros_like(terminal_transition),
            atol=_ATOL,
            rtol=0.0,
            custom_message="terminal states must be absorbing",
        )

        terminal_reward = self.reward * self.terminal[..., None, :, None]
        chex.assert_trees_all_close(
            terminal_reward,
            jnp.zeros_like(terminal_reward),
            atol=_ATOL,
            rtol=0.0,
            custom_message="rewards originating from terminal states must be zero",
        )
