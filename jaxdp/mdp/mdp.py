"""Finite tabular Markov decision processes."""

from dataclasses import fields

import jax
import jax.numpy as jnp
from chex import dataclass
from jax import core

_ATOL = 1e-5


@dataclass(frozen=True)
class MDP:
    """Array-only finite Markov decision process.

    The transition convention is ``transition[..., a, s_next, s]`` and the reward convention is
    ``reward[..., a, s, s_next]``. Every array uses the same leading batch shape.

    Attributes:
        transition: Column-stochastic transition arrays with shape ``(..., A, S, S)``.
        reward: Transition rewards with shape ``(..., A, S, S)``.
        initial: Initial-state distributions with shape ``(..., S)``.
        terminal: Terminal-state indicators with shape ``(..., S)``.
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
        """Validate shapes, values, and terminal semantics."""
        self._validate_shapes()
        if any(isinstance(array, core.Tracer) for array in jax.tree.leaves(self)):
            return

        self._validate_finite()
        self._validate_probabilities()
        self._validate_terminal()

    def _validate_shapes(self) -> None:
        if self.transition.ndim < 3:
            raise ValueError("transition shape must be (..., A, S, S)")

        if (
            self.action_size < 1
            or self.state_size < 1
            or self.transition.shape[-2] != self.state_size
        ):
            raise ValueError("transition shape must be (..., A, S, S) with positive A and S")

        if self.reward.shape != self.transition.shape:
            raise ValueError(f"reward shape must be {self.transition.shape}")

        state_shape = (*self.transition.shape[:-3], self.state_size)
        if self.initial.shape != state_shape:
            raise ValueError(f"initial shape must be {state_shape}")
        if self.terminal.shape != state_shape:
            raise ValueError(f"terminal shape must be {state_shape}")

    def _validate_finite(self) -> None:
        for field in fields(self):
            if not bool(jnp.all(jnp.isfinite(getattr(self, field.name)))):
                raise ValueError(f"{field.name} must contain only finite values")

    def _validate_probabilities(self) -> None:
        if bool(jnp.any(self.transition < 0)):
            raise ValueError("transition probabilities must be nonnegative")
        if not bool(jnp.allclose(self.transition.sum(axis=-2), 1.0, atol=_ATOL, rtol=0.0)):
            raise ValueError("transition matrix must be column stochastic")

        if bool(jnp.any(self.initial < 0)):
            raise ValueError("initial distribution must be nonnegative")
        if not bool(jnp.allclose(self.initial.sum(axis=-1), 1.0, atol=_ATOL, rtol=0.0)):
            raise ValueError("initial distribution must sum to one")

    def _validate_terminal(self) -> None:
        if not bool(jnp.all((self.terminal == 0) | (self.terminal == 1))):
            raise ValueError("terminal indicators must be zero or one")

        identity = jnp.eye(self.state_size, dtype=self.transition.dtype)
        terminal_transition = (self.transition - identity) * self.terminal[..., None, None, :]
        if not bool(jnp.allclose(terminal_transition, 0.0, atol=_ATOL, rtol=0.0)):
            raise ValueError("terminal states must be absorbing")

        terminal_reward = self.reward * self.terminal[..., None, :, None]
        if not bool(jnp.allclose(terminal_reward, 0.0, atol=_ATOL, rtol=0.0)):
            raise ValueError("rewards originating from terminal states must be zero")
