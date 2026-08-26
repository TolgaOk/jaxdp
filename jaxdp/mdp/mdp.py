"""Finite tabular Markov decision processes."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import ClassVar, Self

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
from jax import core
from jax.typing import ArrayLike


def _as_real_array(value: ArrayLike, field: str) -> jax.Array:
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a numerical array") from error

    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise ValueError(f"{field} must be real-valued")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.float32)
    return array


@chex.dataclass(init=False, frozen=True, repr=False, mappable_dataclass=False)
class Mdp:
    """Immutable array-only finite Markov decision process.

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

    _ARRAY_NAMES: ClassVar[tuple[str, ...]] = (
        "transition",
        "reward",
        "initial",
        "terminal",
    )
    _ATOL: ClassVar[float] = 1e-5

    def __init__(
        self,
        transition: ArrayLike,
        reward: ArrayLike,
        initial: ArrayLike,
        terminal: ArrayLike,
        validate: bool = True,
    ) -> None:
        transition_array = _as_real_array(transition, "transition")
        reward_array = _as_real_array(reward, "reward")
        initial_array = _as_real_array(initial, "initial")
        terminal_array = _as_real_array(terminal, "terminal")

        if transition_array.ndim < 3:
            raise ValueError("transition shape must be (..., A, S, S)")

        object.__setattr__(self, "transition", transition_array)
        object.__setattr__(self, "reward", reward_array)
        object.__setattr__(self, "initial", initial_array)
        object.__setattr__(self, "terminal", terminal_array)

        self._validate_shapes()
        if validate:
            self.validate()

    def _validate_shapes(self) -> None:
        if self.transition.ndim < 3:
            raise ValueError("transition shape must be (..., A, S, S)")

        batch_shape = self.transition.shape[:-3]
        action_size, next_state_size, state_size = self.transition.shape[-3:]
        if action_size < 1 or state_size < 1 or next_state_size != state_size:
            raise ValueError("transition shape must be (..., A, S, S) with positive A and S")

        expected_dynamics = (*batch_shape, action_size, state_size, state_size)
        if self.reward.shape != expected_dynamics:
            raise ValueError(f"reward shape must be {expected_dynamics}")
        if self.initial.shape != (*batch_shape, state_size):
            raise ValueError(f"initial shape must be {(*batch_shape, state_size)}")
        if self.terminal.shape != (*batch_shape, state_size):
            raise ValueError(f"terminal shape must be {(*batch_shape, state_size)}")
    def validate(self) -> None:
        """Validate concrete probabilities, numerical values, and terminal-state semantics."""
        if any(
            isinstance(getattr(self, field), core.Tracer) for field in self._ARRAY_NAMES
        ):
            return

        for field in self._ARRAY_NAMES:
            if not bool(jnp.all(jnp.isfinite(getattr(self, field)))):
                raise ValueError(f"{field} must contain only finite values")

        if bool(jnp.any(self.transition < 0)):
            raise ValueError("transition probabilities must be nonnegative")
        if not bool(
            jnp.allclose(
                self.transition.sum(axis=-2),
                1.0,
                atol=self._ATOL,
                rtol=0.0,
            )
        ):
            raise ValueError("transition matrix must be column stochastic")

        if bool(jnp.any(self.initial < 0)):
            raise ValueError("initial distribution must be nonnegative")
        if not bool(
            jnp.allclose(
                self.initial.sum(axis=-1),
                1.0,
                atol=self._ATOL,
                rtol=0.0,
            )
        ):
            raise ValueError("initial distribution must sum to one")

        binary_terminal = jnp.logical_or(self.terminal == 0, self.terminal == 1)
        if not bool(jnp.all(binary_terminal)):
            raise ValueError("terminal indicators must be zero or one")

        identity = jnp.eye(self.state_size, dtype=self.transition.dtype)
        expected_transition = jnp.broadcast_to(identity, self.transition.shape)
        terminal_transition_error = (
            (self.transition - expected_transition) * self.terminal[..., None, None, :]
        )
        if not bool(jnp.allclose(terminal_transition_error, 0.0, atol=self._ATOL, rtol=0.0)):
            raise ValueError("terminal states must be absorbing")

        terminal_reward = self.reward * self.terminal[..., None, :, None]
        if not bool(jnp.allclose(terminal_reward, 0.0, atol=self._ATOL, rtol=0.0)):
            raise ValueError("rewards originating from terminal states must be zero")

    def init_state(self, key: chex.PRNGKey) -> jax.Array:
        """Sample one-hot initial states with shape ``(..., S)``."""
        state = jrd.categorical(key, jnp.log(self.initial), axis=-1)
        return jax.nn.one_hot(state, self.state_size, dtype=self.initial.dtype)

    @property
    def state_size(self) -> int:
        """Number of states."""
        return self.transition.shape[-1]

    @property
    def action_size(self) -> int:
        """Number of actions."""
        return self.transition.shape[-3]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Shared leading batch shape."""
        return self.transition.shape[:-3]

    @classmethod
    def array_names(cls) -> tuple[str, ...]:
        """Names of the arrays defining an MDP."""
        return cls._ARRAY_NAMES

    def __repr__(self) -> str:
        batch = f"batch_shape={self.batch_shape}, " if self.batch_shape else ""
        return f"jaxdp.Mdp({batch}state_size={self.state_size}, action_size={self.action_size})"

    @classmethod
    def load_mdp_from_json(cls, file_path: str | PathLike[str]) -> Self:
        """Load an MDP from a JSON file."""
        with Path(file_path).open(encoding="utf-8") as file:
            data = json.load(file)

        data.pop("name", None)
        required = cls._ARRAY_NAMES
        missing = [field for field in required if field not in data]
        if missing:
            raise ValueError(f"missing MDP arrays: {', '.join(missing)}")

        known = set(cls._ARRAY_NAMES)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown MDP fields: {', '.join(unknown)}")

        return cls(**{field: data[field] for field in required})

    def save_mdp_as_json(self, file_path: str | PathLike[str]) -> None:
        """Save every defining MDP array to a JSON file."""
        data = {field: getattr(self, field).tolist() for field in self._ARRAY_NAMES}
        with Path(file_path).open("w", encoding="utf-8") as file:
            json.dump(data, file)


MDP = Mdp


__all__ = ["Mdp", "MDP"]
