import jax.numpy as jnp
import pytest

from jaxdp.mdp import MDP, sequential_mdp


def test_sequential_mdp_arrays() -> None:
    mdp = sequential_mdp(4)

    assert isinstance(mdp, MDP)
    assert mdp.transition.shape == (2, 4, 4)
    assert mdp.reward.shape == mdp.transition.shape
    assert jnp.allclose(mdp.transition.sum(axis=-2), 1.0)
    assert jnp.allclose(mdp.reward[0, 2], 1.0)
    assert jnp.allclose(mdp.reward[1], 0.0)


def test_sequential_mdp_handles_one_state() -> None:
    mdp = sequential_mdp(1)

    assert jnp.array_equal(mdp.transition, jnp.ones((2, 1, 1)))
    assert jnp.array_equal(mdp.reward, jnp.zeros((2, 1, 1)))


def test_sequential_mdp_rejects_empty_state_space() -> None:
    with pytest.raises(AssertionError, match="state_size"):
        sequential_mdp(0)
