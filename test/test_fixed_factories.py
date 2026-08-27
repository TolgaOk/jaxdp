import jax.numpy as jnp
import pytest

from jaxdp.mdp.forest_mdp import forest_mdp
from jaxdp.mdp.simple_graph import graph_mdp
from jaxdp.mdp.tree_mdp import tree_mdp


def test_forest_waits_grows_and_harvest_resets() -> None:
    mdp = forest_mdp(rotation=2)

    assert jnp.array_equal(
        mdp.transition[0],
        jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
    )
    assert jnp.array_equal(mdp.transition[1, 0], jnp.ones(3))
    assert jnp.array_equal(mdp.reward[1, :, 0], jnp.arange(3))
    assert jnp.array_equal(mdp.initial, jnp.array([1.0, 0.0, 0.0]))


def test_forest_supports_zero_rotation_and_rejects_negative_rotation() -> None:
    mdp = forest_mdp(rotation=0)

    assert mdp.transition.shape == (2, 1, 1)
    with pytest.raises(AssertionError, match="rotation"):
        forest_mdp(rotation=-1)


def test_tree_has_absorbing_leaves_and_outer_leaf_rewards() -> None:
    mdp = tree_mdp(depth=2)

    assert mdp.state_size == 7
    assert jnp.array_equal(mdp.terminal, jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]))
    assert mdp.transition[0, 1, 0] == 1
    assert mdp.transition[1, 2, 0] == 1
    assert jnp.all(mdp.transition[:, 3, 3] == 1)
    assert mdp.reward[0, 1, 3] == 1
    assert mdp.reward[1, 2, 6] == 0.5


def test_tree_rejects_nonpositive_depth() -> None:
    with pytest.raises(AssertionError, match="depth"):
        tree_mdp(depth=0)


def test_graph_matches_its_fixed_transition_and_reward_rules() -> None:
    mdp = graph_mdp()

    assert mdp.transition.shape == (6, 6, 6)
    assert jnp.allclose(mdp.transition[0, :, 0], jnp.array([0.8, 0, 0, 0, 0.2, 0]))
    assert jnp.allclose(mdp.transition[4, :, 0], jnp.array([0.2, 0, 0, 0, 0.8, 0]))
    assert jnp.array_equal(mdp.transition[1, :, 0], mdp.transition[0, :, 0])
    assert mdp.reward[5, 0, 3] == 1
    assert mdp.reward[4, 3, 2] == -1
    assert mdp.reward[0, 1, 4] == -0.05
