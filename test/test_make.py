import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import pytest

import jaxdp.mdp as mdp_module
from jaxdp import MDP, make
from jaxdp.planning import policy_eval


def test_mdp_package_exports_only_models_and_construction() -> None:
    assert mdp_module.__all__ == ["MDP", "MRP", "make", "make_mrp"]


@pytest.mark.parametrize(
    ("name", "state_size", "action_size", "terminal_size"),
    [
        ("cliff-walking", 38, 4, 1),
        ("delayed-reward", 7, 2, 4),
        ("delayed-reward-long", 63, 2, 32),
        ("delayed-reward-long-noisy", 63, 2, 32),
        ("delayed-reward-noisy", 7, 2, 4),
        ("forest", 3, 2, 0),
        ("forest-long", 11, 2, 0),
        ("four-rooms", 104, 4, 1),
        ("frozen-lake", 16, 4, 5),
        ("frozen-lake-deterministic", 16, 4, 5),
        ("garnet", 10, 4, 0),
        ("garnet-dense", 50, 5, 0),
        ("garnet-large", 300, 10, 0),
        ("garnet-medium", 50, 5, 0),
        ("graph", 6, 6, 0),
        ("grid-world", 8, 4, 1),
        ("grid-world-slippery", 8, 4, 1),
        ("sequential", 4, 2, 0),
        ("sequential-long", 20, 2, 0),
        ("tree", 7, 2, 4),
        ("tree-deep", 63, 2, 32),
    ],
)
def test_make_constructs_valid_solvable_recipes(
    name: str,
    state_size: int,
    action_size: int,
    terminal_size: int,
) -> None:
    mdp = make(name)
    mdp.validate()

    assert mdp.state_size == state_size
    assert mdp.action_size == action_size
    assert int(jnp.sum(mdp.terminal)) == terminal_size

    policy = jnp.full(
        (action_size, state_size),
        1 / action_size,
        dtype=mdp.transition.dtype,
    )
    v_val = policy_eval.v(mdp, policy, gamma=0.995)
    q_val = policy_eval.q(mdp, policy, gamma=0.995)
    assert jnp.all(jnp.isfinite(v_val))
    assert jnp.all(jnp.isfinite(q_val))
    assert jnp.allclose(v_val, jnp.sum(policy * q_val, axis=0), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "name",
    [
        "cliff-walking",
        "delayed-reward",
        "delayed-reward-long",
        "delayed-reward-long-noisy",
        "delayed-reward-noisy",
        "four-rooms",
        "frozen-lake",
        "frozen-lake-deterministic",
        "grid-world",
        "grid-world-slippery",
        "tree",
        "tree-deep",
    ],
)
def test_terminal_recipes_absorb_under_uniform_policy(name: str) -> None:
    mdp = make(name)
    transition = np.asarray(mdp.transition, dtype=np.float64).mean(axis=0)
    initial = np.asarray(mdp.initial, dtype=np.float64)
    terminal = np.asarray(mdp.terminal, dtype=bool)
    nonterminal_index = np.flatnonzero(~terminal)
    terminal_index = np.flatnonzero(terminal)
    transient = transition[np.ix_(nonterminal_index, nonterminal_index)]
    direct = transition[np.ix_(terminal_index, nonterminal_index)].sum(axis=0)
    system = np.eye(len(nonterminal_index)) - transient.T

    absorption = np.linalg.solve(system, direct)
    duration = np.linalg.solve(system, np.ones(len(nonterminal_index)))
    initial_absorption = initial[terminal].sum() + initial[~terminal] @ absorption
    initial_duration = initial[~terminal] @ duration

    assert np.isclose(initial_absorption, 1.0, atol=1e-5)
    assert np.isfinite(initial_duration)
    assert initial_duration > 0


def test_frozen_lake_recipes_use_terminal_holes_and_canonical_slip() -> None:
    slippery = make("frozen-lake")
    deterministic = make("frozen-lake-deterministic")

    assert jnp.array_equal(
        slippery.terminal,
        jnp.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]),
    )
    expected = jnp.zeros(16).at[jnp.array([0, 1, 4])].set(1 / 3)
    assert jnp.allclose(slippery.transition[1, :, 0], expected)
    assert deterministic.transition[1, 1, 0] == 1
    assert jnp.sum(deterministic.transition[1, :, 0]) == 1


@pytest.mark.parametrize(
    "name",
    [
        "delayed-reward",
        "delayed-reward-long",
        "delayed-reward-long-noisy",
        "delayed-reward-noisy",
        "garnet",
        "garnet-dense",
        "garnet-large",
        "garnet-medium",
    ],
)
def test_seeded_recipes_are_reproducible(name: str) -> None:
    first = make(name)
    second = make(name)

    chex.assert_trees_all_equal(first, second, make(name, key=None), make(name, key=jrd.key(42)))
    chex.assert_trees_all_equal(make(name, key=jrd.key(7)), make(name, key=jrd.key(7)))


@pytest.mark.parametrize(
    "name",
    [
        "delayed-reward-long-noisy",
        "delayed-reward-noisy",
        "garnet",
        "garnet-dense",
        "garnet-large",
        "garnet-medium",
    ],
)
def test_different_keys_generate_different_models(name: str) -> None:
    first = make(name, key=jrd.key(7))
    second = make(name, key=jrd.key(8))

    first.validate()
    second.validate()
    assert not jnp.array_equal(first.reward, second.reward)


@pytest.mark.parametrize(
    "name",
    [
        "cliff-walking",
        "delayed-reward",
        "delayed-reward-long",
        "forest",
        "forest-long",
        "four-rooms",
        "frozen-lake",
        "frozen-lake-deterministic",
        "graph",
        "grid-world",
        "grid-world-slippery",
        "sequential",
        "sequential-long",
        "tree",
        "tree-deep",
    ],
)
def test_fixed_models_are_unchanged_by_keys(name: str) -> None:
    chex.assert_trees_all_equal(make(name), make(name, key=jrd.key(7)), make(name, key=jrd.key(8)))


@pytest.mark.parametrize("name", ["garnet", "delayed-reward-noisy", "frozen-lake"])
def test_make_composes_with_jit_and_vmap(name: str) -> None:
    def create_one(key: jax.Array | None = None) -> MDP:
        return make(name, key=key)

    create = chex.chexify(jax.jit(create_one), async_check=False)
    create_batch = chex.chexify(jax.jit(jax.vmap(create_one)), async_check=False)
    keys = jrd.split(jrd.key(7), 2)
    expected = jax.tree.map(
        lambda *arrays: jnp.stack(arrays), make(name, key=keys[0]), make(name, key=keys[1])
    )

    chex.assert_trees_all_close(create(), make(name), rtol=1e-5, atol=1e-6)
    chex.assert_trees_all_close(create(None), make(name), rtol=1e-5, atol=1e-6)
    chex.assert_trees_all_close(create(keys[0]), make(name, key=keys[0]), rtol=1e-5, atol=1e-6)
    chex.assert_trees_all_close(create_batch(keys), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", ["garnet", "delayed-reward-noisy"])
def test_make_accepts_legacy_keys(name: str) -> None:
    chex.assert_trees_all_equal(make(name, key=jrd.PRNGKey(7)), make(name, key=jrd.key(7)))


def test_make_rejects_unknown_recipe() -> None:
    with pytest.raises(ValueError, match="unknown MDP"):
        make("unknown")
