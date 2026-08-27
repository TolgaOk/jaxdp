from dataclasses import FrozenInstanceError, fields, is_dataclass

import chex
import jax
import jax.numpy as jnp
import pytest

import jaxdp
from jaxdp.mdp import MDP, MRP, make_mrp


def test_mrp_exposes_canonical_shapes_and_types() -> None:
    mrp = _mrp()

    assert jaxdp.MRP is MRP
    assert is_dataclass(mrp)
    assert tuple(field.name for field in fields(mrp)) == (
        "transition",
        "reward",
        "initial",
        "terminal",
    )
    assert mrp.transition.shape == (2, 2)
    assert mrp.reward.shape == (2,)
    assert mrp.initial.shape == (2,)
    assert mrp.terminal.shape == (2,)
    assert mrp.state_size == 2


def test_mrp_is_an_immutable_jax_pytree() -> None:
    mrp = _mrp()
    copied = jax.jit(lambda model: model)(mrp)
    stacked = jax.tree.map(lambda *models: jnp.stack(models), mrp, mrp)
    stacked.validate()

    assert len(jax.tree.leaves(mrp)) == 4
    assert jax.tree.all(jax.tree.map(jnp.array_equal, copied, mrp))
    assert stacked.transition.shape == (2, 2, 2)
    attribute = "transition"
    with pytest.raises(FrozenInstanceError):
        setattr(mrp, attribute, jnp.zeros_like(mrp.transition))


def test_make_mrp_fixes_the_policy() -> None:
    mdp = _mdp()
    mrp = make_mrp(mdp, _policy())

    assert jnp.allclose(mrp.transition, jnp.array([[0.25, 0.25], [0.75, 0.75]]))
    assert jnp.allclose(mrp.reward, jnp.array([2.5, 2.5]))
    assert jnp.array_equal(mrp.initial, mdp.initial)
    assert jnp.array_equal(mrp.terminal, mdp.terminal)


def test_make_mrp_supports_direct_batches_jit_and_vmap() -> None:
    mdp = _mdp()
    mdps = jax.tree.map(lambda *models: jnp.stack(models), mdp, mdp)
    policies = jnp.stack((_policy(), _policy()))
    convert = chex.chexify(jax.jit(jax.vmap(make_mrp)), async_check=False)

    direct = make_mrp(mdps, policies)
    mapped = convert(mdps, policies)

    assert direct.transition.shape == (2, 2, 2)
    assert jnp.allclose(direct.reward, jnp.full((2, 2), 2.5))
    assert jax.tree.all(jax.tree.map(jnp.array_equal, direct, mapped))


def test_mrp_rejects_inconsistent_shapes_and_values() -> None:
    transition, reward, initial, terminal = _mrp_arrays()

    with pytest.raises(AssertionError, match="transition"):
        _validate(transition[:, :1], reward, initial, terminal)
    with pytest.raises(AssertionError, match="reward shape"):
        _validate(transition, reward[:1], initial, terminal)
    with pytest.raises(AssertionError, match="column stochastic"):
        _validate(transition.at[0, 0].set(0.5), reward, initial, terminal)
    with pytest.raises(AssertionError, match="finite"):
        _validate(transition, reward.at[0].set(jnp.nan), initial, terminal)


def test_mrp_enforces_terminal_state_semantics() -> None:
    transition, reward, initial, terminal = _mrp_arrays()

    with pytest.raises(AssertionError, match="absorbing"):
        _validate(jnp.array([[1.0, 1.0], [0.0, 0.0]]), reward, initial, terminal)
    with pytest.raises(AssertionError, match="originating from terminal"):
        _validate(transition, reward.at[1].set(1.0), initial, terminal)


@pytest.mark.parametrize(
    "policy",
    [
        jnp.ones((1, 2)),
        jnp.array([[1.1, 0.0], [-0.1, 1.0]]),
        jnp.full((2, 2), 0.4),
        jnp.array([[jnp.nan, 0.0], [0.0, 1.0]]),
    ],
)
def test_make_mrp_rejects_invalid_policy(policy: jax.Array) -> None:
    with pytest.raises(AssertionError, match="policy"):
        make_mrp(_mdp(), policy)


def _mrp_arrays() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    return (
        jnp.eye(2),
        jnp.array([1.0, 0.0]),
        jnp.array([1.0, 0.0]),
        jnp.array([0.0, 1.0]),
    )


def _mrp() -> MRP:
    transition, reward, initial, terminal = _mrp_arrays()
    mrp = MRP(
        transition=transition,
        reward=reward,
        initial=initial,
        terminal=terminal,
    )
    mrp.validate()
    return mrp


def _mdp() -> MDP:
    transition = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    reward = jnp.array(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[0.0, 3.0], [4.0, 0.0]],
        ]
    )
    mdp = MDP(
        transition=transition,
        reward=reward,
        initial=jnp.array([1.0, 0.0]),
        terminal=jnp.zeros(2),
    )
    mdp.validate()
    return mdp


def _policy() -> jax.Array:
    return jnp.array([[0.25, 0.75], [0.75, 0.25]])


def _validate(
    transition: jax.Array,
    reward: jax.Array,
    initial: jax.Array,
    terminal: jax.Array,
) -> None:
    MRP(
        transition=transition,
        reward=reward,
        initial=initial,
        terminal=terminal,
    ).validate()
