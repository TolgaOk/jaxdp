from dataclasses import FrozenInstanceError, fields, is_dataclass

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxdp.mdp import MDP


def _arrays() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    transition = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]
    )
    reward = jnp.zeros((2, 2, 2)).at[1, 0, 1].set(2.0)
    initial = jnp.array([1.0, 0.0])
    terminal = jnp.array([0.0, 1.0])
    return transition, reward, initial, terminal


def _mdp() -> MDP:
    transition, reward, initial, terminal = _arrays()
    mdp = MDP(
        transition=transition,
        reward=reward,
        initial=initial,
        terminal=terminal,
    )
    mdp.validate()
    return mdp


def _validate(
    transition: jax.Array,
    reward: jax.Array,
    initial: jax.Array,
    terminal: jax.Array,
) -> None:
    MDP(
        transition=transition,
        reward=reward,
        initial=initial,
        terminal=terminal,
    ).validate()


def _initial_mass(mdp: MDP) -> jax.Array:
    return jnp.sum(mdp.initial)


def _validated_initial(mdp: MDP) -> jax.Array:
    mdp.validate()
    return mdp.initial


def test_mdp_exposes_canonical_shapes_and_types() -> None:
    mdp = _mdp()

    assert is_dataclass(mdp)
    assert tuple(field.name for field in fields(mdp)) == (
        "transition",
        "reward",
        "initial",
        "terminal",
    )
    assert mdp.transition.shape == (2, 2, 2)
    assert mdp.reward.shape == (2, 2, 2)
    assert mdp.initial.shape == (2,)
    assert mdp.terminal.shape == (2,)
    assert mdp.state_size == 2
    assert mdp.action_size == 2
    assert jnp.issubdtype(mdp.terminal.dtype, jnp.floating)
    assert jnp.allclose(_initial_mass(mdp), 1.0)


def test_mdp_supports_direct_batches() -> None:
    transition, reward, initial, terminal = _arrays()
    mdp = MDP(
        transition=jnp.stack((transition, transition)),
        reward=jnp.stack((reward, reward)),
        initial=jnp.stack((initial, initial)),
        terminal=jnp.stack((terminal, terminal)),
    )
    mdp.validate()

    assert mdp.transition.shape[:-3] == (2,)


def test_mdp_is_an_immutable_jax_pytree() -> None:
    mdp = _mdp()
    leaves = jax.tree.leaves(mdp)
    copied = jax.jit(lambda item: item)(mdp)
    stacked = jax.tree.map(lambda *items: jnp.stack(items), mdp, mdp)

    assert len(leaves) == 4
    assert jnp.allclose(copied.transition, mdp.transition)
    assert stacked.transition.shape[:-3] == (2,)
    assert jnp.allclose(jax.vmap(_initial_mass)(stacked), jnp.ones(2))
    attribute = "transition"
    with pytest.raises(FrozenInstanceError):
        setattr(mdp, attribute, jnp.zeros_like(mdp.transition))


def test_mdp_rejects_inconsistent_shapes() -> None:
    transition, reward, initial, terminal = _arrays()

    with pytest.raises(AssertionError, match="reward shape"):
        _validate(transition, reward[:, :, :1], initial, terminal)
    with pytest.raises(AssertionError, match="initial shape"):
        _validate(transition, reward, initial[:1], terminal)
    with pytest.raises(AssertionError, match="terminal shape"):
        _validate(transition, reward, initial, terminal[:1])


def test_mdp_rejects_invalid_probabilities_and_values() -> None:
    transition, reward, initial, terminal = _arrays()
    negative_transition = transition.at[0, 0, 0].set(1.1).at[0, 1, 0].set(-0.1)

    with pytest.raises(AssertionError, match="nonnegative"):
        _validate(negative_transition, reward, initial, terminal)
    with pytest.raises(AssertionError, match="column stochastic"):
        _validate(transition.at[0, 0, 0].set(0.9), reward, initial, terminal)
    with pytest.raises(AssertionError, match="initial probabilities must be nonnegative"):
        _validate(transition, reward, jnp.array([1.1, -0.1]), terminal)
    with pytest.raises(AssertionError, match="finite"):
        _validate(transition, reward.at[0, 0, 0].set(jnp.nan), initial, terminal)
    with pytest.raises(AssertionError, match="zero or one"):
        _validate(transition, reward, initial, jnp.array([0.0, 0.5]))


def test_mdp_enforces_terminal_state_semantics() -> None:
    transition, reward, initial, terminal = _arrays()
    nonabsorbing = transition.at[0, :, 1].set(jnp.array([1.0, 0.0]))
    terminal_reward = reward.at[0, 1, 1].set(1.0)

    with pytest.raises(AssertionError, match="absorbing"):
        _validate(nonabsorbing, reward, initial, terminal)
    with pytest.raises(AssertionError, match="originating from terminal"):
        _validate(transition, terminal_reward, initial, terminal)


def test_mdp_validation_composes_with_jit_and_vmap() -> None:
    mdp = _mdp()
    batch = jax.tree.map(lambda value: jnp.stack((value, value)), mdp)
    validate = chex.chexify(jax.jit(jax.vmap(_validated_initial)), async_check=False)

    assert jnp.array_equal(validate(batch), batch.initial)

    invalid = batch.replace(initial=batch.initial.at[1].set(jnp.array([1.1, -0.1])))
    with pytest.raises(AssertionError, match="initial probabilities must be nonnegative"):
        validate(invalid)
