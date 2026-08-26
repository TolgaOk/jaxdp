from dataclasses import FrozenInstanceError, fields, is_dataclass

import jax
import jax.numpy as jnp
import pytest

from jaxdp.mdp import MDP, Mdp


def _arrays() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    transition = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]
    )
    reward = jnp.zeros((2, 2, 2)).at[1, 0, 1].set(2.0)
    initial = jnp.array([1.0, 0.0])
    terminal = jnp.array([0, 1])
    return transition, reward, initial, terminal


def _mdp() -> Mdp:
    return Mdp(*_arrays())


def _initial_mass(mdp: Mdp) -> jax.Array:
    return jnp.sum(mdp.initial)


def test_mdp_exposes_canonical_shapes_and_types() -> None:
    mdp = _mdp()

    assert is_dataclass(mdp)
    assert MDP is Mdp
    assert tuple(field.name for field in fields(mdp)) == Mdp.array_names()
    assert mdp.transition.shape == (2, 2, 2)
    assert mdp.reward.shape == (2, 2, 2)
    assert mdp.initial.shape == (2,)
    assert mdp.terminal.shape == (2,)
    assert mdp.state_size == 2
    assert mdp.action_size == 2
    assert mdp.batch_shape == ()
    assert jnp.issubdtype(mdp.terminal.dtype, jnp.floating)
    assert jnp.allclose(_initial_mass(mdp), 1.0)


def test_mdp_supports_direct_batches() -> None:
    transition, reward, initial, terminal = _arrays()
    mdp = Mdp(
        jnp.stack((transition, transition)),
        jnp.stack((reward, reward)),
        jnp.stack((initial, initial)),
        jnp.stack((terminal, terminal)),
    )

    assert mdp.batch_shape == (2,)


def test_mdp_is_an_immutable_jax_pytree() -> None:
    mdp = _mdp()
    leaves = jax.tree.leaves(mdp)
    copied = jax.jit(lambda item: item)(mdp)
    stacked = jax.tree.map(lambda *items: jnp.stack(items), mdp, mdp)

    assert len(leaves) == 4
    assert jnp.allclose(copied.transition, mdp.transition)
    assert stacked.batch_shape == (2,)
    assert jnp.allclose(jax.vmap(_initial_mass)(stacked), jnp.ones(2))
    attribute = "transition"
    with pytest.raises(FrozenInstanceError):
        setattr(mdp, attribute, jnp.zeros_like(mdp.transition))


def test_mdp_samples_initial_states_under_jit() -> None:
    mdp = _mdp()
    state = jax.jit(mdp.init_state)(jax.random.key(0))

    assert state.shape == (mdp.state_size,)
    assert jnp.array_equal(state, mdp.initial)


def test_mdp_rejects_inconsistent_shapes() -> None:
    transition, reward, initial, terminal = _arrays()

    with pytest.raises(ValueError, match="reward shape"):
        Mdp(transition, reward[:, :, :1], initial, terminal)
    with pytest.raises(ValueError, match="initial shape"):
        Mdp(transition, reward, initial[:1], terminal)
    with pytest.raises(ValueError, match="terminal shape"):
        Mdp(transition, reward, initial, terminal[:1])


def test_mdp_rejects_invalid_probabilities_and_values() -> None:
    transition, reward, initial, terminal = _arrays()
    negative_transition = transition.at[0, 0, 0].set(1.1).at[0, 1, 0].set(-0.1)

    with pytest.raises(ValueError, match="nonnegative"):
        Mdp(negative_transition, reward, initial, terminal)
    with pytest.raises(ValueError, match="column stochastic"):
        Mdp(transition.at[0, 0, 0].set(0.9), reward, initial, terminal)
    with pytest.raises(ValueError, match="initial distribution must be nonnegative"):
        Mdp(transition, reward, jnp.array([1.1, -0.1]), terminal)
    with pytest.raises(ValueError, match="finite"):
        Mdp(transition, reward.at[0, 0, 0].set(jnp.nan), initial, terminal)
    with pytest.raises(ValueError, match="zero or one"):
        Mdp(transition, reward, initial, jnp.array([0.0, 0.5]))


def test_mdp_enforces_terminal_state_semantics() -> None:
    transition, reward, initial, terminal = _arrays()
    nonabsorbing = transition.at[0, :, 1].set(jnp.array([1.0, 0.0]))
    terminal_reward = reward.at[0, 1, 1].set(1.0)

    with pytest.raises(ValueError, match="absorbing"):
        Mdp(nonabsorbing, reward, initial, terminal)
    with pytest.raises(ValueError, match="originating from terminal"):
        Mdp(transition, terminal_reward, initial, terminal)


def test_mdp_json_round_trip_preserves_all_arrays(tmp_path) -> None:
    mdp = _mdp()
    path = tmp_path / "mdp.json"

    mdp.save_mdp_as_json(path)
    restored = Mdp.load_mdp_from_json(path)

    assert jax.tree.all(jax.tree.map(jnp.array_equal, restored, mdp))
