from dataclasses import is_dataclass

import jax
import jax.numpy as jnp
import pytest

from jaxdp.distribution import Occupancy, Stationary, eigenvalues
from jaxdp.mdp import Mdp


def _periodic_mdp() -> Mdp:
    return Mdp(
        transition=jnp.array([[[0.0, 1.0], [1.0, 0.0]]]),
        reward=jnp.zeros((1, 2, 2)),
        initial=jnp.array([1.0, 0.0]),
        terminal=jnp.zeros(2),
    )


def test_occupancy_propagates_from_the_initial_distribution() -> None:
    mdp = _periodic_mdp()
    policy = jnp.ones((1, 2))

    assert jnp.array_equal(Occupancy(steps=0).v(mdp, policy), jnp.array([1.0, 0.0]))
    assert jnp.array_equal(Occupancy(steps=1).v(mdp, policy), jnp.array([0.0, 1.0]))
    assert jnp.array_equal(Occupancy(steps=2).v(mdp, policy), jnp.array([1.0, 0.0]))
    assert jnp.array_equal(Occupancy(steps=1).q(mdp, policy), jnp.array([[0.0, 1.0]]))


def test_stationary_returns_an_invariant_distribution_for_a_periodic_chain() -> None:
    mdp = _periodic_mdp()
    policy = jnp.ones((1, 2))
    distribution = Stationary().v(mdp, policy)

    assert jnp.allclose(distribution, jnp.array([0.5, 0.5]))
    assert jnp.allclose(mdp.transition[0] @ distribution, distribution)
    assert jnp.allclose(Stationary().q(mdp, policy), policy * distribution)


def test_stationary_selects_the_minimum_norm_distribution_when_nonunique() -> None:
    mdp = Mdp(
        transition=jnp.eye(2)[None, ...],
        reward=jnp.zeros((1, 2, 2)),
        initial=jnp.array([1.0, 0.0]),
        terminal=jnp.zeros(2),
    )
    policy = jnp.ones((1, 2))

    assert jnp.allclose(Stationary().v(mdp, policy), jnp.array([0.5, 0.5]))


def test_distribution_components_compose_with_jit_and_vmap() -> None:
    mdp = _periodic_mdp()
    policies = jnp.ones((3, 1, 2))
    occupancy = Occupancy(steps=3)
    stationary = Stationary()

    finite = jax.jit(jax.vmap(lambda policy: occupancy.v(mdp, policy)))(policies)
    invariant = jax.jit(jax.vmap(lambda policy: stationary.v(mdp, policy)))(policies)

    assert is_dataclass(occupancy)
    assert is_dataclass(stationary)
    assert finite.shape == (3, 2)
    assert invariant.shape == (3, 2)


def test_eigenvalues_describe_the_policy_transition() -> None:
    values = eigenvalues(_periodic_mdp(), jnp.ones((1, 2)))

    assert jnp.allclose(jnp.sort(values.real), jnp.array([-1.0, 1.0]))
    assert jnp.allclose(values.imag, 0)


def test_occupancy_rejects_negative_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        Occupancy(steps=-1)
