from dataclasses import FrozenInstanceError, is_dataclass

import chex
import jax
import jax.numpy as jnp
import pytest

import jaxdp
from jaxdp import mapping
from jaxdp.mapping import (
    EpsilonGreedy,
    GreedyMap,
    MellowMax,
    Occupancy,
    ProjSimplex,
    Reward,
    SoftGreedyMap,
    Stationary,
    eigenvalues,
)
from jaxdp.mdp import MDP
from jaxdp.operator import TransOp


def _two_state_mdp() -> MDP:
    transition = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    reward = (
        jnp.zeros((2, 2, 2))
        .at[0, 1, 1]
        .set(1.0)
        .at[1, 0, 1]
        .set(2.0)
        .at[1, 1, 0]
        .set(3.0)
    )
    initial = jnp.array([1.0, 0.0])
    terminal = jnp.zeros(2)
    return MDP(
        transition=transition,
        reward=reward,
        initial=initial,
        terminal=terminal,
    )


def _periodic_mdp() -> MDP:
    return MDP(
        transition=jnp.array([[[0.0, 1.0], [1.0, 0.0]]]),
        reward=jnp.zeros((1, 2, 2)),
        initial=jnp.array([1.0, 0.0]),
        terminal=jnp.zeros(2),
    )


def test_public_mapping_names() -> None:
    assert jaxdp.mapping is mapping
    assert jaxdp.GreedyMap is GreedyMap
    assert jaxdp.SoftGreedyMap is SoftGreedyMap
    assert jaxdp.ProjSimplex is ProjSimplex
    assert jaxdp.MellowMax is MellowMax
    assert jaxdp.Reward is Reward
    assert jaxdp.Occupancy is Occupancy
    assert jaxdp.Stationary is Stationary
    assert jaxdp.eigenvalues is eigenvalues
    assert not hasattr(jaxdp, "distribution")
    assert not hasattr(jaxdp, "policy")
    assert not hasattr(jaxdp, "Greedy")
    assert not hasattr(jaxdp, "Soft")


def test_mapping_components_share_q_v_api() -> None:
    mdp = _two_state_mdp()
    v_val = jnp.array([4.0, 8.0])
    reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
    q_val = reward + 0.5 * TransOp().sa(mdp, v_val)
    mappings = (
        GreedyMap(),
        SoftGreedyMap(temperature=2.0),
        EpsilonGreedy(epsilon=0.2),
    )

    for value_map in mappings:
        assert jnp.allclose(value_map.v(mdp, v_val, gamma=0.5), value_map.q(q_val))


def test_mapping_components_match_their_definitions() -> None:
    q_val = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    greedy = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    assert jnp.allclose(GreedyMap().q(q_val), greedy)
    assert jnp.allclose(
        SoftGreedyMap(temperature=2.0).q(q_val),
        jax.nn.softmax(q_val / 2.0, axis=0),
    )
    assert jnp.allclose(EpsilonGreedy(epsilon=0.2).q(q_val), 0.8 * greedy + 0.1)


@pytest.mark.parametrize("temperature", [0.0, -1.0, jnp.inf, jnp.nan])
def test_soft_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(AssertionError, match="temperature"):
        SoftGreedyMap(temperature=temperature).q(jnp.zeros((2, 2)))


@pytest.mark.parametrize("epsilon", [-0.1, 1.1, jnp.inf, jnp.nan])
def test_epsilon_greedy_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(AssertionError, match="epsilon"):
        EpsilonGreedy(epsilon=epsilon).q(jnp.zeros((2, 2)))


def test_mapping_components_are_immutable_dataclasses() -> None:
    value_map = SoftGreedyMap(temperature=2.0)

    assert is_dataclass(value_map)
    attribute = "temperature"
    with pytest.raises(FrozenInstanceError):
        setattr(value_map, attribute, 1.0)


def test_mapping_components_support_jit_and_vmap() -> None:
    value_map = SoftGreedyMap(temperature=2.0)
    values = jnp.array(
        [
            [[3.0, 1.0], [1.0, 2.0]],
            [[1.0, 3.0], [2.0, 1.0]],
        ]
    )

    result = chex.chexify(
        jax.jit(jax.vmap(value_map.q)),
        async_check=False,
    )(values)

    assert result.shape == values.shape
    assert jnp.allclose(result.sum(axis=1), 1.0)


def test_simplex_projection_matches_euclidean_projection() -> None:
    q_val = jnp.array(
        [
            [0.2, 2.0, 0.8],
            [0.2, 0.0, 0.6],
            [0.2, 0.0, -0.5],
        ]
    )
    expected = jnp.array(
        [
            [1 / 3, 1.0, 0.6],
            [1 / 3, 0.0, 0.4],
            [1 / 3, 0.0, 0.0],
        ]
    )
    projection = ProjSimplex()
    policy = projection.q(q_val)

    assert is_dataclass(projection)
    assert jnp.allclose(policy, expected)
    assert jnp.allclose(jnp.sum(policy, axis=0), 1)
    assert jnp.all(policy >= 0)
    assert jnp.allclose(projection.q(policy), policy)


def test_simplex_projection_composes_with_jit_and_vmap() -> None:
    q_vals = jnp.array(
        [
            [[0.2, 2.0], [0.2, 0.0], [0.2, 0.0]],
            [[2.0, 0.2], [0.0, 0.2], [0.0, 0.2]],
        ]
    )
    policies = jax.jit(jax.vmap(ProjSimplex().q))(q_vals)

    assert policies.shape == q_vals.shape
    assert jnp.allclose(jnp.sum(policies, axis=1), 1)
    assert jnp.all(policies >= 0)


def test_mellowmax_matches_normalized_log_mean_exp() -> None:
    q_val = jnp.array([[2.0, -1.0], [0.0, 3.0], [1.0, 2.0]])
    temperature = 0.75
    reduction = MellowMax(temperature=temperature)
    expected = temperature * (
        jax.nn.logsumexp(q_val / temperature, axis=0) - jnp.log(q_val.shape[0])
    )

    assert is_dataclass(reduction)
    assert jnp.allclose(reduction.q(q_val), expected)
    assert jnp.allclose(reduction.q(jnp.full((3, 2), 4.0)), 4.0)


@pytest.mark.parametrize("temperature", [0.0, -1.0, jnp.inf, jnp.nan])
def test_mellowmax_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(AssertionError, match="temperature"):
        MellowMax(temperature=temperature).q(jnp.zeros((2, 2)))


def test_mellowmax_composes_with_jit_and_vmap() -> None:
    q_vals = jnp.array(
        [
            [[2.0, -1.0], [0.0, 3.0], [1.0, 2.0]],
            [[-1.0, 2.0], [3.0, 0.0], [2.0, 1.0]],
        ]
    )
    apply = chex.chexify(
        jax.jit(jax.vmap(MellowMax(temperature=0.75).q)),
        async_check=False,
    )
    v_vals = apply(q_vals)

    assert v_vals.shape == (2, 2)
    assert jnp.all(jnp.isfinite(v_vals))


def test_reward_maps_match_transition_expectations() -> None:
    mdp = _two_state_mdp()
    policy = jnp.array([[0.25, 0.75], [0.75, 0.25]])
    reward = Reward()

    assert is_dataclass(reward)
    assert jnp.allclose(reward.sa(mdp), jnp.array([[0.0, 1.0], [2.0, 3.0]]))
    assert jnp.allclose(reward.s(mdp, policy), jnp.array([1.5, 1.5]))


def test_reward_maps_compose_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    policies = jnp.array(
        [
            [[0.25, 0.75], [0.75, 0.25]],
            [[0.75, 0.25], [0.25, 0.75]],
        ]
    )
    reward = Reward()
    apply = chex.chexify(
        jax.jit(jax.vmap(lambda policy: reward.s(mdp, policy))),
        async_check=False,
    )

    assert jax.jit(reward.sa)(mdp).shape == (2, 2)
    assert apply(policies).shape == (2, 2)


def test_occupancy_propagates_from_the_initial_distribution() -> None:
    mdp = _periodic_mdp()
    policy = jnp.ones((1, 2))

    assert jnp.array_equal(Occupancy(step=0).v(mdp, policy), jnp.array([1.0, 0.0]))
    assert jnp.array_equal(Occupancy().v(mdp, policy), jnp.array([0.0, 1.0]))
    assert jnp.array_equal(Occupancy(step=2).v(mdp, policy), jnp.array([1.0, 0.0]))
    assert jnp.array_equal(Occupancy().q(mdp, policy), jnp.array([[0.0, 1.0]]))


def test_discounted_occupancy_is_normalized_and_converges() -> None:
    mdp = _periodic_mdp()
    policy = jnp.ones((1, 2))
    gamma = 0.5
    p_s = mdp.transition[0]
    expected = (1 - gamma) * jnp.linalg.solve(
        jnp.eye(mdp.state_size) - gamma * p_s,
        mdp.initial,
    )

    assert jnp.allclose(
        Occupancy(step=2).v(mdp, policy, gamma),
        jnp.array([0.75, 0.25]),
    )
    assert jnp.allclose(Occupancy(step=40).v(mdp, policy, gamma), expected)
    assert jnp.allclose(jnp.sum(Occupancy(step=40).q(mdp, policy, gamma)), 1)


def test_discounted_occupancy_composes_with_jit_and_vmap() -> None:
    mdp = _periodic_mdp()
    policy = jnp.ones((1, 2))
    apply = chex.chexify(
        jax.jit(jax.vmap(lambda gamma: Occupancy(step=4).v(mdp, policy, gamma))),
        async_check=False,
    )
    occupancies = apply(jnp.array([0.0, 0.5, 1.0]))

    assert occupancies.shape == (3, 2)
    assert jnp.allclose(jnp.sum(occupancies, axis=1), 1)


def test_stationary_returns_an_invariant_distribution_for_a_periodic_chain() -> None:
    mdp = _periodic_mdp()
    policy = jnp.ones((1, 2))
    dist = Stationary().v(mdp, policy)

    assert jnp.allclose(dist, jnp.array([0.5, 0.5]))
    assert jnp.allclose(mdp.transition[0] @ dist, dist)
    assert jnp.allclose(Stationary().q(mdp, policy), policy * dist)


def test_stationary_selects_the_minimum_norm_distribution_when_nonunique() -> None:
    mdp = MDP(
        transition=jnp.eye(2)[None, ...],
        reward=jnp.zeros((1, 2, 2)),
        initial=jnp.array([1.0, 0.0]),
        terminal=jnp.zeros(2),
    )
    policy = jnp.ones((1, 2))

    assert jnp.allclose(Stationary().v(mdp, policy), jnp.array([0.5, 0.5]))


def test_distribution_mappings_compose_with_jit_and_vmap() -> None:
    mdp = _periodic_mdp()
    policies = jnp.ones((3, 1, 2))
    occupancy = Occupancy(step=3)
    stationary = Stationary()

    finite = chex.chexify(
        jax.jit(jax.vmap(lambda policy: occupancy.v(mdp, policy))),
        async_check=False,
    )(policies)
    invariant = chex.chexify(
        jax.jit(jax.vmap(lambda policy: stationary.v(mdp, policy))),
        async_check=False,
    )(policies)

    assert is_dataclass(occupancy)
    assert is_dataclass(stationary)
    assert finite.shape == (3, 2)
    assert invariant.shape == (3, 2)


def test_eigenvalues_describe_the_policy_transition() -> None:
    values = eigenvalues(_periodic_mdp(), jnp.ones((1, 2)))

    assert jnp.allclose(jnp.sort(values.real), jnp.array([-1.0, 1.0]))
    assert jnp.allclose(values.imag, 0)


def test_occupancy_rejects_negative_step() -> None:
    mdp = _periodic_mdp()

    with pytest.raises(AssertionError, match="step"):
        Occupancy(step=-1).v(mdp, jnp.ones((1, 2)))


@pytest.mark.parametrize("gamma", [-0.1, 1.1, jnp.inf, jnp.nan])
def test_occupancy_rejects_invalid_gamma(gamma: float) -> None:
    mdp = _periodic_mdp()

    with pytest.raises(AssertionError, match="gamma"):
        Occupancy().v(mdp, jnp.ones((1, 2)), gamma=gamma)
