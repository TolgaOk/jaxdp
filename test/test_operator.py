from dataclasses import is_dataclass

import jax
import jax.numpy as jnp
import pytest

import jaxdp
from jaxdp.mdp import MDP
from jaxdp.operator import (
    Bellman,
    BellmanOptimality,
    Expected,
    PolicyEvaluation,
    greedy_state_value,
    state_action_value,
)


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
    return MDP(
        transition=transition,
        reward=reward,
        initial=jnp.array([1.0, 0.0]),
        terminal=jnp.zeros(2),
    )


def test_public_name_is_bellman_optimality() -> None:
    assert jaxdp.BellmanOptimality is BellmanOptimality
    assert not hasattr(jaxdp, "Optimality")


def test_value_conversions_and_expectation() -> None:
    mdp = _two_state_mdp()
    value = jnp.array([4.0, 8.0])
    q = state_action_value(mdp, value, gamma=0.5)

    assert jnp.allclose(q, jnp.array([[2.0, 5.0], [6.0, 5.0]]))
    assert jnp.array_equal(greedy_state_value(q), jnp.array([6.0, 5.0]))
    assert Expected().v(mdp, value) == 4.0
    assert Expected().q(mdp, q) == 6.0


def test_state_action_value_does_not_bootstrap_terminal_successors() -> None:
    mdp = MDP(
        transition=jnp.array([[[0.0, 0.0], [1.0, 1.0]]]),
        reward=jnp.zeros((1, 2, 2)).at[0, 0, 1].set(2.0),
        initial=jnp.array([1.0, 0.0]),
        terminal=jnp.array([0.0, 1.0]),
    )

    assert jnp.array_equal(
        state_action_value(mdp, jnp.array([0.0, 100.0]), gamma=0.9),
        jnp.array([[2.0, 0.0]]),
    )
    assert jnp.array_equal(
        PolicyEvaluation().v(mdp, jnp.ones((1, 2)), gamma=0.9),
        jnp.array([2.0, 0.0]),
    )


def test_policy_evaluation_matches_analytical_solution() -> None:
    mdp = _two_state_mdp()
    policy = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    evaluate = PolicyEvaluation()

    assert jnp.allclose(evaluate.v(mdp, policy, 0.5), jnp.array([14.0 / 3.0, 16.0 / 3.0]))
    assert jnp.allclose(
        evaluate.q(mdp, policy, 0.5),
        jnp.array([[7.0 / 3.0, 11.0 / 3.0], [14.0 / 3.0, 16.0 / 3.0]]),
    )


def test_bellman_operators() -> None:
    mdp = _two_state_mdp()
    policy = jnp.full((2, 2), 0.5)
    value = jnp.array([4.0, 8.0])
    q = jnp.array([[2.0, 5.0], [6.0, 5.0]])

    bellman = Bellman()
    bellman_optimality = BellmanOptimality()

    assert jnp.allclose(bellman.v(mdp, policy, value, 0.5), jnp.array([4.0, 5.0]))
    assert jnp.allclose(bellman.q(mdp, policy, q, 0.5), jnp.array([[2.0, 3.5], [4.5, 5.0]]))
    assert jnp.allclose(bellman_optimality.v(mdp, value, 0.5), jnp.array([6.0, 5.0]))
    assert jnp.allclose(
        bellman_optimality.q(mdp, q, 0.5),
        jnp.array([[3.0, 3.5], [4.5, 6.0]]),
    )


@pytest.mark.parametrize("gamma", [-0.1, 1.0, jnp.inf, jnp.nan])
def test_discounted_operators_reject_invalid_gamma(gamma: float) -> None:
    mdp = _two_state_mdp()
    policy = jnp.full((2, 2), 0.5)
    value = jnp.zeros(2)

    with pytest.raises(ValueError, match="gamma"):
        PolicyEvaluation().v(mdp, policy, gamma)
    with pytest.raises(ValueError, match="gamma"):
        Bellman().v(mdp, policy, value, gamma)
    with pytest.raises(ValueError, match="gamma"):
        BellmanOptimality().v(mdp, value, gamma)


def test_operators_are_immutable_dataclasses_and_compose_with_jax() -> None:
    mdp = _two_state_mdp()
    operator = BellmanOptimality()
    values = jnp.array([[4.0, 8.0], [8.0, 4.0]])
    gammas = jnp.array([0.5, 0.25])
    result = jax.jit(jax.vmap(lambda value, gamma: operator.v(mdp, value, gamma)))(
        values,
        gammas,
    )

    assert is_dataclass(operator)
    assert result.shape == values.shape
