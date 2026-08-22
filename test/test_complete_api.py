import jax
import jax.numpy as jnp

from jaxdp.base import (
    bellman_optimality_operator,
    policy_evaluation,
    stationary_distribution,
    to_state_action_value,
)
from jaxdp.mdp import MDP
from jaxdp.policy import EpsilonGreedy, Soft


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
    return MDP(transition, reward, initial, terminal)


def test_value_policies_apply_one_step_lookahead() -> None:
    mdp = _two_state_mdp()
    value = jnp.array([4.0, 8.0])
    gamma = 0.5
    q_value = to_state_action_value(mdp, value, gamma)
    soft = Soft(temperature=2.0)
    epsilon_greedy = EpsilonGreedy(epsilon=0.2)

    assert jnp.allclose(
        soft.v(mdp, value, gamma),
        soft.q(q_value),
    )
    assert jnp.allclose(
        epsilon_greedy.v(mdp, value, gamma),
        epsilon_greedy.q(q_value),
    )


def test_bellman_optimality_v_selects_best_action_value() -> None:
    mdp = _two_state_mdp()
    value = jnp.array([4.0, 8.0])

    result = bellman_optimality_operator.v(mdp, value, gamma=0.5)

    assert jnp.allclose(result, jnp.array([6.0, 5.0]))


def test_policy_evaluation_matches_analytical_solution() -> None:
    mdp = _two_state_mdp()
    policy = jnp.array([[0.0, 0.0], [1.0, 1.0]])

    value = policy_evaluation.v(mdp, policy, gamma=0.5)
    q_value = policy_evaluation.q(mdp, policy, gamma=0.5)

    assert jnp.allclose(value, jnp.array([14.0 / 3.0, 16.0 / 3.0]))
    assert jnp.allclose(
        q_value,
        jnp.array([[7.0 / 3.0, 11.0 / 3.0], [14.0 / 3.0, 16.0 / 3.0]]),
    )


def test_stationary_state_and_action_distributions_are_consistent() -> None:
    mdp = _two_state_mdp()
    policy = jnp.full((2, 2), 0.5)

    initial_v = stationary_distribution.v(mdp, policy, iterations=0)
    initial_q = stationary_distribution.q(mdp, policy, iterations=0)
    next_v = stationary_distribution.v(mdp, policy, iterations=1)
    next_q = stationary_distribution.q(mdp, policy, iterations=1)

    assert jnp.allclose(initial_v, mdp.initial)
    assert jnp.allclose(initial_q, policy * mdp.initial)
    assert jnp.allclose(next_v, jnp.array([0.5, 0.5]))
    assert jnp.allclose(jnp.sum(next_q, axis=0), next_v)


def test_completed_api_supports_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    soft = Soft(temperature=2.0)
    values = jnp.array([[4.0, 8.0], [8.0, 4.0]])

    policies = jax.jit(jax.vmap(lambda value: soft.v(mdp, value, 0.5)))(values)
    optimal_values = jax.jit(
        jax.vmap(lambda value: bellman_optimality_operator.v(mdp, value, 0.5))
    )(values)
    state_distribution = jax.jit(
        lambda policy: stationary_distribution.v(mdp, policy, iterations=3)
    )(jnp.full((2, 2), 0.5))

    assert policies.shape == (2, mdp.action_size, mdp.state_size)
    assert optimal_values.shape == values.shape
    assert jnp.allclose(state_distribution, jnp.array([0.5, 0.5]))
