from dataclasses import is_dataclass

import chex
import jax
import jax.numpy as jnp
import pytest

import jaxdp
from jaxdp.mapping import MellowMax, expectation, greedy_map
from jaxdp.mdp import MDP, make_mrp
from jaxdp.operator import (
    BoltzmannBellmanOp,
    MellowMaxBellmanOptOp,
    SoftBellmanOptOp,
    adj_trans_op,
    bellman_op,
    bellman_opt_op,
    trans_op,
)
from jaxdp.planning import policy_eval


def _two_state_mdp() -> MDP:
    transition = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    reward = jnp.zeros((2, 2, 2)).at[0, 1, 1].set(1.0).at[1, 0, 1].set(2.0).at[1, 1, 0].set(3.0)
    return MDP(
        transition=transition,
        reward=reward,
        initial=jnp.array([1.0, 0.0]),
        terminal=jnp.zeros(2),
    )


def test_public_bellman_names() -> None:
    assert jaxdp.trans_op is trans_op
    assert jaxdp.adj_trans_op is adj_trans_op
    assert jaxdp.bellman_op is bellman_op
    assert jaxdp.bellman_opt_op is bellman_opt_op
    assert jaxdp.SoftBellmanOptOp is SoftBellmanOptOp
    assert jaxdp.MellowMaxBellmanOptOp is MellowMaxBellmanOptOp
    assert jaxdp.BoltzmannBellmanOp is BoltzmannBellmanOp
    assert not hasattr(jaxdp, "MellowmaxBellmanOptOp")
    assert not hasattr(jaxdp, "Bellman")
    assert not hasattr(jaxdp, "BellmanOptimality")
    assert not hasattr(jaxdp, "Optimality")
    assert not hasattr(jaxdp, "ValueMap")
    assert not hasattr(jaxdp, "TransOp")
    assert not hasattr(jaxdp, "AdjTransOp")
    assert not hasattr(jaxdp, "BellmanOp")
    assert not hasattr(jaxdp, "BellmanOptOp")
    assert isinstance(trans_op, type)
    assert isinstance(bellman_op, type)


def test_transition_operators_and_expectation() -> None:
    mdp = _two_state_mdp()
    v_val = jnp.array([4.0, 8.0])
    sa_vec = trans_op.sa(mdp, v_val)
    mrp = make_mrp(mdp, jnp.full((2, 2), 0.5))
    s_vec = trans_op.s(mrp, v_val)
    s_dist = jnp.array([0.25, 0.75])
    sa_dist = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
    q_val = reward + 0.5 * sa_vec
    dist = greedy_map.q(q_val) * mdp.initial

    assert jnp.array_equal(sa_vec, jnp.array([[4.0, 8.0], [8.0, 4.0]]))
    assert jnp.array_equal(s_vec, jnp.array([6.0, 6.0]))
    assert jnp.allclose(
        jnp.sum(s_vec * s_dist),
        jnp.sum(v_val * adj_trans_op.s(mrp, s_dist)),
    )
    assert jnp.allclose(
        jnp.sum(sa_vec * sa_dist),
        jnp.sum(v_val * adj_trans_op.sa(mdp, sa_dist)),
    )
    assert jnp.allclose(q_val, jnp.array([[2.0, 5.0], [6.0, 5.0]]))
    assert expectation.s(v_val, mdp.initial) == 4.0
    assert expectation.sa(q_val, dist) == 6.0


def test_transition_operators_compose_with_jax() -> None:
    mdp = _two_state_mdp()
    mrp = make_mrp(mdp, jnp.full((2, 2), 0.5))
    vec = jnp.array([[4.0, 8.0], [8.0, 4.0]])
    dist = jnp.full((2, 2, 2), 0.25)

    backward = jax.jit(jax.vmap(lambda item: trans_op.sa(mdp, item)))(vec)
    forward = jax.jit(jax.vmap(lambda item: adj_trans_op.sa(mdp, item)))(dist)
    state_forward = jax.jit(jax.vmap(lambda item: adj_trans_op.s(mrp, item)))(vec)

    assert not is_dataclass(trans_op)
    assert not is_dataclass(adj_trans_op)
    assert backward.shape == (2, mdp.action_size, mdp.state_size)
    assert forward.shape == (2, mdp.state_size)
    assert state_forward.shape == vec.shape


def test_state_action_value_does_not_bootstrap_terminal_successors() -> None:
    mdp = MDP(
        transition=jnp.array([[[0.0, 0.0], [1.0, 1.0]]]),
        reward=jnp.zeros((1, 2, 2)).at[0, 0, 1].set(2.0),
        initial=jnp.array([1.0, 0.0]),
        terminal=jnp.array([0.0, 1.0]),
    )

    vec = jnp.array([0.0, 100.0])
    mrp = make_mrp(mdp, jnp.ones((1, 2)))

    assert jnp.array_equal(trans_op.sa(mdp, vec), jnp.zeros((1, 2)))
    assert jnp.array_equal(trans_op.s(mrp, vec), jnp.zeros(2))
    assert jnp.array_equal(adj_trans_op.sa(mdp, jnp.ones((1, 2))), jnp.zeros(2))
    assert jnp.array_equal(adj_trans_op.s(mrp, jnp.ones(2)), jnp.zeros(2))
    assert jnp.array_equal(
        bellman_opt_op.v(mdp, vec, gamma=0.9),
        jnp.array([2.0, 0.0]),
    )
    assert jnp.array_equal(
        policy_eval.v(mdp, jnp.ones((1, 2)), gamma=0.9),
        jnp.array([2.0, 0.0]),
    )


def test_policy_evaluation_matches_analytical_solution() -> None:
    mdp = _two_state_mdp()
    policy = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    evaluate = policy_eval

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

    bellman = bellman_op
    bellman_optimality = bellman_opt_op

    assert jnp.allclose(bellman.v(mdp, policy, value, 0.5), jnp.array([4.0, 5.0]))
    assert jnp.allclose(bellman.q(mdp, policy, q, 0.5), jnp.array([[2.0, 3.5], [4.5, 5.0]]))
    assert jnp.allclose(bellman_optimality.v(mdp, value, 0.5), jnp.array([6.0, 5.0]))
    assert jnp.allclose(
        bellman_optimality.q(mdp, q, 0.5),
        jnp.array([[3.0, 3.5], [4.5, 6.0]]),
    )


def test_smooth_bellman_operators_match_action_reductions() -> None:
    mdp = _two_state_mdp()
    v_val = jnp.array([4.0, 8.0])
    q_val = jnp.array([[2.0, 5.0], [6.0, 5.0]])
    temperature = 2.0
    immediate_q = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
    soft_v = temperature * jax.nn.logsumexp(immediate_q / temperature, axis=0)
    mellowmax_v = MellowMax(temperature=temperature).q(immediate_q)
    boltzmann_policy = jax.nn.softmax(immediate_q / temperature, axis=0)
    boltzmann_v = jnp.sum(boltzmann_policy * immediate_q, axis=0)

    soft = SoftBellmanOptOp(temperature=temperature)
    mellowmax = MellowMaxBellmanOptOp(temperature=temperature)
    boltzmann = BoltzmannBellmanOp(temperature=temperature)

    assert jnp.allclose(soft.v(mdp, v_val, 0.0), soft_v)
    assert jnp.allclose(mellowmax.v(mdp, v_val, 0.0), mellowmax_v)
    assert jnp.allclose(boltzmann.v(mdp, v_val, 0.0), boltzmann_v)

    soft_q = temperature * jax.nn.logsumexp(q_val / temperature, axis=0)
    mellowmax_q = MellowMax(temperature=temperature).q(q_val)
    boltzmann_q = jnp.sum(jax.nn.softmax(q_val / temperature, axis=0) * q_val, axis=0)

    assert jnp.allclose(
        soft.q(mdp, q_val, 0.5),
        immediate_q + 0.5 * trans_op.sa(mdp, soft_q),
    )
    assert jnp.allclose(
        mellowmax.q(mdp, q_val, 0.5),
        immediate_q + 0.5 * trans_op.sa(mdp, mellowmax_q),
    )
    assert jnp.allclose(
        boltzmann.q(mdp, q_val, 0.5),
        immediate_q + 0.5 * trans_op.sa(mdp, boltzmann_q),
    )


def test_soft_and_mellowmax_differ_by_uniform_policy_cost() -> None:
    mdp = _two_state_mdp()
    temperature = 0.75
    v_val = jnp.array([4.0, 8.0])
    difference = SoftBellmanOptOp(temperature=temperature).v(
        mdp,
        v_val,
        0.5,
    ) - MellowMaxBellmanOptOp(temperature=temperature).v(mdp, v_val, 0.5)

    assert jnp.allclose(difference, temperature * jnp.log(mdp.action_size))


@pytest.mark.parametrize(
    "operator",
    [
        SoftBellmanOptOp(temperature=0.0),
        MellowMaxBellmanOptOp(temperature=-1.0),
        BoltzmannBellmanOp(temperature=float("inf")),
        BoltzmannBellmanOp(temperature=float("nan")),
    ],
)
def test_smooth_bellman_operators_reject_invalid_temperature(
    operator: SoftBellmanOptOp | MellowMaxBellmanOptOp | BoltzmannBellmanOp,
) -> None:
    with pytest.raises(AssertionError, match="temperature"):
        operator.v(_two_state_mdp(), jnp.zeros(2), gamma=0.5)


@pytest.mark.parametrize(
    "operator",
    [
        SoftBellmanOptOp(temperature=1.0),
        MellowMaxBellmanOptOp(temperature=1.0),
        BoltzmannBellmanOp(temperature=1.0),
    ],
)
def test_smooth_bellman_operators_compose_with_jax(
    operator: SoftBellmanOptOp | MellowMaxBellmanOptOp | BoltzmannBellmanOp,
) -> None:
    mdp = _two_state_mdp()
    values = jnp.array([[4.0, 8.0], [10_000.0, 10_000.0]])
    apply = chex.chexify(
        jax.jit(jax.vmap(lambda v_val: operator.v(mdp, v_val, gamma=0.5))),
        async_check=False,
    )
    result = apply(values)
    gradient = jax.grad(lambda v_val: jnp.sum(operator.v(mdp, v_val, gamma=0.5)))(values[0])

    assert is_dataclass(operator)
    assert result.shape == values.shape
    assert jnp.all(jnp.isfinite(result))
    assert jnp.all(jnp.isfinite(gradient))


@pytest.mark.parametrize("gamma", [-0.1, 1.0, jnp.inf, jnp.nan])
def test_discounted_operators_reject_invalid_gamma(gamma: float) -> None:
    mdp = _two_state_mdp()
    policy = jnp.full((2, 2), 0.5)
    value = jnp.zeros(2)

    with pytest.raises(AssertionError, match="gamma"):
        policy_eval.v(mdp, policy, gamma)
    with pytest.raises(AssertionError, match="gamma"):
        bellman_op.v(mdp, policy, value, gamma)
    with pytest.raises(AssertionError, match="gamma"):
        bellman_opt_op.v(mdp, value, gamma)


def test_stateless_operators_compose_with_jax() -> None:
    mdp = _two_state_mdp()
    operator = bellman_opt_op
    values = jnp.array([[4.0, 8.0], [8.0, 4.0]])
    gammas = jnp.array([0.5, 0.25])
    apply = chex.chexify(
        jax.jit(jax.vmap(lambda value, gamma: operator.v(mdp, value, gamma))),
        async_check=False,
    )
    result = apply(
        values,
        gammas,
    )

    assert not is_dataclass(operator)
    assert result.shape == values.shape


def test_operator_validation_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    value = jnp.zeros(2)
    operator = bellman_opt_op
    apply_gamma = chex.chexify(
        jax.jit(jax.vmap(lambda gamma: operator.v(mdp, value, gamma))),
        async_check=False,
    )

    assert apply_gamma(jnp.array([0.25, 0.5])).shape == (2, mdp.state_size)
    with pytest.raises(AssertionError, match="gamma"):
        apply_gamma(jnp.array([0.5, 1.0]))

    evaluate = policy_eval
    policies = jnp.full((2, mdp.action_size, mdp.state_size), 0.5)
    apply_policy = chex.chexify(
        jax.jit(jax.vmap(lambda policy: evaluate.v(mdp, policy, 0.5))),
        async_check=False,
    )

    assert apply_policy(policies).shape == (2, mdp.state_size)
    with pytest.raises(AssertionError, match="policy"):
        apply_policy(policies.at[1, 0, 0].set(0.75))
