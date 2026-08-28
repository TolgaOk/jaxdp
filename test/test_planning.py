from dataclasses import is_dataclass

import chex
import jax
import jax.numpy as jnp

import jaxdp
from jaxdp.mapping import greedy_map
from jaxdp.mdp import MDP
from jaxdp.operator import bellman_opt_op
from jaxdp.planning import (
    AcceleratedPolicyIteration,
    AnchoredQValueIteration,
    AnchoredValueIteration,
    MomentumValueIteration,
    PolicyIteration,
    QValueIteration,
    RankOneValueIteration,
    SafeAcceleratedValueIteration,
    ValueIteration,
    policy_eval,
)


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


def test_public_planning_names() -> None:
    assert jaxdp.AcceleratedPolicyIteration is AcceleratedPolicyIteration
    assert jaxdp.ValueIteration is ValueIteration
    assert jaxdp.QValueIteration is QValueIteration
    assert jaxdp.AnchoredValueIteration is AnchoredValueIteration
    assert jaxdp.AnchoredQValueIteration is AnchoredQValueIteration
    assert jaxdp.RankOneValueIteration is RankOneValueIteration
    assert jaxdp.SafeAcceleratedValueIteration is SafeAcceleratedValueIteration
    assert jaxdp.MomentumValueIteration is MomentumValueIteration
    assert jaxdp.PolicyIteration is PolicyIteration
    assert jaxdp.policy_eval is policy_eval


def test_value_iteration_updates_state_values_once() -> None:
    mdp = _two_state_mdp()
    planner = ValueIteration(gamma=0.5)
    state = planner.init(mdp)
    updated = planner.update(mdp, state)
    expected = bellman_opt_op.v(mdp, state.v_val, planner.gamma)

    assert is_dataclass(planner)
    assert is_dataclass(state)
    assert state.v_val.shape == (mdp.state_size,)
    assert jnp.array_equal(state.v_val, jnp.zeros(mdp.state_size))
    assert jnp.allclose(updated.v_val, expected)


def test_q_value_iteration_updates_action_values_once() -> None:
    mdp = _two_state_mdp()
    planner = QValueIteration(gamma=0.5)
    state = planner.init(mdp)
    updated = planner.update(mdp, state)
    expected = bellman_opt_op.q(mdp, state.q_val, planner.gamma)

    assert is_dataclass(planner)
    assert is_dataclass(state)
    assert state.q_val.shape == (mdp.action_size, mdp.state_size)
    assert jnp.array_equal(state.q_val, jnp.zeros((mdp.action_size, mdp.state_size)))
    assert jnp.allclose(updated.q_val, expected)


def test_anchored_value_iteration_matches_paper_recurrence() -> None:
    mdp = _two_state_mdp()
    gamma = 0.5
    v_anchor = jnp.array([1.0, -1.0])
    q_anchor = jnp.array([[1.0, -1.0], [2.0, -2.0]])
    value_iteration = AnchoredValueIteration(gamma=gamma)
    q_value_iteration = AnchoredQValueIteration(gamma=gamma)
    v_state = value_iteration.init(mdp, v_anchor)
    q_state = q_value_iteration.init(mdp, q_anchor)

    for step in range(1, 4):
        previous_v = v_state.v_val
        previous_q = q_state.q_val
        v_state = value_iteration.update(mdp, v_state)
        q_state = q_value_iteration.update(mdp, q_state)
        beta = 1 / jnp.sum(gamma ** (-2 * jnp.arange(step + 1)))
        expected_v = beta * v_anchor + (1 - beta) * bellman_opt_op.v(
            mdp,
            previous_v,
            gamma,
        )
        expected_q = beta * q_anchor + (1 - beta) * bellman_opt_op.q(
            mdp,
            previous_q,
            gamma,
        )

        assert jnp.allclose(v_state.beta, beta)
        assert jnp.allclose(q_state.beta, beta)
        assert jnp.allclose(v_state.v_val, expected_v)
        assert jnp.allclose(q_state.q_val, expected_q)


def test_anchored_value_iteration_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    value_iteration = AnchoredValueIteration(gamma=0.5)
    q_value_iteration = AnchoredQValueIteration(gamma=0.5)
    v_anchors = jnp.array([[1.0, -1.0], [-1.0, 1.0]])
    q_anchors = jnp.array(
        [
            [[1.0, -1.0], [2.0, -2.0]],
            [[-1.0, 1.0], [-2.0, 2.0]],
        ]
    )
    init_v = jax.jit(jax.vmap(value_iteration.init, in_axes=(None, 0)))
    init_q = jax.jit(jax.vmap(q_value_iteration.init, in_axes=(None, 0)))
    update_v = chex.chexify(
        jax.jit(jax.vmap(value_iteration.update, in_axes=(None, 0))),
        async_check=False,
    )
    update_q = chex.chexify(
        jax.jit(jax.vmap(q_value_iteration.update, in_axes=(None, 0))),
        async_check=False,
    )

    v_states = update_v(mdp, init_v(mdp, v_anchors))
    q_states = update_q(mdp, init_q(mdp, q_anchors))

    assert v_states.v_val.shape == v_anchors.shape
    assert q_states.q_val.shape == q_anchors.shape
    assert jnp.array_equal(v_states.v_anchor, v_anchors)
    assert jnp.array_equal(q_states.q_anchor, q_anchors)


def test_rank_one_value_iteration_matches_algorithm_one() -> None:
    mdp = _two_state_mdp()
    planner = RankOneValueIteration(gamma=0.5)
    state = planner.init(
        mdp,
        v_val=jnp.zeros(mdp.state_size),
        dist=jnp.array([0.75, 0.25]),
    )

    updated = planner.update(mdp, state)

    assert jnp.allclose(updated.dist, jnp.array([0.25, 0.75]))
    assert jnp.allclose(updated.v_val, jnp.array([4.75, 5.75]))


def test_rank_one_value_iteration_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    planner = RankOneValueIteration(gamma=0.5)
    v_vals = jnp.array([[0.0, 0.0], [1.0, -1.0]])
    dists = jnp.array([[0.75, 0.25], [0.25, 0.75]])
    init = jax.jit(jax.vmap(planner.init, in_axes=(None, 0, 0)))
    update = chex.chexify(
        jax.jit(jax.vmap(planner.update, in_axes=(None, 0))),
        async_check=False,
    )

    states = update(mdp, init(mdp, v_vals, dists))

    assert states.v_val.shape == v_vals.shape
    assert states.dist.shape == dists.shape
    assert jnp.allclose(jnp.sum(states.dist, axis=-1), jnp.ones(2))


def test_safe_accelerated_value_iteration_matches_paper_recurrence() -> None:
    mdp = _two_state_mdp()
    planner = SafeAcceleratedValueIteration(
        gamma=0.5,
        rate=0.75,
        step_size=1.0,
        momentum=0.25,
    )
    state = planner.init(mdp, jnp.zeros(mdp.state_size))

    updated = planner.update(mdp, state)

    assert jnp.array_equal(state.prev_v_val, jnp.zeros(2))
    assert jnp.allclose(state.v_val, jnp.array([2.0, 3.0]))
    assert jnp.allclose(state.bound, 2.25)
    assert updated.accepted
    assert jnp.allclose(updated.prev_v_val, state.v_val)
    assert jnp.allclose(updated.v_val, jnp.array([3.875, 4.25]))
    assert jnp.allclose(updated.bound, 1.6875)


def test_safe_accelerated_value_iteration_falls_back_to_vi() -> None:
    mdp = _two_state_mdp()
    planner = SafeAcceleratedValueIteration(
        gamma=0.5,
        rate=0.5,
        step_size=10.0,
        momentum=10.0,
    )
    state = planner.init(mdp, jnp.zeros(mdp.state_size))

    updated = planner.update(mdp, state)

    assert not updated.accepted
    assert jnp.allclose(updated.v_val, bellman_opt_op.v(mdp, state.v_val, planner.gamma))


def test_safe_accelerated_value_iteration_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    planner = SafeAcceleratedValueIteration(gamma=0.5)
    v_vals = jnp.array([[0.0, 0.0], [1.0, -1.0]])
    init = chex.chexify(
        jax.jit(jax.vmap(planner.init, in_axes=(None, 0))),
        async_check=False,
    )
    update = chex.chexify(
        jax.jit(jax.vmap(planner.update, in_axes=(None, 0))),
        async_check=False,
    )

    states = update(mdp, init(mdp, v_vals))

    assert states.v_val.shape == v_vals.shape
    assert states.prev_v_val.shape == v_vals.shape
    assert states.bound.shape == (2,)
    assert states.accepted.shape == (2,)


def test_momentum_value_iteration_matches_paper_recurrence() -> None:
    mdp = _two_state_mdp()
    planner = MomentumValueIteration(
        gamma=0.5,
        step_size=0.5,
        momentum=0.25,
    )
    state = planner.init(mdp, jnp.zeros(mdp.state_size))

    updated = planner.update(mdp, state)

    assert jnp.array_equal(state.prev_v_val, jnp.zeros(2))
    assert jnp.allclose(state.v_val, jnp.array([2.0, 3.0]))
    assert jnp.allclose(updated.prev_v_val, state.v_val)
    assert jnp.allclose(updated.v_val, jnp.array([3.25, 4.25]))


def test_momentum_value_iteration_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    planner = MomentumValueIteration(gamma=0.5)
    v_vals = jnp.array([[0.0, 0.0], [1.0, -1.0]])
    init = chex.chexify(
        jax.jit(jax.vmap(planner.init, in_axes=(None, 0))),
        async_check=False,
    )
    update = chex.chexify(
        jax.jit(jax.vmap(planner.update, in_axes=(None, 0))),
        async_check=False,
    )

    states = update(mdp, init(mdp, v_vals))

    assert states.v_val.shape == v_vals.shape
    assert states.prev_v_val.shape == v_vals.shape


def test_accelerated_policy_iteration_matches_degree_two_recurrence() -> None:
    mdp = _two_state_mdp()
    policy = jnp.full((mdp.action_size, mdp.state_size), 0.5)
    planner = AcceleratedPolicyIteration(gamma=0.5, degree=2, tolerance=0.0)
    state = planner.init(mdp, policy)

    updated = planner.update(mdp, state)

    alpha = (1 - jnp.sqrt(0.5)) / (1 + jnp.sqrt(0.5))
    expected_x = jnp.array([1.0, 2.0])
    assert not updated.improved
    assert not updated.stable
    assert jnp.array_equal(updated.policy, policy)
    assert jnp.allclose(updated.history[0], expected_x)
    assert jnp.allclose(updated.v_val, (1 + alpha) * expected_x)


def test_accelerated_policy_iteration_improves_and_detects_stability() -> None:
    mdp = _two_state_mdp()
    policy = jnp.full((mdp.action_size, mdp.state_size), 0.5)
    planner = AcceleratedPolicyIteration(gamma=0.5, degree=2, tolerance=10.0)
    state = planner.init(mdp, policy)

    improved = planner.update(mdp, state)
    stable = planner.update(mdp, improved)

    expected_policy = greedy_map.v(mdp, state.v_val, planner.gamma)
    assert improved.improved
    assert not improved.stable
    assert jnp.array_equal(improved.policy, expected_policy)
    assert jnp.array_equal(improved.v_val, state.v_val)
    assert jnp.array_equal(improved.history, state.history)
    assert stable.improved
    assert stable.stable


def test_accelerated_policy_iteration_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    planner = AcceleratedPolicyIteration(gamma=0.5, degree=4)
    policies = jnp.stack(
        (
            jnp.full((mdp.action_size, mdp.state_size), 0.5),
            jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        )
    )
    v_vals = jnp.array([[0.0, 0.0], [1.0, -1.0]])
    init = jax.jit(jax.vmap(planner.init, in_axes=(None, 0, 0)))
    update = chex.chexify(
        jax.jit(jax.vmap(planner.update, in_axes=(None, 0))),
        async_check=False,
    )

    states = update(mdp, init(mdp, policies, v_vals))

    assert states.policy.shape == policies.shape
    assert states.v_val.shape == v_vals.shape
    assert states.history.shape == (2, planner.degree - 1, mdp.state_size)
    assert states.improved.shape == (2,)
    assert states.stable.shape == (2,)


def test_policy_iteration_keeps_policy_and_value_aligned() -> None:
    mdp = _two_state_mdp()
    initial_policy = jnp.full((mdp.action_size, mdp.state_size), 0.5)
    planner = PolicyIteration(gamma=0.5)
    state = planner.init(mdp, initial_policy)
    updated = planner.update(mdp, state)
    expected_policy = greedy_map.v(mdp, state.v_val, planner.gamma)

    assert jnp.allclose(state.v_val, policy_eval.v(mdp, state.policy, planner.gamma))
    assert jnp.array_equal(updated.policy, expected_policy)
    assert jnp.allclose(
        updated.v_val,
        policy_eval.v(mdp, updated.policy, planner.gamma),
    )


def test_planning_states_compose_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    value_iteration = ValueIteration(gamma=0.5)
    value_states = ValueIteration.State(
        v_val=jnp.array([[4.0, 8.0], [8.0, 4.0]]),
    )
    update_values = chex.chexify(
        jax.jit(jax.vmap(value_iteration.update, in_axes=(None, 0))),
        async_check=False,
    )

    q_value_iteration = QValueIteration(gamma=0.5)
    q_states = QValueIteration.State(
        q_val=jnp.array(
            [
                [[2.0, 5.0], [6.0, 5.0]],
                [[6.0, 5.0], [2.0, 5.0]],
            ]
        ),
    )
    update_q_values = chex.chexify(
        jax.jit(jax.vmap(q_value_iteration.update, in_axes=(None, 0))),
        async_check=False,
    )

    value_states = update_values(mdp, value_states)
    q_states = update_q_values(mdp, q_states)

    assert value_states.v_val.shape == (2, mdp.state_size)
    assert q_states.q_val.shape == (2, mdp.action_size, mdp.state_size)


def test_policy_iteration_state_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    planner = PolicyIteration(gamma=0.5)
    policies = jnp.stack(
        (
            jnp.full((mdp.action_size, mdp.state_size), 0.5),
            jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        )
    )
    init = chex.chexify(
        jax.jit(jax.vmap(planner.init, in_axes=(None, 0))),
        async_check=False,
    )
    update = chex.chexify(
        jax.jit(jax.vmap(planner.update, in_axes=(None, 0))),
        async_check=False,
    )

    states = update(mdp, init(mdp, policies))

    assert states.policy.shape == policies.shape
    assert states.v_val.shape == (2, mdp.state_size)
