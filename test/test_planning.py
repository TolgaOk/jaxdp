from dataclasses import is_dataclass

import chex
import jax
import jax.numpy as jnp

import jaxdp
from jaxdp.mapping import greedy_map
from jaxdp.mdp import MDP
from jaxdp.operator import bellman_opt_op
from jaxdp.planning import PolicyIteration, QValueIteration, ValueIteration, policy_eval


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


def test_public_planning_names() -> None:
    assert jaxdp.ValueIteration is ValueIteration
    assert jaxdp.QValueIteration is QValueIteration
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
