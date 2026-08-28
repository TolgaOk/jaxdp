from dataclasses import is_dataclass, replace

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
    AndersonValueIteration,
    DeflatedValueIteration,
    DynamicBoltzmannValueIteration,
    IterativePolicyEvaluation,
    MomentumValueIteration,
    PIDValueIteration,
    PolicyIteration,
    QuasiPolicyIteration,
    QValueIteration,
    RankOneValueIteration,
    SafeAcceleratedValueIteration,
    SafeAndersonValueIteration,
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
    assert jaxdp.AndersonValueIteration is AndersonValueIteration
    assert jaxdp.SafeAndersonValueIteration is SafeAndersonValueIteration
    assert jaxdp.IterativePolicyEvaluation is IterativePolicyEvaluation
    assert jaxdp.ValueIteration is ValueIteration
    assert jaxdp.QValueIteration is QValueIteration
    assert jaxdp.AnchoredValueIteration is AnchoredValueIteration
    assert jaxdp.AnchoredQValueIteration is AnchoredQValueIteration
    assert jaxdp.RankOneValueIteration is RankOneValueIteration
    assert jaxdp.DeflatedValueIteration is DeflatedValueIteration
    assert jaxdp.QuasiPolicyIteration is QuasiPolicyIteration
    assert jaxdp.DynamicBoltzmannValueIteration is DynamicBoltzmannValueIteration
    assert jaxdp.SafeAcceleratedValueIteration is SafeAcceleratedValueIteration
    assert jaxdp.MomentumValueIteration is MomentumValueIteration
    assert jaxdp.PIDValueIteration is PIDValueIteration
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


def test_iterative_policy_evaluation_updates_once() -> None:
    mdp = _two_state_mdp()
    policy = jnp.full((mdp.action_size, mdp.state_size), 1 / mdp.action_size)
    planner = IterativePolicyEvaluation(gamma=0.5)
    state = planner.init(mdp, policy, jnp.array([1.0, -1.0]))

    updated = planner.update(mdp, state)

    expected = jaxdp.bellman_op.v(mdp, policy, state.v_val, planner.gamma)
    assert jnp.array_equal(updated.policy, policy)
    assert jnp.allclose(updated.v_val, expected)


def test_iterative_policy_evaluation_converges_to_exact_value() -> None:
    mdp = _two_state_mdp()
    policy = jnp.full((mdp.action_size, mdp.state_size), 1 / mdp.action_size)
    planner = IterativePolicyEvaluation(gamma=0.5)
    state = planner.init(mdp, policy)

    for _ in range(40):
        state = planner.update(mdp, state)

    assert jnp.allclose(state.v_val, policy_eval.v(mdp, policy, planner.gamma), atol=1e-5)


def test_iterative_policy_evaluation_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    planner = IterativePolicyEvaluation(gamma=0.5)
    policies = jnp.stack(
        (
            jnp.full((mdp.action_size, mdp.state_size), 1 / mdp.action_size),
            greedy_map.v(mdp, jnp.zeros(mdp.state_size), planner.gamma),
        )
    )
    v_vals = jnp.array([[0.0, 0.0], [1.0, -1.0]])
    init = chex.chexify(
        jax.jit(jax.vmap(planner.init, in_axes=(None, 0, 0))),
        async_check=False,
    )
    update = chex.chexify(
        jax.jit(jax.vmap(planner.update, in_axes=(None, 0))),
        async_check=False,
    )

    states = update(mdp, init(mdp, policies, v_vals))

    assert states.policy.shape == policies.shape
    assert states.v_val.shape == v_vals.shape


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


def test_deflated_value_iteration_matches_rank_one_recurrence() -> None:
    mdp = _two_state_mdp()
    planner = DeflatedValueIteration(gamma=0.5)
    v_val = jnp.array([1.0, -1.0])
    dist = jnp.array([0.25, 0.75])
    state = planner.init(mdp, v_val, dist)

    updated = planner.update(mdp, state)

    expected_w = v_val - planner.gamma * jnp.sum(dist * v_val)
    bellman_w = bellman_opt_op.v(mdp, expected_w, planner.gamma)
    next_w = bellman_w - planner.gamma * jnp.sum(dist * expected_w)
    expected_v = next_w + planner.gamma / (1 - planner.gamma) * jnp.sum(dist * next_w)
    assert jnp.allclose(state.w, expected_w)
    assert jnp.allclose(state.v_val, v_val)
    assert jnp.allclose(updated.w, next_w)
    assert jnp.allclose(updated.v_val, expected_v)


def test_deflated_value_iteration_preserves_greedy_sequence() -> None:
    mdp = _two_state_mdp()
    planner = DeflatedValueIteration(gamma=0.5)
    value_iteration = ValueIteration(gamma=0.5)
    state = planner.init(mdp)
    value_state = value_iteration.init(mdp)

    for _ in range(4):
        state = planner.update(mdp, state)
        value_state = value_iteration.update(mdp, value_state)
        difference = state.v_val - value_state.v_val

        assert jnp.allclose(difference, jnp.full_like(difference, difference[0]))
        assert jnp.array_equal(
            greedy_map.v(mdp, state.v_val, planner.gamma),
            greedy_map.v(mdp, value_state.v_val, value_iteration.gamma),
        )


def test_deflated_value_iteration_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    planner = DeflatedValueIteration(gamma=0.5)
    v_vals = jnp.array([[0.0, 0.0], [1.0, -1.0]])
    dists = jnp.array([[0.5, 0.5], [0.25, 0.75]])
    init = jax.jit(jax.vmap(planner.init, in_axes=(None, 0, 0)))
    update = chex.chexify(
        jax.jit(jax.vmap(planner.update, in_axes=(None, 0))),
        async_check=False,
    )

    states = update(mdp, init(mdp, v_vals, dists))

    assert states.w.shape == v_vals.shape
    assert states.v_val.shape == v_vals.shape
    assert states.dist.shape == dists.shape


def test_quasi_policy_iteration_matches_rank_one_update() -> None:
    mdp = _two_state_mdp()
    planner = QuasiPolicyIteration(gamma=0.5)
    v_val = jnp.array([1.0, -1.0])
    prior = jnp.array([[0.8, 0.3], [0.2, 0.7]])
    state = planner.init(mdp, v_val, prior)

    updated = planner.update(mdp, state)

    identity = jnp.eye(mdp.state_size)
    prior_resolvent = jnp.linalg.solve(identity - planner.gamma * prior.T, identity)
    bellman_v = bellman_opt_op.v(mdp, v_val, planner.gamma)
    policy = greedy_map.v(mdp, v_val, planner.gamma)
    policy_reward = jnp.sum(policy * jaxdp.reward.sa(mdp), axis=0)
    w = bellman_v - policy_reward - (planner.gamma * prior.T) @ v_val
    checked_w = prior_resolvent @ w
    u = v_val - jnp.mean(v_val)
    checked_u = prior_resolvent.T @ u
    gain = 1 / jnp.dot(u, v_val - checked_w)
    residual = v_val - bellman_v
    proposal = (
        v_val
        - prior_resolvent @ residual
        - gain * checked_w * jnp.dot(checked_u, residual)
    )
    proposal_bellman = bellman_opt_op.v(mdp, proposal, planner.gamma)
    proposal_residual = jnp.max(jnp.abs(proposal - proposal_bellman))
    bound = planner.gamma * state.bound
    accepted = proposal_residual <= bound
    expected = jnp.where(accepted, proposal, bellman_v)
    assert jnp.allclose(state.prior_resolvent, prior_resolvent)
    assert jnp.allclose(updated.gain, gain)
    assert jnp.array_equal(updated.accepted, accepted)
    assert jnp.allclose(updated.bound, bound)
    assert jnp.allclose(updated.v_val, expected)


def test_quasi_policy_iteration_safeguards_with_value_iteration() -> None:
    mdp = _two_state_mdp()
    planner = QuasiPolicyIteration(gamma=0.5)
    state = planner.init(mdp, jnp.array([1.0, -1.0]))
    state = replace(state, bound=jnp.zeros_like(state.bound))

    updated = planner.update(mdp, state)

    expected = bellman_opt_op.v(mdp, state.v_val, planner.gamma)
    assert not updated.accepted
    assert jnp.allclose(updated.v_val, expected)


def test_quasi_policy_iteration_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    planner = QuasiPolicyIteration(gamma=0.5)
    v_vals = jnp.array([[0.0, 0.0], [1.0, -1.0]])
    priors = jnp.array(
        [
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.8, 0.3], [0.2, 0.7]],
        ]
    )
    init = chex.chexify(
        jax.jit(jax.vmap(planner.init, in_axes=(None, 0, 0))),
        async_check=False,
    )
    update = chex.chexify(
        jax.jit(jax.vmap(planner.update, in_axes=(None, 0))),
        async_check=False,
    )

    states = update(mdp, init(mdp, v_vals, priors))

    assert states.v_val.shape == v_vals.shape
    assert states.prior.shape == priors.shape
    assert states.prior_resolvent.shape == priors.shape
    assert states.bound.shape == (2,)
    assert states.accepted.shape == (2,)
    assert jnp.all(jnp.isfinite(states.v_val))
    assert jnp.all(jnp.isfinite(states.gain))


def test_dynamic_boltzmann_value_iteration_matches_power_schedule() -> None:
    mdp = _two_state_mdp()
    planner = DynamicBoltzmannValueIteration(gamma=0.5, power=2.0)
    state = planner.init(mdp, jnp.array([1.0, -1.0]))

    first = planner.update(mdp, state)
    second = planner.update(mdp, first)

    first_q = jaxdp.reward.sa(mdp) + planner.gamma * jaxdp.trans_op.sa(mdp, state.v_val)
    first_policy = jax.nn.softmax(first_q - jnp.max(first_q, axis=0, keepdims=True), axis=0)
    second_q = jaxdp.reward.sa(mdp) + planner.gamma * jaxdp.trans_op.sa(mdp, first.v_val)
    second_centered_q = second_q - jnp.max(second_q, axis=0, keepdims=True)
    second_policy = jax.nn.softmax(4 * second_centered_q, axis=0)
    assert first.step == 1
    assert first.beta == 1
    assert jnp.allclose(first.policy, first_policy)
    assert jnp.allclose(first.v_val, jnp.sum(first_policy * first_q, axis=0))
    assert second.step == 2
    assert second.beta == 4
    assert jnp.allclose(second.policy, second_policy)
    assert jnp.allclose(second.v_val, jnp.sum(second_policy * second_q, axis=0))


def test_dynamic_boltzmann_value_iteration_approaches_hard_max() -> None:
    mdp = _two_state_mdp()
    planner = DynamicBoltzmannValueIteration(gamma=0.5, power=2.0)
    state = planner.init(mdp, jnp.array([1.0, -1.0]))
    state = replace(state, step=jnp.asarray(99, dtype=jnp.int32))

    updated = planner.update(mdp, state)

    q_val = jaxdp.reward.sa(mdp) + planner.gamma * jaxdp.trans_op.sa(mdp, state.v_val)
    assert updated.beta == 10_000
    assert jnp.allclose(updated.v_val, jnp.max(q_val, axis=0))


def test_dynamic_boltzmann_value_iteration_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    planner = DynamicBoltzmannValueIteration(gamma=0.5)
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
    assert states.policy.shape == (2, mdp.action_size, mdp.state_size)
    assert states.step.shape == (2,)
    assert states.beta.shape == (2,)
    assert jnp.all(jnp.isfinite(states.v_val))


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


def test_pid_value_iteration_matches_paper_recurrence() -> None:
    mdp = _two_state_mdp()
    planner = PIDValueIteration(
        gamma=0.5,
        kp=0.8,
        ki=0.3,
        kd=0.2,
        alpha=0.25,
        beta=0.75,
    )
    state = planner.init(mdp, jnp.array([1.0, -1.0]))

    for _ in range(2):
        bellman_v = bellman_opt_op.v(mdp, state.v_val, planner.gamma)
        residual = bellman_v - state.v_val
        expected_z = planner.beta * state.z_val + planner.alpha * residual
        expected_v = (
            state.v_val
            + planner.kp * residual
            + planner.ki * expected_z
            + planner.kd * (state.v_val - state.prev_v_val)
        )
        updated = planner.update(mdp, state)

        assert jnp.array_equal(updated.prev_v_val, state.v_val)
        assert jnp.allclose(updated.z_val, expected_z)
        assert jnp.allclose(updated.v_val, expected_v)
        state = updated


def test_pid_value_iteration_defaults_to_value_iteration() -> None:
    mdp = _two_state_mdp()
    pid = PIDValueIteration(gamma=0.5)
    value_iteration = ValueIteration(gamma=0.5)
    pid_state = pid.init(mdp)
    value_state = value_iteration.init(mdp)

    for _ in range(3):
        pid_state = pid.update(mdp, pid_state)
        value_state = value_iteration.update(mdp, value_state)

    assert jnp.allclose(pid_state.v_val, value_state.v_val)


def test_pid_value_iteration_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    planner = PIDValueIteration(gamma=0.5, ki=0.3, kd=0.2)
    v_vals = jnp.array([[0.0, 0.0], [1.0, -1.0]])
    init = jax.jit(jax.vmap(planner.init, in_axes=(None, 0)))
    update = chex.chexify(
        jax.jit(jax.vmap(planner.update, in_axes=(None, 0))),
        async_check=False,
    )

    states = update(mdp, init(mdp, v_vals))

    assert states.v_val.shape == v_vals.shape
    assert states.prev_v_val.shape == v_vals.shape
    assert states.z_val.shape == v_vals.shape


def test_anderson_value_iteration_matches_regularized_recurrence() -> None:
    mdp = _two_state_mdp()
    planner = AndersonValueIteration(gamma=0.5, memory=1, regularization=0.1)
    state = planner.init(mdp, jnp.zeros(mdp.state_size))

    updated = planner.update(mdp, state)

    residual = jnp.array([[1.5, 1.0], [2.0, 3.0]])
    gram = residual @ residual.T + 0.1 * jnp.eye(2)
    solved = jnp.linalg.solve(gram, jnp.ones(2))
    coeff = solved / jnp.sum(solved)
    expected = coeff[0] * jnp.array([3.5, 4.0]) + coeff[1] * jnp.array([2.0, 3.0])
    assert jnp.allclose(state.v_hist, jnp.array([[2.0, 3.0], [0.0, 0.0]]))
    assert jnp.allclose(state.bellman_hist, jnp.array([[3.5, 4.0], [2.0, 3.0]]))
    assert jnp.allclose(updated.coeff, coeff)
    assert jnp.allclose(updated.v_val, expected)
    assert jnp.allclose(updated.v_hist[0], expected)
    assert jnp.allclose(updated.v_hist[1], state.v_val)


def test_anderson_value_iteration_saturates_history() -> None:
    mdp = _two_state_mdp()
    planner = AndersonValueIteration(gamma=0.5, memory=2)
    state = planner.init(mdp)

    state = planner.update(mdp, state)
    assert state.count == 3

    state = planner.update(mdp, state)
    assert state.count == 3
    assert jnp.array_equal(state.v_hist[0], state.v_val)


def test_anderson_value_iteration_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    planner = AndersonValueIteration(gamma=0.5, memory=2)
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
    assert states.v_hist.shape == (2, planner.memory + 1, mdp.state_size)
    assert states.bellman_hist.shape == states.v_hist.shape
    assert states.count.shape == (2,)
    assert states.coeff.shape == (2, planner.memory + 1)


def test_safe_anderson_value_iteration_matches_type_one_update() -> None:
    mdp = _two_state_mdp()
    planner = SafeAndersonValueIteration(gamma=0.5, memory=2, theta=0.9)
    state = planner.init(mdp, jnp.zeros(mdp.state_size))

    step_vec = jnp.array([2.0, 3.0])
    residual_diff = jnp.array([0.5, 2.0])
    eta = jnp.sum(step_vec * residual_diff) / jnp.sum(step_vec**2)
    powell = (1 - planner.theta) / (1 - eta)
    y_tilde = powell * residual_diff - (1 - powell) * jnp.array([-2.0, -3.0])
    expected_left = step_vec - y_tilde
    expected_right = step_vec / jnp.sum(step_vec * y_tilde)
    assert jnp.allclose(state.v_val, jnp.array([2.0, 3.0]))
    assert jnp.allclose(state.residual, jnp.array([-1.5, -1.0]))
    assert jnp.allclose(state.s_hist[0], step_vec / jnp.linalg.norm(step_vec))
    assert jnp.allclose(state.h_left[0], expected_left)
    assert jnp.allclose(state.h_right[0], expected_right)
    assert state.count == 1
    assert state.accepted_count == 1
    assert state.step == 1
    assert state.accepted
    assert not state.restarted

    updated = planner.update(mdp, state)

    h_residual = state.residual + state.h_left[0] * jnp.sum(state.h_right[0] * state.residual)
    assert updated.accepted
    assert jnp.allclose(updated.v_val, state.v_val - h_residual)
    assert updated.count == 2
    assert updated.accepted_count == 2
    assert updated.step == 2


def test_safe_anderson_value_iteration_falls_back_to_vi() -> None:
    mdp = _two_state_mdp()
    planner = SafeAndersonValueIteration(gamma=0.5, memory=2, safeguard=0.01)
    state = planner.init(mdp)

    updated = planner.update(mdp, state)

    assert not updated.accepted
    assert updated.accepted_count == state.accepted_count
    assert jnp.allclose(updated.v_val, state.v_val - state.residual)


def test_safe_anderson_value_iteration_restarts_at_memory_limit() -> None:
    mdp = _two_state_mdp()
    planner = SafeAndersonValueIteration(gamma=0.5, memory=1)
    state = planner.init(mdp)

    updated = planner.update(mdp, state)

    assert updated.restarted
    assert updated.count == 1


def test_safe_anderson_value_iteration_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    planner = SafeAndersonValueIteration(gamma=0.5, memory=2)
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
    assert states.residual.shape == v_vals.shape
    assert states.s_hist.shape == (2, planner.memory, mdp.state_size)
    assert states.h_left.shape == states.s_hist.shape
    assert states.h_right.shape == states.s_hist.shape
    assert states.count.shape == (2,)
    assert states.accepted_count.shape == (2,)
    assert states.step.shape == (2,)
    assert states.accepted.shape == (2,)
    assert states.restarted.shape == (2,)


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
