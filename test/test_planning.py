import chex
import jax
import jax.numpy as jnp
import pytest

from jaxdp.mapping import greedy_map
from jaxdp.mdp import MDP
from jaxdp.operator import bellman_opt_op
from jaxdp.planning import PolicyIteration, ValueIteration, policy_eval


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


def test_value_iteration_applies_fixed_updates() -> None:
    mdp = _two_state_mdp()
    v_val = jnp.array([4.0, 8.0])
    bellman = bellman_opt_op

    once = bellman.v(mdp, v_val, 0.5)
    twice = bellman.v(mdp, once, 0.5)

    assert jnp.allclose(ValueIteration().v(mdp, v_val, 0.5), once)
    assert jnp.allclose(ValueIteration(step=2).v(mdp, v_val, 0.5), twice)


def test_policy_iteration_evaluates_then_improves() -> None:
    mdp = _two_state_mdp()
    policy = jnp.full((2, 2), 0.5)
    evaluation = policy_eval

    expected = greedy_map.q(evaluation.q(mdp, policy, 0.5))
    actual = PolicyIteration().policy(mdp, policy, 0.5)

    assert jnp.array_equal(actual, expected)
    assert jnp.allclose(
        PolicyIteration().q(mdp, policy, 0.5),
        evaluation.q(mdp, actual, 0.5),
    )
    assert jnp.allclose(
        PolicyIteration().v(mdp, policy, 0.5),
        evaluation.v(mdp, actual, 0.5),
    )


@pytest.mark.parametrize("planner", [ValueIteration(step=-1), PolicyIteration(step=-1)])
def test_planning_rejects_negative_step(planner: ValueIteration | PolicyIteration) -> None:
    mdp = _two_state_mdp()

    with pytest.raises(AssertionError, match="step"):
        if isinstance(planner, ValueIteration):
            planner.v(mdp, jnp.zeros(2), 0.5)
        else:
            planner.policy(mdp, jnp.full((2, 2), 0.5), 0.5)


def test_planning_composes_with_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    values = jnp.array([[4.0, 8.0], [8.0, 4.0]])
    iterate = ValueIteration(step=2)
    policies = jnp.full((2, 2, 2), 0.5)
    improve = PolicyIteration(step=2)

    iterated = chex.chexify(
        jax.jit(jax.vmap(lambda v_val: iterate.v(mdp, v_val, 0.5))),
        async_check=False,
    )(values)
    improved = chex.chexify(
        jax.jit(jax.vmap(lambda policy: improve.policy(mdp, policy, 0.5))),
        async_check=False,
    )(policies)

    assert iterated.shape == values.shape
    assert improved.shape == policies.shape
