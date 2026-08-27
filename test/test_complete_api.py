import chex
import jax
import jax.numpy as jnp

from jaxdp.distribution import Occupancy
from jaxdp.mdp import MDP
from jaxdp.operator import TransOp
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
    return MDP(
        transition=transition,
        reward=reward,
        initial=initial,
        terminal=terminal,
    )


def test_value_policies_apply_one_step_lookahead() -> None:
    mdp = _two_state_mdp()
    value = jnp.array([4.0, 8.0])
    gamma = 0.5
    reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
    q_value = reward + gamma * TransOp().sa(mdp, value)
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


def test_state_and_action_occupancies_are_consistent() -> None:
    mdp = _two_state_mdp()
    policy = jnp.full((2, 2), 0.5)

    initial_v = Occupancy(steps=0).v(mdp, policy)
    initial_q = Occupancy(steps=0).q(mdp, policy)
    next_v = Occupancy(steps=1).v(mdp, policy)
    next_q = Occupancy(steps=1).q(mdp, policy)

    assert jnp.allclose(initial_v, mdp.initial)
    assert jnp.allclose(initial_q, policy * mdp.initial)
    assert jnp.allclose(next_v, jnp.array([0.5, 0.5]))
    assert jnp.allclose(jnp.sum(next_q, axis=0), next_v)


def test_completed_api_supports_jit_and_vmap() -> None:
    mdp = _two_state_mdp()
    soft = Soft(temperature=2.0)
    values = jnp.array([[4.0, 8.0], [8.0, 4.0]])

    policies = chex.chexify(
        jax.jit(jax.vmap(lambda value: soft.v(mdp, value, 0.5))),
        async_check=False,
    )(values)
    state_distribution = chex.chexify(
        jax.jit(lambda policy: Occupancy(steps=3).v(mdp, policy)),
        async_check=False,
    )(
        jnp.full((2, 2), 0.5)
    )

    assert policies.shape == (2, mdp.action_size, mdp.state_size)
    assert jnp.allclose(state_distribution, jnp.array([0.5, 0.5]))
