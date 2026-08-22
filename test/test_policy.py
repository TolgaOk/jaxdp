from dataclasses import FrozenInstanceError, is_dataclass

import jax
import jax.numpy as jnp
import pytest

from jaxdp.base import to_state_action_value
from jaxdp.mdp import MDP
from jaxdp.policy import EpsilonGreedy, Greedy, Policy, Soft


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


def _apply_q(policy: Policy, value: jax.Array) -> jax.Array:
    return policy.q(value)


def test_policy_components_share_q_v_contract() -> None:
    mdp = _two_state_mdp()
    value = jnp.array([4.0, 8.0])
    q_value = to_state_action_value(mdp, value, gamma=0.5)
    policies: tuple[Policy, ...] = (
        Greedy(),
        Soft(temperature=2.0),
        EpsilonGreedy(epsilon=0.2),
    )

    for policy in policies:
        assert jnp.allclose(policy.v(mdp, value, gamma=0.5), policy.q(q_value))


def test_policy_components_match_their_definitions() -> None:
    value = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    greedy = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    assert jnp.allclose(_apply_q(Greedy(), value), greedy)
    assert jnp.allclose(Soft(temperature=2.0).q(value), jax.nn.softmax(value / 2.0, axis=0))
    assert jnp.allclose(EpsilonGreedy(epsilon=0.2).q(value), 0.8 * greedy + 0.1)


@pytest.mark.parametrize("temperature", [0.0, -1.0, jnp.inf, jnp.nan])
def test_soft_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        Soft(temperature=temperature)


@pytest.mark.parametrize("epsilon", [-0.1, 1.1, jnp.inf, jnp.nan])
def test_epsilon_greedy_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedy(epsilon=epsilon)


def test_policy_components_are_immutable_dataclasses() -> None:
    policy = Soft(temperature=2.0)

    assert is_dataclass(policy)
    attribute = "temperature"
    with pytest.raises(FrozenInstanceError):
        setattr(policy, attribute, 1.0)


def test_policy_components_support_jit_and_vmap() -> None:
    policy = Soft(temperature=2.0)
    values = jnp.array(
        [
            [[3.0, 1.0], [1.0, 2.0]],
            [[1.0, 3.0], [2.0, 1.0]],
        ]
    )

    result = jax.jit(jax.vmap(policy.q))(values)

    assert result.shape == values.shape
    assert jnp.allclose(result.sum(axis=1), 1.0)
