import jax
import jax.numpy as jnp
import pytest

from jaxdp.mdp import MDP, delayed_reward_mdp


def test_delayed_reward_mdp_arrays() -> None:
    mdp = delayed_reward_mdp(2, 2, 0.0, jax.random.key(0))

    assert isinstance(mdp, MDP)
    assert mdp.transition.shape == (2, 7, 7)
    assert mdp.reward.shape == mdp.transition.shape
    assert jnp.allclose(mdp.transition.sum(axis=-2), 1.0)
    assert jnp.array_equal(mdp.initial, jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert jnp.array_equal(mdp.terminal, jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]))


def test_delayed_reward_mdp_places_rewards_on_reachable_leaf_transitions() -> None:
    mdp = delayed_reward_mdp(2, 2, 0.0, jax.random.key(0))
    expected = (
        jnp.zeros((2, 7, 7))
        .at[0, 1, 3]
        .set(1.0)
        .at[1, 1, 4]
        .set(1.0)
        .at[0, 2, 5]
        .set(-1.0)
        .at[1, 2, 6]
        .set(-1.0)
    )

    assert jnp.array_equal(mdp.reward, expected)
    assert jnp.array_equal(
        jnp.einsum("asx,axs->as", mdp.reward, mdp.transition),
        expected.sum(axis=-1),
    )


def test_delayed_reward_mdp_handles_degenerate_trees() -> None:
    no_delay = delayed_reward_mdp(0, 2, 0.0, jax.random.key(0))
    one_action = delayed_reward_mdp(3, 1, 0.0, jax.random.key(0))

    assert jnp.array_equal(no_delay.transition, jnp.ones((2, 1, 1)))
    assert jnp.array_equal(no_delay.reward, jnp.zeros((2, 1, 1)))
    assert jnp.array_equal(no_delay.terminal, jnp.ones(1))
    assert one_action.transition.shape == (1, 4, 4)
    assert one_action.reward[0, 2, 3] == 1.0


@pytest.mark.parametrize(
    ("delay", "action_size", "reward_std", "message"),
    [
        (-1, 2, 0.0, "delay"),
        (1, 0, 0.0, "action_size"),
        (1, 2, -0.1, "reward_std"),
        (1, 2, jnp.inf, "reward_std"),
    ],
)
def test_delayed_reward_mdp_rejects_invalid_parameters(
    delay: int,
    action_size: int,
    reward_std: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        delayed_reward_mdp(delay, action_size, reward_std, jax.random.key(0))
