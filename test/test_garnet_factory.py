import chex
import jax
import jax.numpy as jnp
import pytest

from jaxdp.mdp import MDP
from jaxdp.mdp.garnet import garnet_mdp


def test_garnet_has_the_requested_sparse_branching_and_reward_bounds() -> None:
    mdp = garnet_mdp(
        jax.random.key(0),
        state_size=7,
        action_size=3,
        branch_size=2,
        min_reward=-2.0,
        max_reward=3.0,
    )

    assert mdp.transition.shape == (3, 7, 7)
    assert mdp.reward.shape == (3, 7, 7)
    assert jnp.allclose(jnp.sum(mdp.transition, axis=-2), 1)
    assert jnp.all(jnp.sum(mdp.transition > 0, axis=-2) == 2)
    assert jnp.all(mdp.reward >= -2.0)
    assert jnp.all(mdp.reward <= 3.0)
    assert jnp.allclose(mdp.initial, jnp.full(7, 1 / 7))


def test_garnet_is_reproducible_and_composes_with_jit_and_vmap() -> None:
    def create_one(key: chex.PRNGKey) -> MDP:
        return garnet_mdp(
            key,
            state_size=5,
            action_size=2,
            branch_size=3,
        )

    create = chex.chexify(jax.jit(create_one), async_check=False)
    create_batch = chex.chexify(jax.jit(jax.vmap(create_one)), async_check=False)
    key = jax.random.key(1)
    first = create(key)
    second = create(key)
    batch = create_batch(jax.random.split(key, 4))

    assert jax.tree.all(jax.tree.map(jnp.array_equal, first, second))
    assert batch.transition.shape == (4, 2, 5, 5)


def test_garnet_caps_branching_at_the_state_count() -> None:
    mdp = garnet_mdp(
        jax.random.key(0),
        state_size=2,
        action_size=3,
        branch_size=5,
    )

    assert jnp.all(jnp.sum(mdp.transition > 0, axis=-2) == 2)


@pytest.mark.parametrize(
    ("state_size", "action_size", "branch_size", "min_reward", "max_reward", "message"),
    [
        (0, 2, 1, 0.0, 1.0, "state_size"),
        (2, 0, 1, 0.0, 1.0, "action_size"),
        (2, 2, 0, 0.0, 1.0, "branch_size"),
        (2, 2, 1, 2.0, 1.0, "min_reward"),
    ],
)
def test_garnet_rejects_invalid_parameters(
    state_size: int,
    action_size: int,
    branch_size: int,
    min_reward: float,
    max_reward: float,
    message: str,
) -> None:
    with pytest.raises(AssertionError, match=message):
        garnet_mdp(
            jax.random.key(0),
            state_size,
            action_size,
            branch_size,
            min_reward,
            max_reward,
        )
