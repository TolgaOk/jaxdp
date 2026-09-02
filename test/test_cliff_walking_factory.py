import jax.numpy as jnp
import pytest

from jaxdp.mdp.cliff_walking import cliff_walking_mdp


def test_cliff_walking_resets_on_cliff_entry_and_terminates_at_goal() -> None:
    mdp = cliff_walking_mdp()

    initial = 36
    goal = 37
    assert mdp.transition.shape == (4, 38, 38)
    assert mdp.initial[initial] == 1
    assert mdp.terminal[goal] == 1

    assert mdp.transition[1, initial, initial] == 1
    assert mdp.reward[1, initial, initial] == -100
    assert mdp.transition[0, initial, initial] == 1
    assert mdp.reward[0, initial, initial] == -1
    assert mdp.transition[2, 24, initial] == 1
    assert mdp.reward[2, initial, 24] == -1

    assert mdp.transition[0, initial, 29] == 1
    assert mdp.reward[0, 29, initial] == -100
    assert mdp.transition[0, goal, 35] == 1
    assert mdp.reward[0, 35, goal] == -1
    assert jnp.all(mdp.transition[:, goal, goal] == 1)
    assert jnp.all(mdp.reward[:, goal] == 0)


@pytest.mark.parametrize(
    ("row_size", "column_size", "message"),
    [
        (1, 12, "row_size"),
        (4, 2, "column_size"),
    ],
)
def test_cliff_walking_rejects_small_grids(
    row_size: int,
    column_size: int,
    message: str,
) -> None:
    with pytest.raises(AssertionError, match=message):
        cliff_walking_mdp(row_size, column_size)
