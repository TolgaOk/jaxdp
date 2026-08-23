from collections.abc import Sequence

import jax.numpy as jnp
import pytest

from jaxdp.mdp.grid_world import grid_world


def test_grid_world_uses_row_major_dynamics_and_terminal_rewards() -> None:
    mdp = grid_world(("#####", "#P @#", "#####"))

    assert mdp.state_size == 3
    assert jnp.array_equal(mdp.initial, jnp.array([1.0, 0.0, 0.0]))
    assert jnp.array_equal(mdp.terminal, jnp.array([0.0, 0.0, 1.0]))
    assert mdp.transition[1, 1, 0] == 1
    assert mdp.transition[1, 2, 1] == 1
    assert jnp.all(mdp.transition[:, 2, 2] == 1)
    assert mdp.reward[1, 1, 2] == 1
    assert jnp.all(mdp.reward[:, 2, :] == 0)


def test_grid_world_splits_slip_between_perpendicular_actions() -> None:
    mdp = grid_world(
        (
            "#####",
            "#   #",
            "# P #",
            "#  @#",
            "#####",
        ),
        p_slip=0.2,
    )

    assert jnp.allclose(
        mdp.transition[2, :, 4],
        jnp.array([0.0, 0.8, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0]),
    )


def test_absorbing_reward_cell_is_nonterminal() -> None:
    mdp = grid_world(("#####", "#P= #", "#####"))

    assert not mdp.terminal[1]
    assert mdp.transition[1, 1, 0] == 1
    assert jnp.all(mdp.transition[:, 1, 1] == 1)
    assert mdp.reward[1, 0, 1] == 1
    assert jnp.all(mdp.reward[:, 1, 1] == 1)


@pytest.mark.parametrize(
    ("board", "p_slip", "message"),
    [
        ((), 0.0, "nonempty"),
        (("###", "#P"), 0.0, "equal length"),
        (("###", "#P?", "###"), 0.0, "invalid cells"),
        (("###", "# #", "###"), 0.0, "exactly one"),
        (("####", "#PP#", "####"), 0.0, "exactly one"),
        (("###", "#P#", "###"), -0.1, "p_slip"),
        (("###", "#P#", "###"), 1.1, "p_slip"),
        (("###", "#P#", "###"), float("inf"), "p_slip"),
        (("###", "#P#", "###"), float("nan"), "p_slip"),
    ],
)
def test_grid_world_rejects_invalid_inputs(
    board: Sequence[str],
    p_slip: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        grid_world(board, p_slip)
