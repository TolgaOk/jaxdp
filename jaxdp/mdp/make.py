"""Named recipes for finite Markov decision processes."""

from collections.abc import Callable

import jax
import jax.random as jrd

from jaxdp.mdp.cliff_walking import cliff_walking_mdp
from jaxdp.mdp.delayed_reward import delayed_reward_mdp
from jaxdp.mdp.forest_mdp import forest_mdp
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.mdp.grid_world import grid_world
from jaxdp.mdp.mdp import MDP
from jaxdp.mdp.sequential import sequential_mdp
from jaxdp.mdp.simple_graph import graph_mdp
from jaxdp.mdp.tree_mdp import tree_mdp

_GRID_WORLD = (
    "#####",
    "#  @#",
    "# #X#",
    "#P  #",
    "#####",
)
_FOUR_ROOMS = (
    "#############",
    "#     #    @#",
    "#     #     #",
    "#           #",
    "#     #     #",
    "#     #     #",
    "## #### #####",
    "#     #     #",
    "#     #     #",
    "#           #",
    "#     #     #",
    "#P    #     #",
    "#############",
)
_FROZEN_LAKE = (
    "######",
    "#P   #",
    "# H H#",
    "#   H#",
    "#H  @#",
    "######",
)

_Recipe = Callable[[jax.Array], MDP]
_RECIPES: dict[str, _Recipe] = {
    "cliff-walking": lambda key: cliff_walking_mdp(),
    "delayed-reward": lambda key: delayed_reward_mdp(
        delay=2,
        action_size=2,
        reward_std=0.0,
        key=key,
    ),
    "delayed-reward-long": lambda key: delayed_reward_mdp(
        delay=5,
        action_size=2,
        reward_std=0.0,
        key=key,
    ),
    "delayed-reward-long-noisy": lambda key: delayed_reward_mdp(
        delay=5,
        action_size=2,
        reward_std=1.0,
        key=key,
    ),
    "delayed-reward-noisy": lambda key: delayed_reward_mdp(
        delay=2,
        action_size=2,
        reward_std=1.0,
        key=key,
    ),
    "forest": lambda key: forest_mdp(rotation=2),
    "forest-long": lambda key: forest_mdp(rotation=10),
    "four-rooms": lambda key: grid_world(board=_FOUR_ROOMS),
    "frozen-lake": lambda key: grid_world(board=_FROZEN_LAKE, p_slip=2 / 3),
    "frozen-lake-deterministic": lambda key: grid_world(board=_FROZEN_LAKE),
    "garnet": lambda key: garnet_mdp(
        key=key,
        state_size=10,
        action_size=4,
        branch_size=2,
    ),
    "garnet-dense": lambda key: garnet_mdp(
        key=key,
        state_size=50,
        action_size=5,
        branch_size=50,
    ),
    "garnet-large": lambda key: garnet_mdp(
        key=key,
        state_size=300,
        action_size=10,
        branch_size=4,
        min_reward=-1.0,
        max_reward=1.0,
    ),
    "garnet-medium": lambda key: garnet_mdp(
        key=key,
        state_size=50,
        action_size=5,
        branch_size=3,
    ),
    "graph": lambda key: graph_mdp(),
    "grid-world": lambda key: grid_world(board=_GRID_WORLD),
    "grid-world-slippery": lambda key: grid_world(board=_GRID_WORLD, p_slip=0.2),
    "sequential": lambda key: sequential_mdp(state_size=4),
    "sequential-long": lambda key: sequential_mdp(state_size=20),
    "tree": lambda key: tree_mdp(depth=2),
    "tree-deep": lambda key: tree_mdp(depth=5),
}


def make(name: str, *, key: jax.Array | None = None) -> MDP:
    """Construct a named MDP recipe with optional random generation.

    Args:
        name: Registered recipe name, fixed when using ``jax.jit`` or ``jax.vmap``.
        key: Key for model generation. Defaults to ``jax.random.key(42)`` when omitted or
            ``None``. Recipes without random generation ignore it.

    Returns:
        Constructed finite MDP.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    try:
        recipe = _RECIPES[name]
    except KeyError:
        names = ", ".join(_RECIPES)
        raise ValueError(f"unknown MDP {name!r}; choose from: {names}") from None
    return recipe(jrd.key(42) if key is None else key)


__all__ = ["make"]
