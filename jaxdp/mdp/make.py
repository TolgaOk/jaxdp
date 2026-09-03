"""Named recipes for finite Markov decision processes."""

from collections.abc import Callable
from functools import partial

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
_KEY = jrd.key(42)

_Recipe = Callable[[], MDP]
_RECIPES: dict[str, _Recipe] = {
    "cliff-walking": cliff_walking_mdp,
    "delayed-reward": partial(
        delayed_reward_mdp,
        delay=2,
        action_size=2,
        reward_std=0.0,
        key=_KEY,
    ),
    "delayed-reward-long": partial(
        delayed_reward_mdp,
        delay=5,
        action_size=2,
        reward_std=0.0,
        key=_KEY,
    ),
    "delayed-reward-long-noisy": partial(
        delayed_reward_mdp,
        delay=5,
        action_size=2,
        reward_std=1.0,
        key=_KEY,
    ),
    "delayed-reward-noisy": partial(
        delayed_reward_mdp,
        delay=2,
        action_size=2,
        reward_std=1.0,
        key=_KEY,
    ),
    "forest": partial(forest_mdp, rotation=2),
    "forest-long": partial(forest_mdp, rotation=10),
    "four-rooms": partial(grid_world, board=_FOUR_ROOMS),
    "frozen-lake": partial(grid_world, board=_FROZEN_LAKE, p_slip=2 / 3),
    "frozen-lake-deterministic": partial(grid_world, board=_FROZEN_LAKE),
    "garnet": partial(
        garnet_mdp,
        key=_KEY,
        state_size=10,
        action_size=4,
        branch_size=2,
    ),
    "garnet-dense": partial(
        garnet_mdp,
        key=_KEY,
        state_size=50,
        action_size=5,
        branch_size=50,
    ),
    "garnet-large": partial(
        garnet_mdp,
        key=_KEY,
        state_size=300,
        action_size=10,
        branch_size=4,
        min_reward=-1.0,
        max_reward=1.0,
    ),
    "garnet-medium": partial(
        garnet_mdp,
        key=_KEY,
        state_size=50,
        action_size=5,
        branch_size=3,
    ),
    "graph": graph_mdp,
    "grid-world": partial(grid_world, board=_GRID_WORLD),
    "grid-world-slippery": partial(grid_world, board=_GRID_WORLD, p_slip=0.2),
    "sequential": partial(sequential_mdp, state_size=4),
    "sequential-long": partial(sequential_mdp, state_size=20),
    "tree": partial(tree_mdp, depth=2),
    "tree-deep": partial(tree_mdp, depth=5),
}


def make(name: str) -> MDP:
    """Construct a fixed MDP recipe by name.

    Stochastic recipes use a fixed key for reproducibility. Use their parameterized factories when
    independently sampled MDPs are required.

    Args:
        name: Registered recipe name.

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
    return recipe()


__all__ = ["make"]
