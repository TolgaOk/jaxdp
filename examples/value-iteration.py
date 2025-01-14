
""" Implementation of Value Iteration (VI) in grid-world.

    Disclaimer:
    This example is not intended to be imported. If you want to use this implementation,
    we suggest copying the source code. Although this approach may seem counterintuitive
    from a software development perspective, we find it more flexible for research purposes.

    This example will be updated!!!
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as jrd
from nestedtuple import nestedtuple

import jaxdp
from jaxdp.planning.algorithms import value_iteration
from jaxdp.planning.runner import train, no_update_state

# By default JAX set float types into float32. The line below enables
# float64 data type.
jax.config.update("jax_enable_x64", True)


@nestedtuple
class Args:
    """ Arguments of the training """
    seed: int = 42

    class train_loop:
        gamma: float = 0.99
        n_iterations: int = 100
        verbose: bool = False

    class value_init:
        minval: float = 0.0
        maxval: float = 1.0

    class mdp_init:
        p_slip: float = 0.1
        board: Tuple[str] = ["#####",
                             "#  @#",
                             "#  X#",
                             "#P  #",
                             "#####"]


if __name__ == "__main__":
    args = Args()
    # Initiate the MDP and the Q values
    key = jrd.PRNGKey(args.seed)
    mdp = jaxdp.mdp.grid_world(**args.mdp_init._asdict())
    init_value = jrd.uniform(key, (mdp.action_size, mdp.state_size,),
                             dtype="float", **args.value_init._asdict())

    # Define value update function and its initial state
    update_state = None
    update_fn = no_update_state(value_iteration.update.q)

    # Train a policy
    metrics, value, update_state = train(
        mdp=mdp,
        init_value=init_value,
        update_state=update_state,
        update_fn=update_fn,
        value_star=jnp.zeros_like(init_value),
        **args.train_loop._asdict()
    )

    # TODO: Save final_state
    # TODO: Make a rendering in possible
    # TODO: (Low priority) Add CLI for args
    # TODO: (Low priority) Remove train and implement it here
