"""Implementation of tabular Q-learning in a finite MDP with synchronous sampling.

    Disclaimer:
    This example is not intended to be imported. If you want to use this implementation,
    we suggest copying the source code. Although this approach may seem counterintuitive
    from a software development perspective, we find it more flexible for research purposes.

    This example will be updated!!!
"""

from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax.random as jrd
import jax
from nestedtuple import nestedtuple

import jaxdp
from jaxdp.learning.algorithms import q_learning
from jaxdp.learning.runner import train

# By default JAX set float types into float32. The line below enables
# float64 data type.
jax.config.update("jax_enable_x64", True)


@nestedtuple
class Args:
    """ Arguments of the training """
    seed: int = 42
    n_seeds: int = 10

    class update_fn:
        alpha: float = 0.1

    class train_loop:
        gamma: float = 0.99
        n_steps: int = 100
        eval_period: int = 10

    class value_init:
        minval: float = 0.0
        maxval: float = 1.0

    class mdp_init:
        p_slip: float = 0.15
        board: Tuple[str] = ("#####",
                             "#  @#",
                             "#  X#",
                             "#P  #",
                             "#####")


if __name__ == "__main__":
    args = Args()

    # Initiate the MDP and the Q values
    _train_key, value_key = jrd.split(jrd.PRNGKey(args.seed), 2)
    train_keys = jrd.split(_train_key, args.n_seeds)
    mdp = jaxdp.mdp.grid_world(**args.mdp_init._asdict())
    init_value = jrd.uniform(value_key, (mdp.action_size, mdp.state_size,),
                             dtype="float", **args.value_init._asdict())

    def update_fn(index, sample, value, learner_state, gamma):
        # Define learner function
        next_value = q_learning.synchronous.step(sample, value, gamma)
        return q_learning.update(value, next_value, alpha=args.update_fn.alpha), None

    # Train a policy for 10 different seeds (After JIT compiling the "batch" train function)
    jitted_batch_train = jax.jit(
        jax.vmap(
            partial(
                train.synchronous,
                learner_state=None,
                value_star=jnp.full_like(init_value, jnp.nan),
                target_policy_fn=lambda q, i: jaxdp.greedy_policy.q(q),
                update_fn=update_fn,
                **args.train_loop._asdict()
            ), in_axes=(None, None, 0))
    )
    metrics, value, learner_state = jitted_batch_train(
        init_value,
        mdp,
        train_keys
    )

    # TODO: Remove runner and implement it here
    # TODO: Save final_state
    # TODO: Make a rendering in possible
    # TODO: (Low priority) Add CLI for args
