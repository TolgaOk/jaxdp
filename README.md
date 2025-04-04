# Jaxdp


**Jaxdp** is a Python package that provides a functional implementation of algorithms and commonly used functions for dynamic programming (DP) on finite state-action Markov decision processes (MDPs) within the <img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" width = 24px alt="logo"></img> ecosystem. By leveraging JAX transformations, you can accelerate DP algorithms (even using GPUs) by running multiple MDPs, initial values, seeds, etc., in a vectorized manner.

# Vectorization

The functions in **jaxdp** are fully compatible with <img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" width = 24px alt="logo"></img> transformations.  They are stateless, with memory explicitly provided to the functions. This design pattern avoids hiding information at the cost of being more verbose. Therefore, **jaxdp** relies on [currying](https://docs.python.org/3/library/functools.html) whenever possible.


## MDPs

In jaxdp, MDPs are Pytrees and are therefore compatible with JAX transformations.

```Python
import jax.numpy as jnp
import jax.tree_util

from jaxdp.mdp.garnet import garnet_mdp as make_garnet


n_mdp = 8
key = jax.random.PRNGKey(42)

# List of random MDPs with different seeds
mdps = [make_garnet(state_size=300, action_size=10, key=key,
                    branch_size=4, min_reward=-1, max_reward=1)
        for key in jrd.split(key, n_mdp)]

# Stacked MDP
stacked_mdp = jax.tree_map(lambda *mdps: jnp.stack(mdps), *mdps)
```

Once stacked, an MDP can be provided to a vectorized function such as training loop.

```Python
> mdps[0].transition.shape
> (10, 300, 300)

> stacked_mdp.transition.shape
> (8, 10, 300, 300)
```

> [!warning]
> The shapes of the components in the list of MDPs need to match to vectorize. Hence, we may not have a vectorized MDP with varying action or state sizes.

# Installation

Recommended: Python 3.9+

Install the package in development mode via:

```bash
pip install -r requirements.txt
pip install -e .
```
