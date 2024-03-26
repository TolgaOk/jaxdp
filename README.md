# Jaxdp

> :warning: Under Development

**Jaxdp** is a Python package that provides simple functional implementation of dynamic programming (DP) algorithms for discrete state-action Markov decision processes (MDP) within the <img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" width = 24px alt="logo"></img> ecosystem. Using the JAX transformations, you can accelerate (even using GPUs) DP algorithms by running multiple MDPs, initial values etc. in a vectorized form.

```Python
from jaxdp.iterations.iteration import value_iteration_update
from jax import vmap

...

# Regular VI step
regular_vi_step = value_iteration_update(value, mdp, gamma)

# Multiple values VI step
mv_vi_step = vmap(value_iteration_update, in_axes=(0, None, None))(values, mdp, gamma)

# Multiple values multiple MDPs VI step
mvmm_vi_step = vmap(mv_vi_step, in_axes=(None, 0, None))(values, mdps, gamma)
```

### List of Algorithms

|Iteration Algorithms  |                  |
|:--------------------:|:----------------:|
|  VI                  |:heavy_check_mark:|
|  PI                  |:heavy_check_mark:|
|  Accelerated VI      |:x:               |
|  Relaxed VI          |:x:               |

|Learning Algorithms |Sync sampled      |Async sampled     |
|:------------------:|:----------------:|:----------------:|
|  SARSA             |:x:               |:x:               |
|  TD($\lambda$)     |:x:               |:heavy_check_mark:|
|  Q-learning (QL)   |:heavy_check_mark:|:heavy_check_mark:|
|  Momentum QL       |:x:               |:x:               |
|  Speedy QL         |:heavy_check_mark:|:x:               |
|  Zap QL            |:heavy_check_mark:|:x:               |


### Typehint

Jaxdp extensively uses typehints and annotations from [jaxtyping](https://github.com/google/jaxtyping#jaxtyping).

### Vectorize MDPs

Jaxdp provides a Pytree definition for MDPs, which allows Jax to vectorize different MDPs to be used in a DP step.

```Python
import jax.numpy as jnp
import jax.tree_util

from jaxdp.mdp.garnet import garnet_mdp as make_garnet


n_mdp = 10
key = jax.random.PRNGKey(42)

mdps = [make_garnet(state_size=1000, action_size=10, key=key,
                    branch_size=4, min_reward=-1, max_reward=1)
        for key in jrd.split(key, n_mdp)]

# Stacked MDP
stacked_mdp = jax.tree_map(lambda *mdps: jnp.stack(mdps), *mdps)

```

## Installation

Recommended: Python 3.9+

Install the package in development mode via:

```bash
pip install -r requirements.txt
pip install -e .
```
