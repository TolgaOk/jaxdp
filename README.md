# Jaxdp

> :warning: Under Development

**Jaxdp** is a Python package that provides simple and functional implementation of dynamic programming (DP) algorithms for discrete state-action Markov decision processes (MDP) within the <img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" width = 24px alt="logo"></img> ecosystem. Using the JAX transformations, you can accelerate (even using GPUs) DP algorithms by running multiple MDPs and values in a vectorized form.

```Python
from jaxdp import ValueIteration as VI
from jax import vmap

...

# Regular VI step
regular_vi_step = VI.step_sync(value, mdp)

# Multiple values VI step
mv_vi_step = vmap(VI.step_sync, in_axes=(0, None), out_axes=0)(values, mdp)

# Multiple values multiple MDPs VI step
mvmm_vi_step = vmap(mv_vi_step, in_axes=(None, 0), out_axes=0)(values, mdps)
```

> List of Algorithms

|Algorithms        |Exact             |Sync sampled      |Async sampled     |
|:----------------:|:----------------:|:----------------:|:----------------:|
|  VI              |:heavy_check_mark:|:heavy_minus_sign:|:heavy_minus_sign:|
|  PI              |:heavy_check_mark:|:heavy_minus_sign:|:heavy_minus_sign:|
|  Q Learning      |:heavy_minus_sign:|:heavy_check_mark:|:heavy_check_mark:|
|  SARSA           |:heavy_minus_sign:|:heavy_check_mark:|:heavy_check_mark:|
|  Accelerated VI  |:x:               |:x:               |:x:               |
|  Momentum VI     |:x:               |:x:               |:x:               |
|  Momentum QL     |:x:               |:x:               |:x:               |
|  Relaxed VI      |:x:               |:x:               |:x:               |
|  Speedy QL       |:x:               |:x:               |:x:               |
|  Zap QL          |:x:               |:x:               |:x:               |

> Typehint

Jaxdp extensively uses typehints and annotations from [jaxtyping](https://github.com/google/jaxtyping#jaxtyping).

> MDP -> Pytree structure

Jaxdp provides a Pytree definition for MDPs, which allows Jaxdp to vectorize different MDPs for a DP step.

```Python
import jax.numpy as jnp
import jax.tree_util
from jaxdp.mdp import MDP


mdp_1 = MDP(
    transition=...,
    terminal=...,
    initial=...,
    reward=...,
)

...

# Stacked MDPs
mdps = jax.tree_util.tree_map(lambda *mdps: jnp.stack(mdps), mdp_1, mdp_2)
```

## Installation

Recommended: Python 3.9+

Install the package in development mode via:

```bash
pip install -r requirements.txt
pip install -e .
```
