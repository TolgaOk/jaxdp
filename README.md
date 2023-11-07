# Jaxdp

> :warning: Under Development

**Jaxdp** is a Python package that provides simple and functional implementation of dynamic programming (DP) algorithms for discrete state-action Markov decision processes (MDP) within the <img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" width = 24px alt="logo"></img> ecosystem. Using the JAX transformations, you can accelerate (even using GPUs) DP algorithms by running multiple MDPs and values in a vectorized form.

```Python
from jaxdp import ValueIteration as VI
from jax import vmap

...

# Regular VI step
VI.step(value, mdp)

# Multiple values VI step
vi_step = vmap(VI.step, in_axes=(0, None), out_axes=0)(values, mdp)

# Multiple values multiple MDPs VI step
vmap(vi_step, in_axes=(None, 0), out_axes=0)(values, mdps)
```

> List of Algorithms

| Algorithms       | Sync             | Async     |
|:----------------:|:----------------:|:---------:|
|  VI              |:heavy_check_mark:|:x:        |
|  PI              |:x:               |:x:        |
|  Accelerated VI  |:x:               |:x:        |
|  Momentum VI     |:x:               |:x:        |
|  Momentum QL     |:x:               |:x:        |
|  Relaxed VI      |:x:               |:x:        |
|  Speedy QL       |:x:               |:x:        |
|  Zap QL          |:x:               |:x:        |

> Typehint

Jaxdp extensively uses typehints and annotations from [jaxtyping](https://github.com/google/jaxtyping#jaxtyping).

> MDP -> Pytree structure

Jaxdp provides a Pytree definition for MDPs, which allows Jaxdp to vectorize different MDPs for a DP step.

```Python
import jax.numpy as jnp
import jax.tree_util
from jaxdp import MDPtree


mdp_1 = MDPtree(
    transition=...,
    terminal=...,
    initial=...,
    reward=...,
)

...

# Stacked MDPs
mdps = jax.tree_util.tree_map(lambda *mdp: jnp.stack(mdp), (mdp_1, mdp_2))
```

## Installation

Recommended: Python 3.9+

Install the package in development mode via:

```bash
pip install -r requirements.txt
pip install -e .
```
