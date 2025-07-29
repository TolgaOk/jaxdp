# jaxdp

**jaxdp** is a Python package providing functional implementations of dynamic programming (DP) algorithms for finite state-action Markov decision processes (MDPs) within the <img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" width = 24px alt="logo"></img> ecosystem. By leveraging JAX transformations, you can accelerate DP algorithms (including GPU acceleration) through vectorized execution across multiple MDP instances, initial values, and parameters.

## Vectorization

**jaxdp** functions are fully compatible with JAX transformations. They are stateless with memory explicitly provided to functions.

### Algorithm Example

The `examples` directory contains implementations and benchmarks of planning algorithms using **jaxdp**. Below is a code snippet for [Momentum accelerated Value Iteration](https://arxiv.org/pdf/1905.09963):

```python
"""
◈─────────────────────────────────────────────────────────────────────────◈
Momentum accelerated Value Iteration.
◈─────────────────────────────────────────────────────────────────────────◈
"""
@struct.dataclass
class State:
    q_val: jnp.ndarray
    prev_q_val: jnp.ndarray
    gamma: jnp.ndarray
    beta: jnp.ndarray
    alpha: jnp.ndarray


def update(s: State, mdp: MDP, step: int) -> State:
    diff = s.q_val - s.prev_q_val
    b_residual = jaxdp.bellman_optimality_operator.q(mdp, s.q_val, s.gamma) - s.q_val
    next_q = s.q_val + s.alpha * b_residual + s.beta * diff
    
    return s.replace(q_val=next_q, prev_q_val=s.q_val)
```

You can vectorize the update function to run across:

- Multiple initial **values**
- Multiple **gamma** or **beta** values  
- Multiple **MDP** instances

Example for multiple gamma values using `jax.vmap`:


```python
init_state = State(
    q_val=init_q_vals,
    prev_q_val=init_q_vals,
    gamma=jnp.array([0.9, 0.95, 0.99, 0.999]),
    beta=0.99,
    alpha=0.1
)

final_state, all_states = jax.lax.scan(
    jax.vmap(                     # vmapped update function
        lambda s, ix: (update(s, mdp, ix), s),
        in_axes=(State(0, 0, 0, None, None), None)
    ),    
    init_state,                   # initial state
    jnp.arange(100)               # Number of iterations
)
```

### MDPs

In jaxdp, MDPs are PyTrees and therefore compatible with JAX transformations.

```python
import jax
import jax.numpy as jnp
from jaxdp.mdp.garnet import garnet_mdp as make_garnet

n_mdp = 8
key = jax.random.PRNGKey(42)

# List of random MDPs with different seeds
mdps = [make_garnet(state_size=300, action_size=10, key=key,
                    branch_size=4, min_reward=-1, max_reward=1)
        for key in jax.random.split(key, n_mdp)]

# Stacked MDP
stacked_mdp = jax.tree_map(lambda *mdps: jnp.stack(mdps), *mdps)
```

Once stacked, MDPs can be provided to vectorized functions:

```Python
> mdps[0].transition.shape
> (10, 300, 300)

> stacked_mdp.transition.shape
> (8, 10, 300, 300)
```

> **Warning:** MDP components must have matching shapes for vectorization. Variable action or state sizes are not supported.

## Installation

Requires Python 3.11+

```bash
pip install -r requirements.txt
pip install -e .
```
