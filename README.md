# jaxdp

**`jaxdp`** provides functional dynamic-programming algorithms for finite state-action Markov
decision processes in JAX. Its components support accelerated and vectorized execution across MDP
instances, initial values, and parameters.

## Vectorization

**`jaxdp`** components are compatible with JAX transformations. Iterative planners keep dynamic
values in explicit `State` pytrees and expose one update at a time.

Chex validates array shapes and numerical values. Shape assertions run during ordinary JAX
tracing. Numerical validation failures raise `AssertionError`. Value assertions require
`chex.chexify` outside the outermost transformed function.

```python
import chex
import jax
import jaxdp


def evaluate_policy(policy):
    return jaxdp.policy_eval.v(mdp, policy, 0.99)


checked_evaluate = chex.chexify(
    jax.jit(jax.vmap(evaluate_policy)),
    async_check=False,
)
v_vals = checked_evaluate(policies)

planner = jaxdp.ValueIteration(gamma=0.99)
state = planner.init(mdp)
checked_update = chex.chexify(jax.jit(planner.update), async_check=False)
state = checked_update(mdp, state)
```

### MDPs

In `jaxdp`, MDPs are PyTrees and therefore compatible with JAX transformations.

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
stacked_mdp = jax.tree.map(lambda *mdps: jnp.stack(mdps), *mdps)
```

Once stacked, MDPs can be provided to vectorized functions:

```Python
> mdps[0].transition.shape
> (10, 300, 300)

> stacked_mdp.transition.shape
> (8, 10, 300, 300)
```

> [!Warning]
> MDP components must have matching shapes for vectorization. Variable action or state sizes are
> not supported.

## Installation

Requires Python 3.11+

```bash
pip install jaxdp
```

For development:

```bash
pip install -e ".[dev]"
```
