# jaxdp

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![JAX 0.8+](https://img.shields.io/badge/JAX-0.8%2B-green)](https://github.com/jax-ml/jax)
[![version](https://img.shields.io/badge/version-0.4.0-orange)](https://github.com/TolgaOk/jaxdp)

Exact dynamic programming (DP) for finite Markov decision processes (MDPs) in JAX.

`jaxdp` provides `jax.jit` and `jax.vmap` compatible implementations of **MDP**s, DP **operators** and **mappings**, and **planning algorithms**.

## Installation

```bash
uv add jaxdp
uv add "jaxdp[dev]"   # for development
```

## Quick start

Make an MDP and apply value iteration steps.

```python
import jaxdp

mdp = jaxdp.make("frozen-lake-deterministic")
algo = jaxdp.ValueIteration(gamma=0.99)
state = algo.init(mdp)

for _ in range(100):
    state = algo.update(mdp, state)

pi = jaxdp.greedy_map.v(mdp, state.v_val, algo.gamma)
v_pi = jaxdp.policy_eval.v(mdp, pi, algo.gamma)
```

You can use `jax.vmap` to compute target values for different discount factors.

```python
import jax
import jax.numpy as jnp
import jaxdp


mdp = jaxdp.make("garnet")
v_val = jnp.linspace(0.0, 1.0, mdp.state_size)


@jax.jit
@jax.vmap
def target_value(gamma: jax.Array) -> jax.Array:
    q_val = jaxdp.reward.sa(mdp) + gamma * jaxdp.trans_op.sa(mdp, v_val)
    return jnp.max(q_val, axis=0)


gammas = jnp.array([0.9, 0.99, 0.995, 0.999])
v_vals = target_value(gammas)
# >>> v_vals.shape
# (4, ...)
```

See the [component reference](https://github.com/TolgaOk/jaxdp/blob/master/jaxdp/README.md) for the public API and MDPs.

## Citation

If you use `jaxdp` in your research, please cite:

```bibtex
@software{tolgaok_jaxdp_2026,
  author  = {Tolga Ok},
  title   = {{jaxdp}: Exact dynamic programming for finite Markov decision processes in JAX},
  year    = {2026},
  version = {0.4.0},
  url     = {https://github.com/TolgaOk/jaxdp},
}
```
