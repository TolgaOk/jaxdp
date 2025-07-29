# `jaxdp` Examples

This directory contains example implementations and benchmarks for dynamic programming algorithms using JAX.

## Planning Algorithms
- **Value Iteration** (`vi`) - Standard value iteration algorithm for solving MDPs
- **Nesterov Accelerated Value Iteration** (`nesterov_vi`) - Accelerated value iteration using Nesterov momentum ([Nesterov, 1983](https://link.springer.com/article/10.1007/BF01065553))
- **Policy Iteration** (`pi`) - Classic policy iteration algorithm

### `benchmark.py`
Run the example planning algorithms on simple MDPs via:
```bash
python benchmark.py vi                 # Value iteration in GridWorld
python benchmark.py multi_seed_vi      # Multi-seed comparison in GridWorld
python benchmark.py multi_gamma_vi     # Multi-gamma comparison in GridWorld
python benchmark.py benchmark          # Full algorithm comparison
```

> [!NOTE]  
> We use `StaticMeta` to create static classes and use them only as namespaces.
> ```python
> class vi(metaclass=StaticMeta)
> ```
> With that we can call `vi.update` function for example.