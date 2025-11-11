# `jaxdp` Learning Examples

This directory contains example implementations and benchmarks for sample-based reinforcement learning algorithms using JAX.

## Learning Algorithms
- **TD Learning** (`td`) - Temporal Difference learning
- **Q-Learning** (`q_learning`) - Off-policy value-based learning
- **SARSA** (`sarsa`) - On-policy value-based learning
- **Expected SARSA** (`expected_sarsa`) - Expected value variant of SARSA

### `benchmark.py` (Coming Soon)
Run the example learning algorithms on simple MDPs via:
```bash
python benchmark.py td                 # TD learning in GridWorld
python benchmark.py multi_seed_td      # Multi-seed comparison in GridWorld
python benchmark.py multi_gamma_td     # Multi-gamma comparison in GridWorld
python benchmark.py benchmark          # Full algorithm comparison
```

> [!NOTE]
> We use `StaticMeta` to create static classes and use them only as namespaces.
> ```python
> class td(metaclass=StaticMeta)
> ```
> With that you can call any attribute via `td.` notation.
