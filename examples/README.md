# `jaxdp` Examples

This directory contains example implementations and benchmarks for dynamic programming algorithms using JAX.

## Planning Algorithms
- **Value Iteration** (`vi`)
- **[Nesterov Accelerated Value Iteration](https://pubsonline.informs.org/doi/abs/10.1287/opre.2022.2269?casa_token=Pn5-2vDJXSsAAAAA:dZoGqfnlSbdOf4PXuNcn6g-NYXJrDLQzrZHsegktTRrKbyQd_K6r0SFlP8Wy8r5r_jsgpQ8)** (`nesterov_vi`)
- **Policy Iteration** (`pi`)

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
> With that you can call any attribute via `vi.` notation.