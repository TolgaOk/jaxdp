# `jaxdp` Planning Examples

This directory contains example implementations and benchmarks for model-based dynamic programming algorithms using JAX.

The library planners expose `State`, `init`, and one-step `update`; the benchmark owns repeated
iteration and metric collection.

## Garnet Value Error

`garnet.py` compares the public value-based planners on one reproducible Garnet MDP. Exact policy
iteration supplies the reference value, and the plot reports
`max_s |v_k(s) - v*(s)|` after each planner update.
The plot requires Matplotlib.

```bash
python garnet.py
python garnet.py --gamma 0.99 --step 200 --show
```

The horizontal axis counts planner updates, whose computational costs differ. Some planners perform
a required warm-start update during `init`. M-VI and Accelerated PI also have spectral assumptions
that a random Garnet need not satisfy, so their error can increase. PID VI uses its safe default
gains and therefore coincides with ordinary VI.

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
