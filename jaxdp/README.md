# `jaxdp` components

This directory contains the `jaxdp` source package. The root package re-exports the main public
components. Stateless namespaces use lowercase aliases, while configured and stateful components
use PascalCase names.

## Models and construction

| Component | Purpose |
| --- | --- |
| `MDP` | Finite Markov decision process with transition, reward, initial, and terminal arrays. |
| `MRP` | Finite Markov reward process induced by a fixed policy. |
| `make_mrp` | Convert an `MDP` and policy into an `MRP`. |
| `make` | Construct a reproducible named MDP recipe. |

Parameterized factories remain available from their defining `jaxdp.mdp` modules. For example,
import `grid_world` from `jaxdp.mdp.grid_world`.

Named recipes are grouped by family:

- Grid: `cliff-walking`, `four-rooms`, `frozen-lake`, `frozen-lake-deterministic`,
  `grid-world`, and `grid-world-slippery`.
- Delayed reward: `delayed-reward`, `delayed-reward-noisy`, `delayed-reward-long`, and
  `delayed-reward-long-noisy`.
- Forest: `forest` and `forest-long`.
- Garnet: `garnet`, `garnet-medium`, `garnet-large`, and `garnet-dense`.
- Graph: `graph`.
- Sequential: `sequential` and `sequential-long`.
- Tree: `tree` and `tree-deep`.

## Operators

| Component | Purpose |
| --- | --- |
| `trans_op` | Apply terminal-aware state and state-action transition operators. |
| `adj_trans_op` | Apply the corresponding forward measure operators. |
| `resolvent` | Solve discounted state and state-action linear systems. |
| `bellman_op` | Apply fixed-policy Bellman operators to state or action values. |
| `bellman_opt_op` | Apply Bellman optimality operators to state or action values. |
| `SoftBellmanOptOp` | Apply configured entropy-regularized Bellman optimality operators. |
| `MellowMaxBellmanOptOp` | Apply configured Mellowmax Bellman optimality operators. |
| `BoltzmannBellmanOp` | Apply configured Boltzmann-expectation Bellman operators. |

## Mappings

| Component | Purpose |
| --- | --- |
| `reward` | Compute expected state-action or policy rewards. |
| `greedy_map` | Map state or action values to a greedy policy. |
| `SoftGreedyMap` | Map values to an entropy-regularized policy. |
| `EpsilonGreedy` | Map values to an epsilon-greedy policy. |
| `proj_simplex` | Project action vectors onto the probability simplex. |
| `MellowMax` | Reduce action values with normalized log-mean-exp. |
| `expectation` | Evaluate functions under supplied state or state-action distributions. |
| `Occupancy` | Compute finite-step normalized discounted occupancy measures. |
| `stationary` | Compute invariant state or state-action distributions. |
| `eigenvalues` | Compute eigenvalues of a policy-induced state transition operator. |

## Planning

Every iterative planner is a configured dataclass with a nested `State`, an `init` method, and an
`update` method that performs one algorithmic step.

| Component | Purpose |
| --- | --- |
| `policy_eval` | Compute exact state or action values for a fixed policy. |
| `IterativePolicyEvaluation` | Apply one fixed-policy Bellman evaluation update. |
| `ValueIteration` | Apply one state-value iteration update. |
| `QValueIteration` | Apply one action-value iteration update. |
| `AnchoredValueIteration` | Apply one anchored state-value iteration update. |
| `AnchoredQValueIteration` | Apply one anchored action-value iteration update. |
| `SafeAcceleratedValueIteration` | Apply one safeguarded accelerated value update. |
| `MomentumValueIteration` | Apply one momentum value-iteration update. |
| `PIDValueIteration` | Apply one fixed-gain PID value-iteration update. |
| `AndersonValueIteration` | Apply one regularized Anderson acceleration update. |
| `SafeAndersonValueIteration` | Apply one globally safeguarded Type-I Anderson update. |
| `RankOneValueIteration` | Apply one rank-one value-iteration update. |
| `DeflatedValueIteration` | Apply one rank-one deflated-dynamics update. |
| `QuasiPolicyIteration` | Apply one safeguarded quasi-policy-iteration update. |
| `DynamicBoltzmannValueIteration` | Apply one dynamic Boltzmann value update. |
| `AcceleratedPolicyIteration` | Apply one accelerated evaluation or improvement micro-step. |
| `PolicyIteration` | Apply one exact policy-improvement and evaluation update. |
