# Building a Compositional DP Operator Language

## Overview

This directory contains examples and guides for building custom DP algorithms using **compositional operators** - a design pattern where complex algorithms are built by combining simple, reusable primitives.

## 📁 Files in This Directory

### Documentation

- **`COMPOSITIONAL_GUIDE.md`** - Complete user guide with patterns, techniques, and examples
  - Available primitives reference
  - Composition patterns (sequential, parallel, conditional, parametric)
  - Step-by-step custom operator construction
  - Advanced techniques (higher-order operators, curricula, multi-objective)
  - Complete working examples

### Code Examples

- **`compositional_operators.py`** - Library of compositional operator patterns
  - `OptimisticVI` - VI with optimism bonuses
  - `TemperatureScheduledVI` - Soft VI with annealing
  - `MomentumPI` - PI with momentum acceleration
  - `OperatorComposition` - Higher-order combinators (alternate, blend, conditional)
  - `PolicyCombinator` - Policy mixing and blending utilities
  - `CompositionalSampling` - Multi-policy and adaptive sampling
  - `DPOperator` - Operator overloading wrapper class
  - `DPDSL` - Fluent DSL for algorithm construction

- **`demo_compositional_language.py`** - Interactive demonstrations
  - 6 comprehensive demos showing all patterns in action
  - Runnable examples with output

### Existing Algorithm Examples

- **`algorithms.py`** - Built-in algorithm implementations
  - Value Iteration (`vi`)
  - Policy Iteration (`pi`)
  - Nesterov VI (`nesterov_vi`)
  - Q-Learning (`q_learning`)

## 🎯 Quick Start

### 1. Understanding the Primitives

jaxdp provides these composable primitives (from `jaxdp/base.py`):

**Policy Extraction:**
```python
from jaxdp import greedy_policy, soft_policy, e_greedy_policy

pi = greedy_policy.q(q_values)                    # Greedy
pi = soft_policy.q(q_values, temperature=1.0)     # Softmax
pi = e_greedy_policy.q(q_values, epsilon=0.1)     # Epsilon-greedy
```

**Value Operations:**
```python
from jaxdp import policy_evaluation, bellman_operator, bellman_optimality_operator

q = policy_evaluation.q(mdp, policy, gamma)           # Evaluate policy
q = bellman_operator.q(mdp, policy, q_old, gamma)     # Bellman update
q = bellman_optimality_operator.q(mdp, q_old, gamma)  # Greedy Bellman
```

**Sampling:**
```python
from jaxdp import sample_from, sync_sample

action = sample_from(key, policy, state)              # Sample action
trajectory = sync_sample(key, mdp, policy, state, n)  # Sample trajectory
```

### 2. Basic Composition Pattern

Build algorithms by composing primitives:

```python
from flax import struct
from jaxdp.typehints import StaticMeta, QType, MDP

class MyAlgorithm(metaclass=StaticMeta):
    @struct.dataclass
    class State:
        q_vals: QType
        gamma: jnp.ndarray

    @staticmethod
    def init(mdp: MDP, key, gamma: float):
        return MyAlgorithm.State(
            q_vals=jnp.zeros((mdp.num_actions, mdp.num_states)),
            gamma=jnp.array(gamma),
        )

    @staticmethod
    def update(state: "MyAlgorithm.State", mdp: MDP, step: int):
        # Compose primitives here
        policy = greedy_policy.q(state.q_vals)
        q_vals = policy_evaluation.q(mdp, policy, state.gamma)
        return state.replace(q_vals=q_vals)
```

### 3. Run Your Algorithm

```python
from jaxdp.mdp import garnet_mdp
from examples.utils import loop

# Setup
key = jax.random.PRNGKey(42)
mdp = garnet_mdp(key, num_states=50, num_actions=4, branch_factor=3)

# Initialize
state = MyAlgorithm.init(mdp, key, gamma=0.99)

# Run
final_state, metrics = loop(
    mdp=mdp,
    alg_state=state,
    args={'num_iterations': 100},
    update_fn=MyAlgorithm.update,
)
```

## 🧩 Composition Patterns

### Sequential (Pipeline)

Chain operators one after another:

```python
def update(state, mdp, step):
    policy = greedy_policy.q(state.q_vals)          # Step 1
    q_vals = policy_evaluation.q(mdp, policy, ...)  # Step 2
    return state.replace(q_vals=q_vals)             # Step 3
```

### Parallel (Blending)

Combine outputs from multiple operators:

```python
def update(state, mdp, step):
    q1 = bellman_optimality_operator.q(mdp, state.q_vals, gamma)

    policy = greedy_policy.q(state.q_vals)
    q2 = policy_evaluation.q(mdp, policy, gamma)

    # Blend 70% VI + 30% PI
    q_blend = 0.7 * q1 + 0.3 * q2
    return state.replace(q_vals=q_blend)
```

### Conditional (Adaptive)

Switch between operators based on conditions:

```python
def update(state, mdp, step):
    if step < 50:  # Explore phase
        policy = soft_policy.q(state.q_vals, temperature=2.0)
    else:  # Exploit phase
        policy = greedy_policy.q(state.q_vals)

    q_vals = policy_evaluation.q(mdp, policy, gamma)
    return state.replace(q_vals=q_vals)
```

### Parametric (Scheduled)

Use time-varying parameters:

```python
def update(state, mdp, step, total_steps=100):
    # Anneal temperature
    alpha = step / total_steps
    temp = 10.0 * (1 - alpha) + 0.1 * alpha

    policy = soft_policy.q(state.q_vals, temp)
    q_vals = policy_evaluation.q(mdp, policy, gamma)
    return state.replace(q_vals=q_vals)
```

## 🔧 Advanced Patterns

### Higher-Order Combinators

Functions that take/return operators:

```python
def blend_operators(op1_update, op2_update, alpha=0.5):
    """Blend two operators with weight alpha."""
    def blended(state, mdp, step):
        s1 = op1_update(state, mdp, step)
        s2 = op2_update(state, mdp, step)
        q_blend = alpha * s1.q_vals + (1 - alpha) * s2.q_vals
        return s1.replace(q_vals=q_blend)
    return blended

# Usage
from examples.algorithms import vi, pi
hybrid = blend_operators(vi.update, pi.update, alpha=0.7)
```

### Operator Overloading

Python operators for symbolic composition:

```python
from examples.compositional_operators import DPOperator
from examples.algorithms import vi, pi

vi_op = DPOperator(vi.update, vi.State)
pi_op = DPOperator(pi.update, pi.State)

# Use Python operators
hybrid = (vi_op * 0.7) + (pi_op * 0.3)  # Blend
alternating = vi_op | pi_op              # Alternate
pipeline = vi_op >> pi_op                # Sequential
```

### Fluent DSL

Chainable API for algorithm construction:

```python
from examples.compositional_operators import DPDSL

algorithm = (
    DPDSL(mdp)
    .with_exploration(temperature=2.0)
    .with_optimism(bonus=1.0)
    .with_momentum(beta=0.9)
    .with_exploitation()
    .build()
)
```

## 📚 Complete Examples

See `compositional_operators.py` for full implementations:

1. **OptimisticVI** - Bellman + optimism bonuses
2. **TemperatureScheduledVI** - Soft policy + temperature annealing
3. **MomentumPI** - Policy iteration + momentum acceleration
4. **Multi-policy sampling** - Sample from multiple policy types
5. **Adaptive rollout** - Switch policies during rollout
6. **Ensemble methods** - Blend multiple algorithm outputs

## 🎓 Learn More

1. **Read the guide:** `COMPOSITIONAL_GUIDE.md` for comprehensive documentation
2. **Study primitives:** `jaxdp/base.py` to see how operators are implemented
3. **Review algorithms:** `algorithms.py` for VI, PI, and Nesterov VI examples
4. **Run demos:** `demo_compositional_language.py` for interactive examples

## 🚀 Key Principles

1. **Primitives are Pure Functions** - No hidden state or side effects
2. **Explicit State Passing** - State is always explicit in function signatures
3. **JAX-Compatible** - Full support for `jit`, `vmap`, `grad`
4. **Type-Safe** - Uses Flax `@struct.dataclass` and JaxTyping annotations
5. **Composable by Design** - Mix and match any operators

## 💡 Design Philosophy

> "Complex algorithms emerge from simple, composable primitives."

Instead of monolithic algorithm implementations, jaxdp treats DP operators as:
- **Lego blocks** that snap together
- **Functions** that compose naturally
- **Reusable components** across different algorithms
- **JAX transformations** that work seamlessly

This compositional approach enables:
- ✅ Rapid prototyping of new algorithms
- ✅ Easy hyperparameter sweeps with `vmap`
- ✅ Clear, readable algorithm definitions
- ✅ Systematic exploration of algorithm spaces
- ✅ Reproducible research with explicit composition

## 🤝 Contributing

Have a new composition pattern or operator? Add it to `compositional_operators.py`!

Follow the template:
1. Define state with `@struct.dataclass`
2. Use `StaticMeta` metaclass
3. Implement `init()` and `update()` static methods
4. Compose existing primitives in `update()`
5. Document which primitives you're composing

---

**Happy Composing! 🎵**
