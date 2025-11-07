# Compositional DP Operator Language - User Guide

This guide shows you how to build custom DP algorithms using compositional primitives in jaxdp.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Available Primitives](#available-primitives)
3. [Composition Patterns](#composition-patterns)
4. [Building Custom Operators](#building-custom-operators)
5. [Advanced Techniques](#advanced-techniques)
6. [Complete Examples](#complete-examples)

---

## Core Concepts

### What is Compositional DP?

jaxdp treats DP operators as **composable functions** that can be combined to build complex algorithms from simple primitives. Think of it like building with LEGO blocks:

- **Primitives** = Basic blocks (greedy_policy, bellman_operator, etc.)
- **Composition** = Connecting blocks (function calls, combinators)
- **Algorithms** = Complex structures built from primitives

### Three Layers of Abstraction

```
┌─────────────────────────────────────┐
│  DSL Layer (Optional)               │  Fluent APIs, operator overloading
├─────────────────────────────────────┤
│  Combinator Layer                   │  Higher-order composition functions
├─────────────────────────────────────┤
│  Primitive Layer                    │  Base DP operators (greedy, bellman, etc.)
└─────────────────────────────────────┘
```

---

## Available Primitives

### Policy Extraction Primitives

Extract policies from value functions:

```python
from jaxdp import greedy_policy, soft_policy, e_greedy_policy

# Greedy: argmax policy
pi_greedy = greedy_policy.q(q_values)

# Softmax with temperature
pi_soft = soft_policy.q(q_values, temperature=1.0)

# Epsilon-greedy exploration
pi_explore = e_greedy_policy.q(q_values, epsilon=0.1)
```

### Value Evaluation Primitives

Evaluate or update value functions:

```python
from jaxdp import policy_evaluation, bellman_operator, bellman_optimality_operator

# Evaluate a policy
q_eval = policy_evaluation.q(mdp, policy, gamma)

# Bellman operator with policy
q_new = bellman_operator.q(mdp, policy, q_old, gamma)

# Greedy Bellman (for VI)
q_new = bellman_optimality_operator.q(mdp, q_old, gamma)
```

### Distribution Primitives

Work with state distributions:

```python
from jaxdp import stationary_distribution, expected_value

# Find stationary distribution
mu_stationary = stationary_distribution.q(mdp, policy, iterations=100)

# Compute expected value
v_expected = expected_value.q(mdp, initial_distribution)
```

### Sampling Primitives

Sample trajectories and actions:

```python
from jaxdp import sample_from, sync_sample

# Sample action from policy
action = sample_from(key, policy, state)

# Sample full trajectory
trajectory = sync_sample(key, mdp, policy, initial_state, num_steps)
```

---

## Composition Patterns

### Pattern 1: Sequential Composition (Pipeline)

Apply operators one after another:

```python
def my_algorithm_update(state, mdp, step):
    # Step 1: Extract policy
    policy = greedy_policy.q(state.q_vals)

    # Step 2: Evaluate policy
    q_vals = policy_evaluation.q(mdp, policy, state.gamma)

    # Step 3: Update state
    return state.replace(q_vals=q_vals)

# This is Policy Iteration!
```

### Pattern 2: Parallel Composition (Blending)

Combine outputs of multiple operators:

```python
def blended_update(state, mdp, step):
    # Compute two different updates
    q_vi = bellman_optimality_operator.q(mdp, state.q_vals, state.gamma)

    policy = greedy_policy.q(state.q_vals)
    q_pi = policy_evaluation.q(mdp, policy, state.gamma)

    # Blend them
    q_blend = 0.7 * q_vi + 0.3 * q_pi

    return state.replace(q_vals=q_blend)
```

### Pattern 3: Conditional Composition

Switch between operators based on conditions:

```python
def adaptive_update(state, mdp, step):
    # Use soft policy early, greedy later
    explore_phase = step < 50

    if explore_phase:
        policy = soft_policy.q(state.q_vals, temperature=2.0)
    else:
        policy = greedy_policy.q(state.q_vals)

    q_vals = policy_evaluation.q(mdp, policy, state.gamma)
    return state.replace(q_vals=q_vals)
```

### Pattern 4: Parametric Composition

Add parameters to primitives:

```python
def temperature_annealed_update(state, mdp, step, max_steps=100):
    # Anneal temperature over time
    temperature = 10.0 * (1 - step / max_steps) + 0.1

    policy = soft_policy.q(state.q_vals, temperature)
    q_vals = policy_evaluation.q(mdp, policy, state.gamma)

    return state.replace(q_vals=q_vals)
```

---

## Building Custom Operators

### Step 1: Define State Class

Use Flax's `@struct.dataclass` for JAX compatibility:

```python
from flax import struct
from jaxdp.typehints import QType, StaticMeta

class MyAlgorithm(metaclass=StaticMeta):
    @struct.dataclass
    class State:
        q_vals: QType          # Required: Q-values
        gamma: jnp.ndarray     # Required: discount factor
        # Add your custom fields:
        temperature: float
        momentum: float
        iteration: int
```

### Step 2: Define Initialization

```python
    @staticmethod
    def init(mdp, key, gamma, temperature=1.0, momentum=0.9):
        return MyAlgorithm.State(
            q_vals=jnp.zeros((mdp.num_actions, mdp.num_states)),
            gamma=jnp.array(gamma),
            temperature=temperature,
            momentum=momentum,
            iteration=0,
        )
```

### Step 3: Define Update Function

Compose primitives in the update:

```python
    @staticmethod
    def update(state: "MyAlgorithm.State", mdp, step: int):
        # 1. Extract policy with temperature
        policy = soft_policy.q(state.q_vals, state.temperature)

        # 2. Evaluate policy
        new_q = policy_evaluation.q(mdp, policy, state.gamma)

        # 3. Add momentum
        if state.iteration > 0:
            new_q = new_q + state.momentum * (new_q - state.q_vals)

        # 4. Anneal temperature
        new_temp = state.temperature * 0.99

        return state.replace(
            q_vals=new_q,
            temperature=new_temp,
            iteration=state.iteration + 1
        )
```

### Complete Custom Operator

```python
from flax import struct
import jax.numpy as jnp
from jaxdp import soft_policy, policy_evaluation
from jaxdp.typehints import MDP, QType, StaticMeta

class AdaptiveSoftPI(metaclass=StaticMeta):
    """
    Policy Iteration with adaptive temperature and momentum.

    Composes:
    - soft_policy (with annealing)
    - policy_evaluation
    - momentum acceleration
    """

    @struct.dataclass
    class State:
        q_vals: QType
        prev_q: QType
        gamma: jnp.ndarray
        temperature: float
        momentum: float
        iteration: int

    @staticmethod
    def init(mdp: MDP, key, gamma: float,
             init_temp: float = 5.0, momentum: float = 0.95):
        init_q = jnp.zeros((mdp.num_actions, mdp.num_states))
        return AdaptiveSoftPI.State(
            q_vals=init_q,
            prev_q=init_q,
            gamma=jnp.array(gamma),
            temperature=init_temp,
            momentum=momentum,
            iteration=0,
        )

    @staticmethod
    def update(state: "AdaptiveSoftPI.State", mdp: MDP, step: int):
        # Compose primitives
        policy = soft_policy.q(state.q_vals, state.temperature)
        new_q = policy_evaluation.q(mdp, policy, state.gamma)

        # Add momentum
        if state.iteration > 0:
            momentum_term = state.momentum * (state.q_vals - state.prev_q)
            new_q = new_q + momentum_term

        # Anneal temperature (exponential decay)
        new_temp = state.temperature * 0.99

        return state.replace(
            q_vals=new_q,
            prev_q=state.q_vals,
            temperature=new_temp,
            iteration=state.iteration + 1,
        )
```

---

## Advanced Techniques

### Technique 1: Higher-Order Operators

Create operators that take operators as input:

```python
def create_momentum_wrapper(base_update_fn, momentum=0.9):
    """
    Wrap any update function with momentum.
    """
    def momentum_update(state, mdp, step):
        # Save old Q-values
        old_q = state.q_vals

        # Run base update
        new_state = base_update_fn(state, mdp, step)

        # Add momentum if not first iteration
        if step > 0:
            momentum_q = new_state.q_vals + momentum * (new_state.q_vals - old_q)
            return new_state.replace(q_vals=momentum_q)

        return new_state

    return momentum_update

# Usage:
from examples.algorithms import vi
vi_with_momentum = create_momentum_wrapper(vi.update, momentum=0.95)
```

### Technique 2: Policy Mixtures

Blend multiple policy types:

```python
def mixture_policy(q_vals, weights, temperature=1.0, epsilon=0.1):
    """
    Create policy mixture from multiple primitives.

    Args:
        weights: [w_greedy, w_soft, w_egreedy] must sum to 1
    """
    from jaxdp import greedy_policy, soft_policy, e_greedy_policy

    pi_greedy = greedy_policy.q(q_vals)
    pi_soft = soft_policy.q(q_vals, temperature)
    pi_egreedy = e_greedy_policy.q(q_vals, epsilon)

    # Weighted mixture
    pi_mix = (
        weights[0] * pi_greedy +
        weights[1] * pi_soft +
        weights[2] * pi_egreedy
    )

    # Renormalize
    return pi_mix / jnp.sum(pi_mix, axis=0, keepdims=True)
```

### Technique 3: Curriculum Learning

Create schedules for parameters:

```python
class CurriculumScheduler:
    """Schedule parameter changes over training."""

    @staticmethod
    def linear_anneal(start, end, step, total_steps):
        alpha = min(step / total_steps, 1.0)
        return (1 - alpha) * start + alpha * end

    @staticmethod
    def exponential_decay(start, decay_rate, step):
        return start * (decay_rate ** step)

    @staticmethod
    def cosine_anneal(start, end, step, total_steps):
        alpha = min(step / total_steps, 1.0)
        cos_factor = (1 + jnp.cos(jnp.pi * alpha)) / 2
        return end + (start - end) * cos_factor


def curriculum_vi_update(state, mdp, step, total_steps=1000):
    """VI with curriculum temperature annealing."""

    # Anneal temperature
    temperature = CurriculumScheduler.cosine_anneal(
        start=10.0, end=0.01, step=step, total_steps=total_steps
    )

    # Use soft policy with scheduled temperature
    policy = soft_policy.q(state.q_vals, temperature)
    q_vals = policy_evaluation.q(mdp, policy, state.gamma)

    return state.replace(q_vals=q_vals)
```

### Technique 4: Multi-Objective Composition

Optimize multiple objectives:

```python
def multi_objective_update(state, mdp, step, risk_aversion=0.5):
    """
    Combine expected return with risk minimization.

    Composes:
    - Standard Bellman for expected return
    - Variance penalty for risk
    """
    from jaxdp import bellman_optimality_operator

    # Objective 1: Expected return
    q_expected = bellman_optimality_operator.q(mdp, state.q_vals, state.gamma)

    # Objective 2: Minimize variance (simplified - use value variance as proxy)
    q_variance = jnp.var(state.q_vals, axis=0, keepdims=True)
    q_risk_adjusted = q_expected - risk_aversion * q_variance

    return state.replace(q_vals=q_risk_adjusted)
```

### Technique 5: Operator Overloading

Create symbolic DSL with Python operators:

```python
class DPOperator:
    """Wrap operators to support symbolic composition."""

    def __init__(self, update_fn):
        self.update_fn = update_fn

    def __add__(self, other):
        """+ operator: equal-weighted blend"""
        return DPOperator(
            lambda s, m, t: blend_operators(
                self.update_fn(s, m, t),
                other.update_fn(s, m, t),
                alpha=0.5
            )
        )

    def __mul__(self, weight):
        """* operator: scale influence"""
        return DPOperator(
            lambda s, m, t: scale_operator(
                self.update_fn(s, m, t),
                s,
                weight
            )
        )

    def __or__(self, other):
        """| operator: alternate between operators"""
        return DPOperator(
            lambda s, m, t: (
                self.update_fn(s, m, t) if t % 2 == 0
                else other.update_fn(s, m, t)
            )
        )

    def __call__(self, state, mdp, step):
        return self.update_fn(state, mdp, step)


# Usage:
vi_op = DPOperator(vi.update)
pi_op = DPOperator(pi.update)

hybrid = (vi_op * 0.7) + (pi_op * 0.3)  # 70% VI + 30% PI
alternating = vi_op | pi_op              # Alternate VI/PI
```

---

## Complete Examples

### Example 1: Optimistic VI with Bonus Decay

```python
from flax import struct
import jax.numpy as jnp
from jaxdp import bellman_optimality_operator
from jaxdp.typehints import MDP, QType, StaticMeta

class OptimisticVI(metaclass=StaticMeta):
    """Value Iteration with optimism bonuses."""

    @struct.dataclass
    class State:
        q_vals: QType
        gamma: jnp.ndarray
        optimism: float
        decay_rate: float

    @staticmethod
    def init(mdp: MDP, key, gamma: float, optimism: float = 1.0, decay: float = 0.99):
        return OptimisticVI.State(
            q_vals=jnp.zeros((mdp.num_actions, mdp.num_states)),
            gamma=jnp.array(gamma),
            optimism=optimism,
            decay_rate=decay,
        )

    @staticmethod
    def update(state: "OptimisticVI.State", mdp: MDP, step: int):
        # Bellman update
        q_new = bellman_optimality_operator.q(mdp, state.q_vals, state.gamma)

        # Add decaying optimism bonus
        bonus = state.optimism * (state.decay_rate ** step)
        q_optimistic = q_new + bonus

        return state.replace(q_vals=q_optimistic)
```

### Example 2: Ensemble DP Algorithm

```python
class EnsembleDP(metaclass=StaticMeta):
    """
    Run multiple algorithms and blend their outputs.
    """

    @struct.dataclass
    class State:
        q_vals_list: list[QType]  # One per algorithm
        weights: jnp.ndarray       # Blending weights
        gamma: jnp.ndarray

    @staticmethod
    def init(mdp: MDP, key, gamma: float, num_algorithms: int = 3):
        init_q = jnp.zeros((mdp.num_actions, mdp.num_states))
        return EnsembleDP.State(
            q_vals_list=[init_q for _ in range(num_algorithms)],
            weights=jnp.ones(num_algorithms) / num_algorithms,
            gamma=jnp.array(gamma),
        )

    @staticmethod
    def update(state: "EnsembleDP.State", mdp: MDP, step: int):
        from jaxdp import (
            bellman_optimality_operator,
            greedy_policy,
            soft_policy,
            policy_evaluation,
        )

        # Algorithm 1: VI
        q1 = bellman_optimality_operator.q(mdp, state.q_vals_list[0], state.gamma)

        # Algorithm 2: Greedy PI
        pi2 = greedy_policy.q(state.q_vals_list[1])
        q2 = policy_evaluation.q(mdp, pi2, state.gamma)

        # Algorithm 3: Soft PI
        pi3 = soft_policy.q(state.q_vals_list[2], temperature=1.0)
        q3 = policy_evaluation.q(mdp, pi3, state.gamma)

        # Blend with weights
        q_blend = (
            state.weights[0] * q1 +
            state.weights[1] * q2 +
            state.weights[2] * q3
        )

        return state.replace(
            q_vals_list=[q1, q2, q3],
            # Optionally: update weights based on performance
        )
```

### Example 3: Hierarchical Policy Composition

```python
def hierarchical_policy(q_vals, state, level='high'):
    """
    Different policy behaviors at different abstraction levels.
    """
    from jaxdp import greedy_policy, soft_policy, e_greedy_policy

    if level == 'high':
        # High-level: greedy exploitation
        return greedy_policy.q(q_vals)

    elif level == 'mid':
        # Mid-level: temperature-based exploration
        return soft_policy.q(q_vals, temperature=1.0)

    elif level == 'low':
        # Low-level: random exploration
        return e_greedy_policy.q(q_vals, epsilon=0.3)

    else:
        # Adaptive: choose based on state value confidence
        max_q = jnp.max(q_vals, axis=0)
        confidence = max_q - jnp.mean(q_vals, axis=0)

        # High confidence → greedy, low confidence → explore
        threshold = 1.0
        use_greedy = confidence[state] > threshold

        return jax.lax.cond(
            use_greedy,
            lambda: greedy_policy.q(q_vals),
            lambda: soft_policy.q(q_vals, temperature=2.0),
        )
```

---

## Usage Patterns

### Run Custom Algorithm

```python
from jaxdp.mdp import garnet_mdp
from examples.utils import loop

# Create MDP
key = jax.random.PRNGKey(42)
mdp = garnet_mdp(key, num_states=100, num_actions=5, branch_factor=3)

# Initialize your custom algorithm
state = MyAlgorithm.init(mdp, key, gamma=0.99)

# Run iterations
num_iterations = 500
final_state, metrics = loop(
    mdp=mdp,
    alg_state=state,
    args={'num_iterations': num_iterations},
    update_fn=MyAlgorithm.update,
    metrics_fn=None,  # Optional metrics
)

# Extract final policy
final_policy = greedy_policy.q(final_state.q_vals)
```

### Vectorize Across Parameters

```python
# Test multiple hyperparameters in parallel
temperatures = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])

# Vectorized initialization
vmap_init = jax.vmap(MyAlgorithm.init, in_axes=(None, None, None, 0))
states = vmap_init(mdp, key, 0.99, temperatures)

# Vectorized updates
vmap_update = jax.vmap(MyAlgorithm.update, in_axes=(0, None, None))

for step in range(100):
    states = vmap_update(states, mdp, step)

# Compare results across temperatures
final_q_values = states.q_vals  # Shape: (5, num_actions, num_states)
```

---

## Summary

**Compositional DP in jaxdp follows these principles:**

1. **Primitives are Pure Functions**: All operators are stateless, taking explicit inputs and returning explicit outputs

2. **Composition via Function Calls**: Combine operators by passing outputs as inputs

3. **JAX-Compatible Design**: Full support for `jit`, `vmap`, `grad`, and other transformations

4. **Type-Safe State**: Use Flax `@struct.dataclass` for algorithm state

5. **Explicit over Implicit**: No hidden state, magic, or side effects

**Your toolkit:**
- ✅ Policy primitives: `greedy_policy`, `soft_policy`, `e_greedy_policy`
- ✅ Value primitives: `policy_evaluation`, `bellman_operator`, `bellman_optimality_operator`
- ✅ Distribution primitives: `stationary_distribution`, `expected_value`
- ✅ Sampling primitives: `sample_from`, `sync_sample`
- ✅ Composition patterns: sequential, parallel, conditional, parametric
- ✅ Higher-order operators: combinators, wrappers, schedulers

**Next steps:**
1. Browse `examples/algorithms.py` for built-in algorithm examples
2. Study `jaxdp/base.py` to see primitive implementations
3. Run `examples/demo_compositional_language.py` for interactive demonstrations
4. Build your own operators using this guide!

Happy composing! 🎵
