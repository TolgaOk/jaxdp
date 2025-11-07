# Composability Visualization

This document provides visual diagrams showing how composition works in jaxdp.

## The Compositional Ladder

```
┌────────────────────────────────────────────────────────────────┐
│                    LEVEL 7: Full Pipeline                       │
│     Adaptive Multi-Stage + Momentum + Blending                  │
│                         ▲                                        │
└─────────────────────────┼──────────────────────────────────────┘
                          │
┌─────────────────────────┼──────────────────────────────────────┐
│              LEVEL 6: Higher-Order Composition                  │
│           momentum( SimplePolicyIteration )                     │
│           momentum( BlendedAlgorithm )                          │
│                         ▲                                        │
└─────────────────────────┼──────────────────────────────────────┘
                          │
┌─────────────────────────┼──────────────────────────────────────┐
│              LEVEL 5: Multi-Stage Composition                   │
│     Stage 1: soft → eval                                        │
│     Stage 2: e_greedy → eval                                    │
│     Stage 3: greedy → eval                                      │
│                         ▲                                        │
└─────────────────────────┼──────────────────────────────────────┘
                          │
┌─────────────────────────┼──────────────────────────────────────┐
│              LEVEL 4: Parallel Composition                      │
│        Branch A: bellman_optimality                             │
│        Branch B: greedy → eval                                  │
│        Blend: 0.6*A + 0.4*B                                     │
│                         ▲                                        │
└─────────────────────────┼──────────────────────────────────────┘
                          │
┌─────────────────────────┼──────────────────────────────────────┐
│             LEVEL 3: Parametric Composition                     │
│        soft_policy(temp=f(t)) → policy_evaluation               │
│                         ▲                                        │
└─────────────────────────┼──────────────────────────────────────┘
                          │
┌─────────────────────────┼──────────────────────────────────────┐
│             LEVEL 2: Sequential Composition                     │
│          greedy_policy → policy_evaluation                      │
│                         ▲                                        │
└─────────────────────────┼──────────────────────────────────────┘
                          │
┌─────────────────────────┼──────────────────────────────────────┐
│                  LEVEL 1: Primitives                            │
│   greedy_policy  soft_policy  e_greedy_policy                   │
│   policy_evaluation  bellman_optimality_operator                │
└─────────────────────────────────────────────────────────────────┘
```

## Composition Patterns

### 1. Sequential Composition (Pipeline)

```
Input State
     │
     ▼
┌─────────────────┐
│  Primitive A    │  greedy_policy.q(q_vals)
│  Extract Policy │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Primitive B    │  policy_evaluation.q(mdp, policy, gamma)
│  Evaluate       │
└────────┬────────┘
         │
         ▼
  New Q-values
```

**Example: Policy Iteration**
```python
policy = greedy_policy.q(q_vals)              # Step 1
q_new = policy_evaluation.q(mdp, policy, γ)   # Step 2
```

---

### 2. Parallel Composition (Blending)

```
       Input State
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌─────────┐
│ Path A │   │ Path B  │
│  VI    │   │   PI    │
└───┬────┘   └────┬────┘
    │             │
    └──────┬──────┘
           ▼
     ┌──────────┐
     │  Blend   │  w₁*A + w₂*B
     └────┬─────┘
          ▼
    New Q-values
```

**Example: Blended VI + PI**
```python
q_vi = bellman_optimality_operator.q(mdp, q_vals, γ)   # Path A
policy = greedy_policy.q(q_vals)
q_pi = policy_evaluation.q(mdp, policy, γ)              # Path B
q_blend = 0.6 * q_vi + 0.4 * q_pi                       # Combine
```

---

### 3. Conditional Composition (Adaptive)

```
   Input State
       │
       ▼
  ┌─────────┐
  │Condition│  if step < threshold
  └────┬────┘
       │
  ┌────┴────┐
  ▼         ▼
┌────┐   ┌─────┐
│ A  │   │  B  │
└─┬──┘   └──┬──┘
  └────┬────┘
       ▼
   New State
```

**Example: Explore then Exploit**
```python
if step < 50:
    policy = soft_policy.q(q_vals, temperature=2.0)    # Explore
else:
    policy = greedy_policy.q(q_vals)                   # Exploit

q_new = policy_evaluation.q(mdp, policy, γ)
```

---

### 4. Parametric Composition (Scheduled)

```
   Input + Time
       │
       ▼
  ┌─────────┐
  │Schedule │  temp = f(time)
  └────┬────┘
       │
       ▼
  ┌─────────┐
  │Primitive│  soft_policy.q(q_vals, temp)
  │with Param
  └────┬────┘
       │
       ▼
   New State
```

**Example: Temperature Annealing**
```python
temperature = 10.0 * (1 - step/max_steps) + 0.1   # Schedule
policy = soft_policy.q(q_vals, temperature)        # Parametric primitive
q_new = policy_evaluation.q(mdp, policy, γ)
```

---

## Real Example: Policy Iteration Decomposed

### The Primitive View

```
┌──────────────────────────────────────────────────────┐
│                   Policy Iteration                    │
│                                                       │
│   Input: Q-values                                     │
│      │                                                │
│      ▼                                                │
│   ┌──────────────────────────────┐                   │
│   │ Primitive 1: greedy_policy   │                   │
│   │ Extract greedy policy        │                   │
│   │ from Q-values                │                   │
│   └──────────────┬───────────────┘                   │
│                  │                                    │
│                  ▼                                    │
│   ┌──────────────────────────────┐                   │
│   │ Primitive 2: policy_eval     │                   │
│   │ Evaluate the greedy policy   │                   │
│   │ to get new Q-values          │                   │
│   └──────────────┬───────────────┘                   │
│                  │                                    │
│                  ▼                                    │
│   Output: New Q-values                               │
└──────────────────────────────────────────────────────┘
```

### The Code

```python
class PolicyIteration(metaclass=StaticMeta):
    @staticmethod
    def update(state, mdp, step):
        # Composition of two primitives
        policy = greedy_policy.q(state.q_vals)              # ← Primitive 1
        q_vals = policy_evaluation.q(mdp, policy, gamma)    # ← Primitive 2
        return state.replace(q_vals=q_vals)
```

---

## Building Complexity: From Primitives to Algorithms

### Example: Optimistic VI with Momentum

```
                      ┌─────────────────────────────────┐
                      │     Optimistic VI + Momentum    │
                      │   (Level 6 Composition)         │
                      └───────────────┬─────────────────┘
                                      │
                      ┌───────────────┴─────────────────┐
                      │                                 │
           ┌──────────▼───────────┐         ┌──────────▼─────────┐
           │  Optimistic VI       │         │  Momentum Wrapper  │
           │  (Level 3)           │         │  (Higher-Order)    │
           └──────────┬───────────┘         └──────────┬─────────┘
                      │                                 │
        ┌─────────────┴──────────────┐                 │
        │                            │                 │
┌───────▼────────┐        ┌─────────▼────────┐        │
│ Bellman Op     │        │ Optimism Bonus   │        │
│ (Primitive)    │        │ (Custom Logic)   │        │
└────────────────┘        └──────────────────┘        │
                                                       │
                         ┌─────────────────────────────┘
                         │
              ┌──────────▼─────────────┐
              │  Q_new + β*(Q_new-Q_old)│
              │  (Momentum Formula)     │
              └─────────────────────────┘
```

**The Code:**
```python
# Level 2: Optimistic VI (Primitive + Custom)
def optimistic_vi_update(state, mdp, step):
    q_new = bellman_optimality_operator.q(mdp, q_vals, γ)  # Primitive
    q_new = q_new + optimism_bonus / sqrt(step + 1)        # Custom
    return state.replace(q_vals=q_new)

# Level 6: Wrap with Momentum (Higher-Order)
optimistic_vi_with_momentum = create_momentum_wrapper(
    optimistic_vi_update,
    momentum=0.9
)
```

---

## Composition Table

| Pattern | Operator | Description | Example |
|---------|----------|-------------|---------|
| **Sequential** | `→` | Chain outputs | `A → B → C` |
| **Parallel** | `⊕` | Blend outputs | `w₁*A ⊕ w₂*B` |
| **Conditional** | `?:` | Choose operator | `cond ? A : B` |
| **Parametric** | `∘` | Apply parameter | `A(f(t))` |
| **Higher-Order** | `F()` | Wrap operator | `F(Algorithm)` |

---

## The Key Insight

### Traditional Approach (Monolithic)
```
┌─────────────────────────────────────┐
│                                     │
│   Giant Black Box Algorithm         │
│   - Can't modify parts              │
│   - Can't reuse components          │
│   - Hard to understand              │
│                                     │
└─────────────────────────────────────┘
```

### Compositional Approach (Modular)
```
┌────────┐   ┌────────┐   ┌────────┐
│ Piece  │ → │ Piece  │ → │ Piece  │
│   A    │   │   B    │   │   C    │
└────────┘   └────────┘   └────────┘
    ↑            ↑            ↑
    │            │            │
  Reusable   Reusable   Reusable
  Modifiable Modifiable Modifiable
  Testable   Testable   Testable
```

**Benefits:**
- ✅ **Reusability**: Use same primitives in different algorithms
- ✅ **Modularity**: Swap components without rewriting everything
- ✅ **Testability**: Test each primitive independently
- ✅ **Clarity**: See exactly what algorithm does at each step
- ✅ **Flexibility**: Mix and match to create new algorithms
- ✅ **JAX-Compatible**: All compositions work with `jit`, `vmap`, `grad`

---

## Practical Example: Three Algorithms from Same Primitives

All three use the same primitives in different combinations:

### Algorithm 1: Value Iteration
```python
q_new = bellman_optimality_operator.q(mdp, q_vals, γ)
```

### Algorithm 2: Policy Iteration
```python
policy = greedy_policy.q(q_vals)
q_new = policy_evaluation.q(mdp, policy, γ)
```

### Algorithm 3: Soft Value Iteration
```python
policy = soft_policy.q(q_vals, temperature)
q_new = policy_evaluation.q(mdp, policy, γ)
```

**Same primitives (`policy_evaluation`, `greedy_policy`, `soft_policy`, `bellman_optimality_operator`), different compositions!**

---

## Next Steps

1. **Run the example**: `python examples/composability_example.py`
2. **Study the code**: See how each level builds on the previous
3. **Create your own**: Combine primitives in new ways
4. **Share**: Contribute your compositions back to the library!

The compositional approach transforms DP algorithm design from **art to science** - systematic, reproducible, and infinitely flexible.
