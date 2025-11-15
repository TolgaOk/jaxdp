# Base API: Before vs After

## The Core Question

**How do we make `jaxdp` base primitives work for BOTH planning and learning?**

---

## Current State (One-Hot Only)

```python
# jaxdp/base.py - Current Implementation

def async_sample_step(mdp, action, state, episode_step, episode_length, key):
    """
    Sample MDP transition.

    Args:
        action: ONE-HOT array [A]
        state: ONE-HOT array [S]

    Returns:
        next_state: ONE-HOT array [S]
        reward: scalar
        ...
    """
    # Uses einsum with one-hot vectors
    next_state_p = jnp.einsum("a,axs,s->x", action, mdp.transition, state)
    next_state = distrax.OneHotCategorical(probs=next_state_p).sample(seed=key)
    reward = jnp.einsum("asx,a,s,x->", mdp.reward, action, state, next_state)
    # ...
    return next_state, reward, terminal, timeout, new_state, new_step
```

### Problem

```python
# Learning example - WASTEFUL
state = jnp.array([0, 0, 0, 1, 0, 0, ..., 0])  # 100,000 elements for S=100K!
action = jnp.array([0, 1, 0, 0])                # Just to represent state 3, action 1

next_state, reward, ... = async_sample_step(mdp, action, state, ...)
# next_state is ALSO 100,000 elements!

# For batch of 1000: 800 MB just for states!
```

---

## Proposed Solution (Type Polymorphic)

```python
# jaxdp/base.py - New Implementation

def async_sample_step(mdp, action, state, episode_step, episode_length, key):
    """
    Sample MDP transition - works with INDICES or ONE-HOT.

    Args:
        action: int (index) OR array (one-hot)
        state: int (index) OR array (one-hot)

    Returns:
        next_state: SAME TYPE as input (int or array)
        reward: scalar
        ...

    Examples:
        # Indexed (efficient)
        >>> next_s, r, ... = async_sample_step(mdp, 1, 3, ...)
        >>> type(next_s)
        int

        # One-hot (compatible)
        >>> next_s, r, ... = async_sample_step(mdp, [0,1,0], [0,0,0,1,0], ...)
        >>> next_s.shape
        (5,)
    """
    # Auto-detect input type
    if jnp.ndim(state) == 0:  # scalar → indexed
        return _async_sample_step_indexed(mdp, action, state, ...)
    else:  # array → one-hot
        return _async_sample_step_onehot(mdp, action, state, ...)


def _async_sample_step_indexed(mdp, action: int, state: int, ...):
    """Efficient indexed implementation."""
    # Direct indexing - no einsum
    next_state_probs = mdp.transition[action, :, state]
    next_state = jrd.choice(key, mdp.state_size, p=next_state_probs)  # Returns int
    reward = mdp.reward[action, state, next_state]
    # ...
    return next_state, reward, ...  # next_state is int!


def _async_sample_step_onehot(mdp, action: Array, state: Array, ...):
    """Backward-compatible one-hot implementation."""
    # Current implementation (unchanged)
    next_state_p = jnp.einsum("a,axs,s->x", action, mdp.transition, state)
    next_state = distrax.OneHotCategorical(probs=next_state_p).sample(seed=key)
    # ...
    return next_state, reward, ...  # next_state is array
```

### Benefit

```python
# Learning example - EFFICIENT
state = 3       # Just an integer!
action = 1      # Just an integer!

next_state, reward, ... = async_sample_step(mdp, action, state, ...)
# next_state is ALSO an integer!

# For batch of 1000: 12 KB instead of 800 MB
# 66,000x memory savings!
```

---

## Side-by-Side Comparison

### Scenario: Collect 1000 transitions for Q-learning (S=100K states)

#### Current (One-Hot Only)

```python
from jaxdp import async_sample_step_pi

transitions = []
state = jnp.zeros(100_000)  # One-hot initial state
state = state.at[0].set(1.0)

for _ in range(1000):
    action, next_state, reward, terminal, _, state, _ = \
        async_sample_step_pi(mdp, policy, state, step, max_steps, key)

    # state: [100000] array
    # action: [10] array
    # next_state: [100000] array

    transitions.append((state, action, reward, next_state, terminal))

# Memory: 1000 * (100000 + 100000) * 4 bytes = 800 MB
```

#### Proposed (Type Polymorphic)

```python
from jaxdp import async_sample_step_pi

transitions = []
state = 0  # Integer initial state

for _ in range(1000):
    action, next_state, reward, terminal, _, state, _ = \
        async_sample_step_pi(mdp, policy, state, step, max_steps, key)

    # state: int
    # action: int
    # next_state: int

    transitions.append((state, action, reward, next_state, terminal))

# Memory: 1000 * 3 * 4 bytes = 12 KB
# 66,000x smaller!
```

---

## What Changes in Base Library

### File: `jaxdp/base.py`

**Functions that need modification:**

1. **`async_sample_step()`** - Add type detection and dispatch
2. **`async_sample_step_pi()`** - Use new `async_sample_step()`
3. **`sample_from()`** - Add optional state parameter with type support

**Functions that DON'T change:**

1. ✓ `bellman_optimality_operator.q()` - Works on Q-tables (always dense)
2. ✓ `bellman_operator.q()` - Works on Q-tables
3. ✓ `greedy_policy.q()` - Works on Q-tables
4. ✓ `policy_evaluation.q()` - Works on Q-tables
5. ✓ All other Bellman/policy functions

**Why most functions don't change:**
- They operate on **Q-value tables** which are always dense `[A, S]` matrices
- Planning and learning both use the same Q-tables
- Only **sampling** needs to support both representations

---

## What Changes in Examples

### Planning Examples (`examples/planning/`) - NO CHANGE

```python
# examples/planning/algorithms.py

from jaxdp import bellman_optimality_operator

class vi:
    def update(state, mdp, step):
        # Still works exactly as before
        next_q = bellman_optimality_operator.q(mdp, state.q_vals, state.gamma)
        return state.replace(q_vals=next_q)

# Planning doesn't use sampling much, so unaffected!
```

### Learning Examples (`examples/learning/`) - CAN USE INDICES

```python
# examples/learning/algorithms.py

from jaxdp import async_sample_step_pi

# NEW: Efficient indexed version
class q_learning_indexed:

    def collect_transitions(mdp, policy, initial_state_idx, n_samples, key):
        """Collect transitions using indexed representation."""
        transitions = []
        state = initial_state_idx  # Integer!

        for _ in range(n_samples):
            # Pass integer, get integer back
            action, next_state, reward, terminal, _, state, _ = \
                async_sample_step_pi(mdp, policy, state, 0, 100, key)

            transitions.append({
                'state': state,      # int
                'action': action,    # int
                'reward': reward,
                'next_state': next_state,  # int
                'terminal': terminal
            })

        return transitions

    def update(q_vals, transition, alpha, gamma):
        """Update Q-values using indexed transition."""
        s, a = transition['state'], transition['action']
        curr_q = q_vals[a, s]  # Direct indexing!

        s_next = transition['next_state']
        max_next_q = jnp.max(q_vals[:, s_next])

        target = transition['reward'] + gamma * max_next_q
        q_vals = q_vals.at[a, s].add(alpha * (target - curr_q))

        return q_vals


# OLD: Backward compatible one-hot version
class q_learning_onehot:

    def collect_transitions(mdp, policy, initial_state_oh, n_samples, key):
        """Collect transitions using one-hot representation."""
        transitions = []
        state = initial_state_oh  # Array!

        for _ in range(n_samples):
            # Pass array, get array back
            action, next_state, reward, terminal, _, state, _ = \
                async_sample_step_pi(mdp, policy, state, 0, 100, key)

            transitions.append({
                'state': state,          # array [S]
                'action': action,        # array [A]
                'reward': reward,
                'next_state': next_state,  # array [S]
                'terminal': terminal
            })

        return transitions

    def update(q_vals, transition, alpha, gamma):
        """Update Q-values using one-hot transition."""
        curr_q = jnp.einsum("as,a,s->", q_vals,
                           transition['action'], transition['state'])
        # ... rest of einsum-based update
        return q_vals
```

---

## User Experience

### Planning User (No Change)

```python
from jaxdp import bellman_optimality_operator, greedy_policy
from jaxdp.mdp import GridWorld

mdp = GridWorld(10, 10)
q_vals = jnp.zeros((mdp.action_size, mdp.state_size))

# Value iteration - unchanged
for _ in range(100):
    q_vals = bellman_optimality_operator.q(mdp, q_vals, gamma=0.99)

policy = greedy_policy.q(q_vals)
# Works exactly as before!
```

### Learning User - Small MDP (Can Use Either)

```python
from jaxdp import async_sample_step_pi

# Small MDP - one-hot is fine
mdp = GridWorld(10, 10)  # 100 states

state = jnp.zeros(100)
state = state.at[0].set(1.0)  # One-hot

action, next_state, ... = async_sample_step_pi(mdp, policy, state, ...)
# Works as before!
```

### Learning User - Large MDP (Use Indices)

```python
from jaxdp import async_sample_step_pi

# Large MDP - use indices for efficiency
mdp = GridWorld(316, 316)  # 100K states

state = 0  # Integer!

action, next_state, ... = async_sample_step_pi(mdp, policy, state, ...)
# Returns integers - efficient!

# Collect batch
transitions = []
for _ in range(1000):
    action, next_state, reward, terminal, _, state, _ = \
        async_sample_step_pi(mdp, policy, state, ...)
    transitions.append((state, action, next_state, reward, terminal))

# Only 12 KB instead of 800 MB!
```

---

## Implementation Complexity

### Changes Required

**jaxdp/base.py** (core library):
```
Lines changed: ~100-150
New functions: 2-3 (_async_sample_step_indexed, etc.)
Modified functions: 3 (async_sample_step, async_sample_step_pi, sample_from)
Breaking changes: 0 (backward compatible)
```

**examples/learning/** (examples):
```
New files: 1-2 (showing indexed usage)
Changed files: 0 (old examples still work)
Documentation: Update to show both approaches
```

**tests/**:
```
New tests: ~20 (test indexed versions)
Modified tests: 0 (existing tests unchanged)
```

### Maintenance Burden

**Low:**
- Only 3 functions need dispatch logic
- Both paths share similar structure
- Type detection is simple (`jnp.ndim(x) == 0`)
- JIT compiles away the dispatch overhead

---

## Decision Tree for Users

```
Are you implementing a planning algorithm?
│
├─ YES → Use Bellman operators
│         from jaxdp import bellman_optimality_operator
│         q_vals = bellman_optimality_operator.q(mdp, q_vals, gamma)
│         ✓ No change from current usage
│
└─ NO (learning algorithm)
    │
    How many states in your MDP?
    │
    ├─ < 10,000 states
    │  → Either representation works fine
    │     Use integers for simplicity
    │
    └─ > 10,000 states
       → MUST use indexed (integers)
          state = 3  # not jnp.array([0,0,0,1,0,...])
          Otherwise: Out of Memory!
```

---

## Summary

### The Design

**Base library (`jaxdp/`):**
- Bellman operators: unchanged (work on Q-tables)
- Sampling primitives: type-polymorphic (detect int vs array)

**Examples:**
- Planning: unchanged (uses Bellman operators)
- Learning: can use indexed (efficient) or one-hot (compatible)

### The API

**One function, two modes:**
```python
# Mode 1: Indexed (efficient)
next_s, r, ... = async_sample_step(mdp, 1, 3, ...)

# Mode 2: One-hot (compatible)
next_s, r, ... = async_sample_step(mdp, [0,1,0], [0,0,0,1,0], ...)
```

**Same function. Different input type. Automatic optimization.**

### The Benefits

1. ✅ **Simple API** - one function for both
2. ✅ **No duplication** - single implementation with dispatch
3. ✅ **Backward compatible** - existing code works
4. ✅ **Efficient** - indexed mode for large MDPs
5. ✅ **Clear** - type determines behavior

This is the **simplest possible API** that supports **both planning and learning** efficiently.
