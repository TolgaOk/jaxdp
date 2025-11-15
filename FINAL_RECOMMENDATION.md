# Final Recommendation: Scaling jaxdp to Large Finite MDPs

## Executive Summary

**Goal:** Scale jaxdp to handle 100K-1M+ state MDPs efficiently

**Solution:** Type-polymorphic base primitives that auto-detect and optimize based on input type

**Impact:**
- 100,000x memory savings for learning (guaranteed)
- 2-10x speed improvement (estimated, needs benchmarking)
- Zero breaking changes
- Single simple API

---

## Understanding the Architecture

### What jaxdp IS

```
jaxdp/                    ← THE LIBRARY (what we ship)
├── base.py              ← Bellman operators, sampling primitives
├── mdp/                 ← MDP implementations
└── typehints.py

examples/                 ← HOW TO USE the library (not shipped)
├── planning/            ← Example: Value Iteration, Policy Iteration
└── learning/            ← Example: Q-learning, TD learning
```

**Key insight:** Learning is NOT in the library, it's an example showing how to use the primitives.

---

## The Core Problem

### Current Situation

**Base primitives return one-hot:**
```python
# jaxdp/base.py
next_state, reward, ... = async_sample_step(mdp, action, state, ...)
# state: [S] one-hot array
# action: [A] one-hot array
# next_state: [S] one-hot array
```

**Planning examples (fine):**
```python
# examples/planning/algorithms.py
q_vals = bellman_optimality_operator.q(mdp, q_vals, gamma)
# Works on full Q-table [A, S] - one-hot is okay
```

**Learning examples (wasteful):**
```python
# examples/learning/algorithms.py
state = jnp.array([0,0,0,...,1,...,0])  # 100,000 elements!
action, next_state, ... = async_sample_step_pi(mdp, policy, state, ...)
# Memory: 800 MB for batch of 1000 transitions
```

---

## The Solution

### Type-Polymorphic Base Primitives

**Modify `jaxdp/base.py` to accept BOTH integer and array inputs:**

```python
# jaxdp/base.py

def async_sample_step(mdp, action, state, episode_step, episode_length, key):
    """
    Sample MDP transition.

    Args:
        action: int (index) OR array (one-hot)
        state: int (index) OR array (one-hot)

    Returns:
        next_state: SAME TYPE as input
        reward: scalar
        ...

    Auto-detects type and dispatches to efficient implementation.
    """
    if jnp.ndim(state) == 0:  # scalar → indexed
        return _async_sample_step_indexed(mdp, action, state, ...)
    else:  # array → one-hot
        return _async_sample_step_onehot(mdp, action, state, ...)
```

### Indexed Implementation (New)

```python
def _async_sample_step_indexed(mdp, action: int, state: int, ...):
    """Efficient implementation for integer indices."""
    # Direct array indexing (not einsum)
    next_state_probs = mdp.transition[action, :, state]
    next_state = jrd.choice(key, mdp.state_size, p=next_state_probs)  # Returns int!
    reward = mdp.reward[action, state, next_state]
    terminal = mdp.terminal[next_state]
    # ...
    return next_state, reward, terminal, ...
```

### One-Hot Implementation (Existing)

```python
def _async_sample_step_onehot(mdp, action: Array, state: Array, ...):
    """Backward-compatible implementation for one-hot arrays."""
    # Current implementation - unchanged
    next_state_p = jnp.einsum("a,axs,s->x", action, mdp.transition, state)
    next_state = distrax.OneHotCategorical(probs=next_state_p).sample(seed=key)
    reward = jnp.einsum("asx,a,s,x->", mdp.reward, action, state, next_state)
    # ...
    return next_state, reward, terminal, ...
```

---

## What Changes

### In Base Library (`jaxdp/base.py`)

**Modify these functions:**
1. `async_sample_step()` - Add type detection and dispatch
2. `async_sample_step_pi()` - Use updated `async_sample_step()`
3. `sample_from()` - Add optional state parameter with type support

**DON'T modify these:**
- `bellman_optimality_operator.q()` ✓ (works on Q-tables)
- `bellman_operator.q()` ✓ (works on Q-tables)
- `greedy_policy.q()` ✓ (works on Q-tables)
- `policy_evaluation.q()` ✓ (works on Q-tables)
- All other Bellman/policy functions ✓

**Why most don't change:** They operate on Q-value tables which are always dense `[A, S]` matrices. Planning and learning both use the same Q-tables.

### In Examples

**Planning (`examples/planning/`):** NO CHANGE
- Already uses Bellman operators on Q-tables
- Doesn't use sampling much

**Learning (`examples/learning/`):** CAN NOW USE INDICES
- Old one-hot code still works (backward compatible)
- New indexed code is more efficient
- User chooses by input type

---

## Usage Examples

### Planning (Unchanged)

```python
from jaxdp import bellman_optimality_operator
from jaxdp.mdp import GridWorld

mdp = GridWorld(10, 10)
q_vals = jnp.zeros((mdp.action_size, mdp.state_size))

# Value iteration - works exactly as before
for _ in range(100):
    q_vals = bellman_optimality_operator.q(mdp, q_vals, gamma=0.99)
```

### Learning - Small MDP (Either Works)

```python
from jaxdp import async_sample_step_pi

mdp = GridWorld(10, 10)  # 100 states

# Option 1: Use indices (recommended)
state = 0  # Integer
action, next_state, reward, ... = async_sample_step_pi(mdp, policy, state, ...)
# Returns integers

# Option 2: Use one-hot (backward compatible)
state = jnp.array([1,0,0,...,0])  # One-hot
action, next_state, reward, ... = async_sample_step_pi(mdp, policy, state, ...)
# Returns arrays
```

### Learning - Large MDP (MUST Use Indices)

```python
from jaxdp import async_sample_step_pi

mdp = GridWorld(316, 316)  # 100K states - TOO LARGE for one-hot!

# Use indices
state = 0  # Integer
for _ in range(1000):
    action, next_state, reward, terminal, _, state, _ = \
        async_sample_step_pi(mdp, policy, state, ...)

    # Update Q-value using indexed access
    curr_q = q_vals[action, state]
    max_next_q = jnp.max(q_vals[:, next_state])
    target = reward + gamma * max_next_q
    q_vals = q_vals.at[action, state].add(alpha * (target - curr_q))

# Memory: 12 KB instead of 800 MB!
```

---

## Benefits

### 1. Simple API

**One function for both representations:**
```python
# Same function!
next_state, reward, ... = async_sample_step(mdp, action, state, ...)

# Input type determines behavior:
# - Integers → indexed (efficient)
# - Arrays → one-hot (compatible)
```

### 2. No Code Duplication

- Single `async_sample_step()` function (not two separate functions)
- Type detection handles dispatch internally
- Examples choose representation by using int or array

### 3. Backward Compatible

- Existing code with arrays → works unchanged
- New code with integers → automatically efficient
- No breaking changes to API

### 4. Automatic Optimization

- User doesn't think about representation
- Just use integers → automatically fast
- Use arrays → automatically compatible

### 5. Clear Semantics

- Planning: operates on Q-tables → use Bellman operators
- Learning: samples transitions → choose int (fast) or array (compat)

---

## Implementation Plan

### Phase 1: Base Primitives (Week 1-2)

**Modify `jaxdp/base.py`:**
```
- Add _is_scalar(x) helper
- Implement _async_sample_step_indexed()
- Modify async_sample_step() to dispatch
- Update sample_from() to support state parameter
- Update async_sample_step_pi()
```

**Estimated changes:** ~100-150 lines

### Phase 2: Testing (Week 2)

```
- Test indexed sampling returns correct types
- Test one-hot sampling still works (backward compat)
- Test error handling (can't mix types)
- Benchmark memory usage
- Benchmark speed (validate estimates)
```

### Phase 3: Examples (Week 3)

```
- Add examples/learning/q_learning_indexed.py
- Update documentation
- Create tutorial notebook
- Performance comparison
```

### Phase 4: Validation (Week 4)

```
- Run benchmarks on large MDPs
- Validate memory savings
- Validate speed improvements
- Update estimates with real data
```

---

## Performance Impact

### Memory (Guaranteed)

| Scenario | One-Hot | Indexed | Savings |
|----------|---------|---------|---------|
| 1 transition (S=100K) | 800 KB | 8 bytes | 100,000x |
| Batch=1000 (S=100K) | 800 MB | 12 KB | 66,000x |
| Batch=1000 (S=1M) | 8 GB | 12 KB | 666,000x |

### Speed (Estimated - Needs Benchmarking)

| State Size | Expected Speedup | Confidence |
|------------|-----------------|------------|
| 100 | 1.0-1.5x | Low (overhead dominates) |
| 10,000 | 2-4x | Medium |
| 100,000 | 3-7x | Medium-High |
| 1,000,000 | 5-15x | High |

**Note:** These are theoretical estimates. Real benchmarks needed.

---

## Addressing Your Concerns

### 1. "How did you compute speed?"

**Answer:** I estimated from theory (O(AS) → O(1)), not measurements.

**Action:** Need to run real benchmarks to validate. Created methodology doc with what to measure.

**Safe claim:** Memory savings are guaranteed. Speed likely 2-10x but needs validation.

### 2. "How do we provide simple API for BOTH planning and learning?"

**Answer:** Type-polymorphic primitives in `jaxdp/base.py`.

**The API:**
- Planning: Uses Bellman operators (unchanged)
- Learning: Uses sampling primitives (accepts int or array)
- Same functions, different input types → automatic optimization

**User experience:**
```python
# Just use integers for learning
state = 3
next_state, reward, ... = async_sample_step(mdp, action, state, ...)

# Library handles the rest!
```

---

## Why This is the Simplest API

### No Duplication

**Bad approach (rejected):**
```python
# Don't do this!
from jaxdp import async_sample_step_onehot
from jaxdp import async_sample_step_indexed
# User has to choose and remember two APIs
```

**Good approach (recommended):**
```python
# Do this!
from jaxdp import async_sample_step
# Works with both - auto-detects type
```

### No Configuration

**Bad approach (rejected):**
```python
# Don't do this!
config = SamplingConfig(use_indexed=True)
state = async_sample_step(mdp, action, state, config=config)
# User has to configure
```

**Good approach (recommended):**
```python
# Do this!
state = 3  # Integer → indexed
next_state = async_sample_step(mdp, action, state, ...)
# Type determines behavior
```

### No Explicit Choice Needed

**Bad approach (rejected):**
```python
# Don't do this!
if mdp.state_size > 10000:
    from jaxdp.indexed import async_sample_step
else:
    from jaxdp.onehot import async_sample_step
# User has to make decision
```

**Good approach (recommended):**
```python
# Do this!
from jaxdp import async_sample_step
# Just use int or array - library optimizes
```

---

## Next Steps

### Immediate (This Week)

1. ✅ Complete planning documents
2. ⬜ Review with team
3. ⬜ Get approval for base library changes
4. ⬜ Create implementation branch

### Short-term (2-3 Weeks)

1. ⬜ Implement type-polymorphic primitives in `base.py`
2. ⬜ Add comprehensive tests
3. ⬜ Run benchmarks to validate estimates
4. ⬜ Create indexed learning examples

### Medium-term (1 Month)

1. ⬜ Complete documentation
2. ⬜ Tutorial notebooks
3. ⬜ Performance validation on real MDPs
4. ⬜ Prepare for release

---

## Summary

### The Question

**"How can we provide the simplest API such that both planning and learning use it and (at least in learning) it is index based?"**

### The Answer

**Type-polymorphic base primitives:**

1. **Bellman operators don't change** (they work on Q-tables, always dense)
2. **Sampling primitives accept int OR array** (auto-detect and dispatch)
3. **Planning examples unchanged** (use Bellman operators)
4. **Learning examples can use indices** (pass integers instead of arrays)

**The API:**
```python
from jaxdp import async_sample_step

# Use with integers (indexed - efficient)
next_state, reward, ... = async_sample_step(mdp, 1, 3, ...)

# Use with arrays (one-hot - compatible)
next_state, reward, ... = async_sample_step(mdp, [0,1,0], [0,0,0,1,0], ...)

# SAME FUNCTION. Different input type. Automatic optimization.
```

### Why This is Simple

1. ✅ One import: `from jaxdp import async_sample_step`
2. ✅ One function: `async_sample_step()`
3. ✅ No configuration needed
4. ✅ Type determines behavior (int=fast, array=compat)
5. ✅ Backward compatible (existing code works)
6. ✅ No user-facing complexity

**The user just chooses to use integers or arrays. The library does the rest.**

---

## Conclusion

This approach provides:
- ✅ **Simplest possible API** (single function, auto-detect)
- ✅ **Functional for both** (planning uses Bellman, learning uses sampling)
- ✅ **Efficient** (100,000x memory, 2-10x speed for learning)
- ✅ **Compatible** (zero breaking changes)
- ✅ **Clear** (type determines behavior)

Ready to implement once approved.
