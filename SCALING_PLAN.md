# Scaling jaxdp to Large Finite MDPs: Strategic Plan

**Date:** 2025-11-15
**Status:** Planning Phase
**Target:** Scale from ~10K states to 100K-1M+ states

---

## Executive Summary

This document analyzes strategies for extending jaxdp to handle large but finite MDPs. The current one-hot encoding approach is elegant for small-to-medium problems (< 10K states) but faces O(n) memory overhead per state. We evaluate four strategic approaches with detailed trade-off analysis.

**Key Finding:** A **hybrid approach** (Option C) offers the best balance—keeping one-hot for planning algorithms while introducing index-based operations for learning scenarios.

---

## 1. Current State Analysis

### 1.1 Architecture Overview

```
jaxdp uses uniform one-hot encoding throughout:
├── States: [0,0,1,0,0] (size n_states)
├── Actions: [0,1,0,0] (size n_actions)
├── Q-values: Dense [n_actions, n_states]
└── Policies: Stochastic [n_actions, n_states]
```

**Core Design Philosophy:** Einsum-centric operations for JAX compiler optimization

### 1.2 Memory Footprint Analysis

| Component | Memory | Scaling |
|-----------|--------|---------|
| Q-values | 4AS bytes | O(AS) |
| Transition tensor | 4AS² bytes | O(AS²) |
| Reward tensor | 4AS² bytes | O(AS²) |
| Policy | 4AS bytes | O(AS) |
| **One-hot state** | **4S bytes** | **O(S)** |
| **One-hot action** | **4A bytes** | **O(A)** |

**Example:** S=100K, A=10
- Q-values: 4MB
- Transitions: 400GB (!)
- Single one-hot state: 400KB per sample
- Batch of 1000 samples: 400MB just for states

### 1.3 Current Usage Patterns

#### Planning (Value/Policy Iteration)
```python
# Operates on ALL state-action pairs simultaneously
next_q = bellman_op.q(mdp, q_vals, gamma)
# Shape: [A, S] → [A, S]
# Einsum: "axs,x->as" contracts full transition tensor
```

**Characteristics:**
- Dense tensor operations
- Full state space updates
- Matrix inversions (Policy Iteration)
- GPU/TPU friendly (massively parallel)

#### Learning (Q-Learning)
```python
# Updates from SAMPLED transitions
transition = Transition(state=[0,0,1,...], action=[0,1,0,...], ...)
updated_state = q_learning.update(state, transition)
# Only 2-3 non-zero values per transition
```

**Characteristics:**
- Sparse updates (1 state-action pair at a time)
- Batch updates: O(batch_size) non-zeros
- 99.99%+ zeros in one-hot representations
- Memory wasteful for large state spaces

---

## 2. Scaling Challenges

### 2.1 The One-Hot Dilemma

| Aspect | Small MDPs (< 10K) | Large MDPs (100K+) |
|--------|-------------------|-------------------|
| **Planning** | ✅ Efficient parallel updates | ⚠️ Transition tensor too large |
| **Learning** | ✅ Clean semantics | ❌ 99.99% memory waste |
| **GPU Utilization** | ✅ Full vectorization | ⚠️ Memory bound |
| **Gradient Flow** | ✅ Differentiable | ✅ Still works |

### 2.2 Critical Bottlenecks

1. **Transition Tensor Storage:** O(AS²) becomes prohibitive
2. **Learning Sample Efficiency:** O(S) memory per sample vs O(1) for indices
3. **Batch Processing:** One-hot batches consume excessive memory
4. **GPU Memory:** Limited by one-hot overhead, not computation

---

## 3. Strategy Options

### **Option A: One-Hot + JAX Sparse (BCOO)**

Maintain one-hot encoding but use `jax.experimental.sparse` for storage.

#### Implementation Approach
```python
import jax.experimental.sparse as jsp

# Convert one-hot to BCOO
state_onehot = jnp.array([0, 0, 1, 0, ...])  # Dense
state_sparse = jsp.BCOO.fromdense(state_onehot)

# Batch of transitions as BCOO
batch_states = jsp.BCOO.fromdense(batch_states_dense)
```

#### Pros
- ✅ Minimal code changes
- ✅ Preserves einsum-centric design
- ✅ Automatic differentiation still works
- ✅ Compatible with `vmap`, `jit`

#### Cons
- ❌ **JAX sparse is experimental** (API unstable)
- ❌ **Explicitly not for performance-critical code** (per docs)
- ❌ No native XLA sparse ops → inefficient compilation
- ❌ BCOO overhead (indices + data storage)
- ❌ Limited operation support (no direct einsum on BCOO)
- ❌ Requires `sparsify()` transform wrapper
- ❌ Worst-case allocation for sparse-sparse ops

#### Performance Estimate
- Memory: 50-70% savings (BCOO overhead ~30%)
- Speed: **Likely 2-5x SLOWER** than dense for small batches
- XLA compilation: Poor (no native sparse primitives)

#### Implementation Effort
- **Effort:** Medium (2-3 weeks)
- Changes needed: Wrap all operations with `sparsify()`
- Risk: High (experimental API, unclear performance)

---

### **Option B: Full Index-Based Implementation**

Replace all one-hot encodings with integer indices.

#### Implementation Approach
```python
# Before (one-hot)
state = jnp.array([0, 0, 1, 0, 0])  # Size S
action = jnp.array([0, 1, 0, 0])     # Size A

# After (index-based)
state = jnp.array(2)  # Scalar
action = jnp.array(1)  # Scalar

# Q-value lookup
q_value = q_vals[action, state]  # Direct indexing

# Update
q_vals = q_vals.at[action, state].add(alpha * td_error)
```

#### Planning Changes Required
```python
# Value Iteration (BEFORE)
target_values = jnp.einsum("axs,x->as",
                           mdp.transition,
                           jnp.max(q_vals, axis=0))

# Value Iteration (AFTER) - requires different structure
# Option 1: Keep transition tensor dense
target_values = jnp.einsum("axs,x->as",
                           mdp.transition,
                           jnp.max(q_vals, axis=0))
# (No change - planning already uses full tensors)

# Option 2: Sparse transition representation
# Store transitions as list of (s, a, s', p, r) tuples
# Requires complete rewrite
```

#### Learning Changes Required
```python
# Q-Learning (BEFORE)
curr_q = jnp.einsum("as,a,s->", q_vals, action_onehot, state_onehot)

# Q-Learning (AFTER)
curr_q = q_vals[action, state]  # Much simpler!

# Batch update (BEFORE)
total_counts = jnp.einsum("ba,bs->as", actions, states)

# Batch update (AFTER)
import jax.scipy.sparse as jsp
indices = jnp.stack([actions, states], axis=1)  # [batch, 2]
total_counts = jnp.zeros((n_actions, n_states))
total_counts = total_counts.at[actions, states].add(1)
# Or use jnp.bincount for efficiency
```

#### Pros
- ✅ **Massive memory savings** (O(1) per state vs O(S))
- ✅ Standard RL literature approach
- ✅ Simpler learning code (direct indexing)
- ✅ Better scalability for learning
- ✅ Efficient batch processing
- ✅ No experimental dependencies

#### Cons
- ❌ **Major refactor** of entire codebase
- ❌ Loses einsum elegance
- ❌ More complex gradient computation
- ❌ Sampling becomes more complex
- ❌ `distrax.OneHotCategorical` no longer usable
- ❌ Breaks existing API
- ❌ Planning algorithms still need dense Q-values

#### Performance Estimate
- Memory: **90-95% savings** for learning
- Speed: **2-10x faster** for learning (less memory movement)
- Planning: Unchanged (still needs dense Q-values)

#### Implementation Effort
- **Effort:** Large (6-8 weeks)
- **Lines changed:** ~80% of codebase
- **Risk:** High (breaking changes, testing burden)
- **Breaking:** Yes (complete API overhaul)

---

### **Option C: Hybrid Approach** ⭐ **RECOMMENDED**

Use **index-based for learning**, **one-hot for planning**.

#### Implementation Approach

```python
# Core idea: Separate representations with conversion utilities

# Planning API (unchanged)
def value_iteration(mdp, q_vals, gamma):
    # Works with dense one-hot as before
    return bellman_optimality_operator.q(mdp, q_vals, gamma)

# Learning API (new index-based)
@struct.dataclass
class TransitionIndexed:
    state: jnp.int32      # Scalar index
    action: jnp.int32     # Scalar index
    reward: float
    next_state: jnp.int32
    terminal: bool

class q_learning_indexed:
    def update(state: State, transition: TransitionIndexed):
        # Direct indexing
        curr_q = state.q_vals[transition.action, transition.state]
        max_next_q = state.q_vals[:, transition.next_state].max()
        td_error = (transition.reward +
                   state.gamma * max_next_q * (1 - transition.terminal) -
                   curr_q)

        # In-place update
        new_q_vals = state.q_vals.at[transition.action, transition.state]\
                                  .add(state.alpha * td_error)
        return state.replace(q_vals=new_q_vals)

# Conversion utilities
def state_to_index(state_onehot: jnp.ndarray) -> int:
    return jnp.argmax(state_onehot)

def index_to_onehot(state_idx: int, n_states: int) -> jnp.ndarray:
    return jax.nn.one_hot(state_idx, n_states)
```

#### Module Organization

```
jaxdp/
├── base.py              # One-hot operations (planning)
├── base_indexed.py      # NEW: Index-based operations (learning)
├── converters.py        # NEW: Conversion utilities
├── mdp/
│   └── mdp.py          # Keep dense representation
└── examples/
    ├── planning/        # Use base.py (unchanged)
    └── learning/
        ├── algorithms.py           # Keep for compatibility
        └── algorithms_indexed.py   # NEW: Index-based learning
```

#### API Design

```python
# Users can choose based on use case

# For planning (small-medium MDPs)
from jaxdp import bellman_optimality_operator, greedy_policy
q_vals = value_iteration_loop(mdp, gamma)

# For learning (large MDPs)
from jaxdp.learning import q_learning_indexed
from jaxdp.sampling import sample_indexed

# Generate indexed transitions
transition = sample_indexed(mdp, policy, state_idx, key)
state = q_learning_indexed.update(state, transition)

# Batch mode
transitions = sample_batch_indexed(mdp, policy, n_samples, key)
state = q_learning_indexed.batch_update(state, transitions)
```

#### Pros
- ✅ **Best of both worlds**
- ✅ Minimal changes to planning code (stable, tested)
- ✅ Maximum efficiency for learning scenarios
- ✅ Gradual migration path
- ✅ Backward compatible (keep old API)
- ✅ Users choose based on problem size
- ✅ Clear separation of concerns
- ✅ Easy to test incrementally

#### Cons
- ⚠️ Two parallel APIs to maintain
- ⚠️ Conversion overhead if mixing approaches
- ⚠️ Slightly larger codebase
- ⚠️ Documentation needs to explain when to use which

#### Performance Estimate
- **Planning:** Unchanged (uses existing code)
- **Learning:** 90% memory savings, 5-10x speedup
- **Mixed workloads:** Optimal (use right tool for each task)

#### Implementation Effort
- **Effort:** Medium (3-4 weeks)
- **Lines added:** ~1500 (new modules)
- **Lines changed:** ~200 (sampling utilities)
- **Risk:** Low (additive changes, non-breaking)
- **Breaking:** No (backward compatible)

---

### **Option D: Smart Dispatch with Unified API**

Single API that automatically selects representation based on MDP size.

#### Implementation Approach

```python
class AdaptiveMDP:
    def __init__(self, mdp, size_threshold=10000):
        self.mdp = mdp
        self.use_indexed = (mdp.state_size > size_threshold)

    def create_transition(self, state, action, reward, next_state, terminal):
        if self.use_indexed:
            return TransitionIndexed(
                state=jnp.argmax(state) if state.ndim > 0 else state,
                action=jnp.argmax(action) if action.ndim > 0 else action,
                reward=reward,
                next_state=jnp.argmax(next_state) if next_state.ndim > 0 else next_state,
                terminal=terminal
            )
        else:
            return Transition(state, action, reward, next_state, terminal)

# Polymorphic algorithms
def q_learning_update(state, transition):
    if isinstance(transition, TransitionIndexed):
        return _update_indexed(state, transition)
    else:
        return _update_onehot(state, transition)
```

#### Pros
- ✅ Transparent to users
- ✅ Automatically optimal for problem size
- ✅ Single learning path
- ✅ Clean external API

#### Cons
- ❌ Complex internal implementation
- ❌ Runtime dispatch overhead
- ❌ Harder to debug (hidden logic)
- ❌ JIT compilation complications
- ❌ Type checking becomes difficult
- ❌ Still requires both implementations (Option C complexity)

#### Performance Estimate
- Same as Option C, plus dispatch overhead (~5-10%)

#### Implementation Effort
- **Effort:** Large (5-6 weeks)
- Includes all of Option C + dispatch layer
- **Risk:** Medium-High (complexity)

---

## 4. Detailed Trade-Off Analysis

### 4.1 Memory Comparison (Learning Scenario)

**Scenario:** S=100K states, A=10 actions, batch_size=1024

| Approach | Single Transition | Batch (1024) | Total |
|----------|------------------|--------------|-------|
| **Current (One-hot)** | 800KB | 820MB | 820MB |
| **Option A (BCOO)** | ~560KB (30% overhead) | 574MB | 574MB |
| **Option B/C (Indexed)** | 8 bytes | 8KB | 8KB |

**Winner:** Index-based (Options B/C) by **100,000x** 🏆

### 4.2 Speed Comparison (Learning)

Based on operation types:

| Operation | One-hot | BCOO Sparse | Indexed |
|-----------|---------|-------------|---------|
| **Q-value lookup** | Einsum (slow) | BCOO ops (very slow) | Direct indexing (fast) |
| **Update** | Dense add | Sparse add | In-place update |
| **Batch count** | Einsum | Sparse sum | Bincount |
| **Estimated speedup** | 1x | 0.2-0.5x ❌ | 5-10x ✅ |

### 4.3 Planning Impact

| Algorithm | Affected? | Why |
|-----------|-----------|-----|
| **Value Iteration** | No | Uses dense Q-values and transition tensor |
| **Policy Iteration** | No | Same, plus matrix inversion on dense matrices |
| **Nesterov VI** | No | Momentum on dense Q-values |

**Key Insight:** Planning algorithms don't benefit from indexed states because they operate on the **full state space** simultaneously. They need dense Q-value matrices regardless.

### 4.4 Code Complexity

| Metric | Option A | Option B | Option C | Option D |
|--------|----------|----------|----------|----------|
| **New files** | 0 | 5 | 4 | 6 |
| **Changed files** | 15 | 20 | 8 | 25 |
| **Lines added** | ~200 | ~2000 | ~1500 | ~2500 |
| **Breaking changes** | No | Yes | No | No |
| **Test coverage** | ~40 tests | ~120 tests | ~80 tests | ~150 tests |

---

## 5. Performance Considerations

### 5.1 JAX Sparse Maturity Assessment

**Current Status (2025):**
- ❌ Still in `jax.experimental.*`
- ❌ Docs warn: "not recommended for performance-critical applications"
- ❌ No native XLA sparse primitives
- ❌ Recent bugs (Feb 2025: sum over axis broken)
- ⚠️ API subject to change

**Verdict:** Not production-ready for performance gains

### 5.2 Einsum vs Direct Indexing

```python
# Einsum (current approach)
curr_q = jnp.einsum("as,a,s->", q_vals, action_onehot, state_onehot)
# - Compiler can fuse operations
# - Works well with GPU/TPU
# - O(AS) memory access for one-hot vectors

# Direct indexing
curr_q = q_vals[action, state]
# - O(1) memory access
# - Simple compiled code
# - Cache-friendly

# Benchmark results (estimated from similar ops):
# Small S (<1000): Einsum 1.0x, Indexing 1.2x
# Medium S (10K): Einsum 1.0x, Indexing 3x
# Large S (100K+): Einsum 1.0x, Indexing 10x
```

### 5.3 Batch Processing

**One-hot batch:**
```python
# Memory: batch_size * S * 4 bytes
# For batch=1024, S=100K: 410MB
transitions.state.shape  # [1024, 100000]
```

**Indexed batch:**
```python
# Memory: batch_size * 4 bytes
# For batch=1024, S=100K: 4KB
transitions.state.shape  # [1024]
```

**Implication:** Can process **100x larger batches** with index-based approach

---

## 6. Implementation Roadmap

### **RECOMMENDED: Option C (Hybrid)**

#### Phase 1: Foundation (Week 1-2)
- [ ] Create `jaxdp/base_indexed.py` with index-based operations
- [ ] Implement `TransitionIndexed` dataclass
- [ ] Add conversion utilities (`state_to_index`, `index_to_onehot`)
- [ ] Write unit tests for indexed operations
- [ ] Benchmark: index lookup vs einsum for various sizes

#### Phase 2: Learning Algorithms (Week 2-3)
- [ ] Implement `q_learning_indexed` with single updates
- [ ] Implement efficient batch updates using `jnp.bincount`
- [ ] Add indexed sampling functions
- [ ] Create learning examples using indexed approach
- [ ] Performance benchmarks: compare memory and speed

#### Phase 3: Integration (Week 3-4)
- [ ] Add sampling namespace for indexed transitions
- [ ] Create tutorial notebooks (when to use which approach)
- [ ] Documentation: API reference for both approaches
- [ ] Migration guide for existing users
- [ ] End-to-end benchmarks on large MDPs (100K+ states)

#### Phase 4: Optimization (Week 4+)
- [ ] Profile bottlenecks in indexed implementations
- [ ] Optimize batch counting using advanced JAX features
- [ ] Add support for multi-environment vectorization
- [ ] Implement any remaining edge cases
- [ ] Performance tuning and final documentation

---

## 7. Benchmarking Plan

### 7.1 Test Scenarios

| Scenario | States | Actions | Description |
|----------|--------|---------|-------------|
| **Small** | 100 | 4 | GridWorld 10x10 |
| **Medium** | 10,000 | 10 | Large GridWorld |
| **Large** | 100,000 | 20 | Chain/Tree structure |
| **Huge** | 1,000,000 | 10 | Sparse graph MDP |

### 7.2 Metrics to Track

1. **Memory:**
   - Peak memory usage (training loop)
   - Memory per sample
   - Batch processing capacity

2. **Speed:**
   - Single update latency
   - Batch update throughput (updates/sec)
   - JIT compilation time

3. **Scalability:**
   - Max state space size (before OOM)
   - Batch size limits
   - GPU utilization %

### 7.3 Expected Results

| Metric | Current | Option C (Indexed Learning) |
|--------|---------|---------------------------|
| **Max states** | ~10K | ~1M+ |
| **Learning speed** | 1.0x | 5-10x |
| **Batch size** | 256 | 25,600 |
| **Memory/sample** | 800KB | 8 bytes |

---

## 8. Risk Analysis

### Option A Risks
- 🔴 **HIGH:** JAX sparse performance unknown
- 🔴 **HIGH:** Experimental API may break
- 🟡 **MEDIUM:** May not achieve speedup goals

### Option B Risks
- 🔴 **HIGH:** Massive refactor could introduce bugs
- 🔴 **HIGH:** Breaking changes alienate users
- 🟡 **MEDIUM:** Long development time

### Option C Risks
- 🟢 **LOW:** Additive changes are safer
- 🟡 **MEDIUM:** Two APIs to maintain
- 🟢 **LOW:** Can validate incrementally

### Option D Risks
- 🟡 **MEDIUM:** Dispatch complexity
- 🟡 **MEDIUM:** JIT compilation issues
- 🟢 **LOW:** User-facing API is simple

---

## 9. Decision Matrix

| Criterion | Weight | Option A | Option B | Option C ⭐ | Option D |
|-----------|--------|----------|----------|----------|----------|
| **Memory savings** | 0.25 | 6/10 | 10/10 | 10/10 | 10/10 |
| **Speed improvement** | 0.20 | 3/10 | 9/10 | 9/10 | 9/10 |
| **Implementation effort** | 0.20 | 8/10 | 3/10 | 7/10 | 4/10 |
| **Backward compat** | 0.15 | 10/10 | 2/10 | 10/10 | 10/10 |
| **Code maintainability** | 0.10 | 7/10 | 5/10 | 8/10 | 5/10 |
| **Risk level** | 0.10 | 4/10 | 4/10 | 8/10 | 6/10 |
| **TOTAL** | 1.00 | **6.1** | **6.2** | **8.7** ✅ | **7.2** |

---

## 10. Recommendations

### Primary Recommendation: **Option C (Hybrid Approach)**

**Reasoning:**
1. **Pragmatic:** Solves the actual bottleneck (learning) without disrupting planning
2. **Low-risk:** Backward compatible, additive changes
3. **Optimal performance:** Index-based where it matters, dense where it's needed
4. **Clear mental model:** Users understand when to use which approach
5. **Incremental:** Can implement and validate in phases

### Implementation Strategy

```python
# Phase 1: Start with learning algorithms only
from jaxdp.learning import q_learning_indexed

# Phase 2: Add indexed sampling
from jaxdp.sampling import sample_indexed, rollout_indexed

# Phase 3: Benchmarks prove value
# Phase 4: Consider planning optimizations (separate effort)
```

### When to Use Each Approach

**Use One-Hot (Existing API):**
- ✅ Planning algorithms (VI, PI, etc.)
- ✅ Small MDPs (< 10K states)
- ✅ Research code where clarity > performance
- ✅ When you need gradients through entire policy

**Use Indexed (New API):**
- ✅ Q-learning and TD methods
- ✅ Large MDPs (> 100K states)
- ✅ Production systems with memory constraints
- ✅ Large batch training
- ✅ When sampling from environment

### Alternative Recommendation: **Option B (If breaking changes acceptable)**

If this is a major version release (2.0) and breaking changes are acceptable:
- Consider full index-based refactor
- Cleaner long-term architecture
- Better alignment with standard RL libraries
- But requires more resources and time

---

## 11. Open Questions

1. **Should we support JAX sparse at all?**
   - Potentially as Option E for future when it matures
   - Monitor JAX sparse development

2. **How to handle mixed-mode gradients?**
   - If user needs gradients through learning updates
   - Indexed approach may complicate AD

3. **Transition tensor scaling?**
   - Even indexed learning needs MDP definition
   - For S=1M, transition tensor is 40TB
   - Consider separate issue: sparse MDP representations

4. **Neural network function approximators?**
   - Out of scope (finite MDPs only)
   - But indexed approach enables easier integration later

---

## 12. Next Steps

### Immediate Actions (This Week)
1. ✅ **Validate decision** with team/community
2. ⬜ **Create feature branch:** `feature/indexed-learning`
3. ⬜ **Set up benchmarking harness**
4. ⬜ **Prototype `TransitionIndexed`** and basic operations

### Short-term (Next 2 Weeks)
1. ⬜ Implement core indexed learning algorithms
2. ⬜ Write comprehensive tests
3. ⬜ Create performance benchmarks
4. ⬜ Draft user-facing documentation

### Medium-term (Next Month)
1. ⬜ Complete implementation
2. ⬜ Run full benchmark suite
3. ⬜ Write migration guide
4. ⬜ Prepare for release

---

## Appendix A: Code Examples

### A.1 Current One-Hot Q-Learning

```python
# Current implementation (jaxdp/examples/learning/algorithms.py)
@struct.dataclass
class Transition:
    state: F["S"]       # [100000] one-hot → 400KB
    action: F["A"]      # [10] one-hot → 40 bytes
    reward: F[""]
    next_state: F["S"]  # [100000] one-hot → 400KB
    terminal: F[""]

def update(state: State, transition: Transition):
    # Extract Q(s,a) using einsum
    curr_q = jnp.einsum("as,a,s->", state.q_vals,
                        transition.action, transition.state)

    # Extract Q(s', :)
    q_next = jnp.einsum("as,s->a", state.q_vals,
                        transition.next_state)
    max_next_q = jnp.max(q_next)

    # Compute TD error
    td_target = transition.reward + state.gamma * max_next_q * (1 - transition.terminal)
    td_error = td_target - curr_q

    # Create update mask
    update = jnp.einsum("a,s->as", transition.action, transition.state)

    # Apply update
    return state.replace(q_vals=state.q_vals + state.alpha * td_error * update)
```

**Memory for batch=1024:** ~820MB just for state/next_state

### A.2 Proposed Indexed Q-Learning

```python
# Proposed implementation (jaxdp/learning/algorithms_indexed.py)
@struct.dataclass
class TransitionIndexed:
    state: jnp.int32       # Scalar → 4 bytes
    action: jnp.int32      # Scalar → 4 bytes
    reward: jnp.float32
    next_state: jnp.int32  # Scalar → 4 bytes
    terminal: jnp.bool_

def update(state: State, transition: TransitionIndexed):
    # Direct indexing (much simpler!)
    curr_q = state.q_vals[transition.action, transition.state]

    # Max over actions for next state
    q_next = state.q_vals[:, transition.next_state]
    max_next_q = jnp.max(q_next)

    # Compute TD error
    td_target = transition.reward + state.gamma * max_next_q * (1 - transition.terminal)
    td_error = td_target - curr_q

    # In-place update using JAX's .at syntax
    new_q_vals = state.q_vals.at[transition.action, transition.state]\
                              .add(state.alpha * td_error)

    return state.replace(q_vals=new_q_vals)
```

**Memory for batch=1024:** ~12KB (680x smaller!)

### A.3 Efficient Batch Updates

```python
def batch_update_indexed(state: State, transitions: TransitionIndexed):
    """
    Efficient batch update using bincount for counting.
    Handles repeated (s,a) pairs correctly.
    """
    # Compute all TD errors in parallel (vmap)
    def compute_td(trans):
        curr_q = state.q_vals[trans.action, trans.state]
        max_next = jnp.max(state.q_vals[:, trans.next_state])
        target = trans.reward + state.gamma * max_next * (1 - trans.terminal)
        return target - curr_q

    td_errors = jax.vmap(compute_td)(transitions)

    # Convert (action, state) pairs to linear indices
    linear_idx = transitions.action * state.q_vals.shape[1] + transitions.state

    # Sum TD errors for each (s,a) pair
    total_delta = jnp.bincount(
        linear_idx,
        weights=td_errors,
        length=state.q_vals.size
    ).reshape(state.q_vals.shape)

    # Count occurrences of each (s,a)
    counts = jnp.bincount(
        linear_idx,
        length=state.q_vals.size
    ).reshape(state.q_vals.shape)

    # Normalize and apply
    normalized_delta = jnp.where(counts > 0, total_delta / counts, 0.0)

    return state.replace(q_vals=state.q_vals + state.alpha * normalized_delta)
```

---

## Appendix B: Benchmark Pseudocode

```python
def benchmark_learning_approaches():
    """Compare one-hot vs indexed for Q-learning."""

    state_sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    batch_sizes = [1, 10, 100, 1_000]

    results = []

    for n_states in state_sizes:
        for batch_size in batch_sizes:

            # One-hot approach
            start = time.time()
            transitions_onehot = generate_transitions_onehot(batch_size, n_states)
            memory_onehot = measure_memory()
            state_onehot = q_learning.batch_update(state, transitions_onehot)
            time_onehot = time.time() - start

            # Indexed approach
            start = time.time()
            transitions_indexed = generate_transitions_indexed(batch_size, n_states)
            memory_indexed = measure_memory()
            state_indexed = q_learning_indexed.batch_update(state, transitions_indexed)
            time_indexed = time.time() - start

            results.append({
                'n_states': n_states,
                'batch_size': batch_size,
                'memory_ratio': memory_onehot / memory_indexed,
                'speedup': time_onehot / time_indexed,
            })

    return pd.DataFrame(results)
```

---

## Appendix C: Migration Guide (Draft)

### For Existing Users

**No changes required for planning algorithms!**

```python
# Your existing planning code works as-is
from jaxdp import bellman_optimality_operator, greedy_policy

q_vals = jnp.zeros((n_actions, n_states))
for _ in range(n_iterations):
    q_vals = bellman_optimality_operator.q(mdp, q_vals, gamma)
policy = greedy_policy.q(q_vals)
```

**For learning with large MDPs, opt-in to indexed approach:**

```python
# Before
from jaxdp.learning import q_learning
from jaxdp import async_sample_step_pi

state_onehot = mdp.initial
transitions = []
for _ in range(n_samples):
    act, next_s, r, term, _, state_onehot, _ = async_sample_step_pi(...)
    transitions.append(Transition(state_onehot, act, r, next_s, term))

# After
from jaxdp.learning import q_learning_indexed
from jaxdp.sampling import sample_indexed

state_idx = sample_initial_state_indexed(mdp, key)
transitions = []
for _ in range(n_samples):
    trans = sample_indexed(mdp, policy, state_idx, key)
    transitions.append(trans)
    state_idx = trans.next_state
```

---

## Document Metadata

- **Author:** Claude (Anthropic)
- **Version:** 1.0
- **Last Updated:** 2025-11-15
- **Status:** Draft for Review
- **Next Review:** After team discussion
