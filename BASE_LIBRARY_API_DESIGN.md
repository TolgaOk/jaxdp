# Base Library API Design for Index Support

## Problem Statement

**Current situation:**
- `jaxdp/base.py` provides primitives (Bellman operators, sampling, policies)
- `examples/planning/` uses these primitives for VI, PI, etc.
- `examples/learning/` uses these primitives for Q-learning, etc.

**Issue:**
- All sampling primitives return **one-hot** states/actions
- This is fine for planning (operates on full state space)
- This is wasteful for learning (samples individual transitions)

**Goal:**
Design base primitives that:
1. ✅ Support BOTH one-hot and indexed representations
2. ✅ Work seamlessly for planning examples (full space operations)
3. ✅ Work efficiently for learning examples (sampled operations)
4. ✅ **Single simple API** - no duplication

---

## Current Base Primitives Analysis

### What Works Well (No Changes Needed)

```python
# Bellman operators work on Q-tables (always dense [A, S])
bellman_optimality_operator.q(mdp, q_vals, gamma)  # Returns [A, S]
bellman_operator.q(mdp, policy, q_vals, gamma)     # Returns [A, S]

# Policy extraction works on Q-tables
greedy_policy.q(q_vals)  # Takes [A, S], returns [A, S]
e_greedy_policy.q(q_vals, epsilon)  # Takes [A, S], returns [A, S]

# These are perfect - planning and learning both use Q-tables!
```

**These don't need to change** because Q-values are always dense matrices.

### What Needs Index Support

```python
# Sampling primitives - currently return one-hot
next_state_oh, reward, terminal, ... = async_sample_step(
    mdp, action_oh, state_oh, ...
)  # state_oh is [S] array → wasteful for learning

# Policy sampling - currently returns one-hot
action_oh = sample_from(policy, key)  # Returns [A, S] → wasteful

# These need index versions for efficient learning!
```

---

## Design Solution: Type-Polymorphic Primitives

### Core Idea

**Make primitives work with BOTH representations based on input type:**
- Input is **scalar** → treat as index, return indices
- Input is **array** → treat as one-hot, return one-hot

### Implementation Strategy

```python
# jaxdp/base.py

def async_sample_step(
    mdp: MDP,
    action: Union[int, Array],  # Can be index OR one-hot
    state: Union[int, Array],   # Can be index OR one-hot
    episode_step: Scalar,
    episode_length: int,
    key: PRNGKey
) -> tuple[...]:
    """
    Sample MDP transition - works with indices OR one-hot.

    Args:
        mdp: Markov Decision Process
        action: Action (int index or [A] one-hot)
        state: State (int index or [S] one-hot)
        episode_step: Current step count
        episode_length: Max episode length
        key: PRNG key

    Returns:
        next_state: Same type as input (index or one-hot)
        reward: Scalar
        terminal: Scalar
        timeout: Scalar
        new_state: After reset if terminal (same type as input)
        new_step: Updated step count

    Examples:
        # Indexed (efficient for learning)
        >>> next_s, r, term, ... = async_sample_step(mdp, 1, 3, ...)
        >>> type(next_s)  # int (scalar)

        # One-hot (compatible for planning/existing code)
        >>> next_s, r, term, ... = async_sample_step(
        ...     mdp, jnp.array([0,1,0]), jnp.array([0,0,0,1,0]), ...
        ... )
        >>> next_s.shape  # (5,) - one-hot
    """
    # Detect input type
    is_indexed = _is_scalar(state)

    if is_indexed:
        return _async_sample_step_indexed(mdp, action, state,
                                          episode_step, episode_length, key)
    else:
        return _async_sample_step_onehot(mdp, action, state,
                                        episode_step, episode_length, key)
```

---

## Detailed Implementation

### Helper Functions

```python
# jaxdp/base.py

def _is_scalar(x) -> bool:
    """Check if x is a scalar (index) vs array (one-hot)."""
    return jnp.ndim(x) == 0


def _is_indexed(state, action) -> bool:
    """Check if state/action are indexed."""
    return _is_scalar(state) and _is_scalar(action)
```

### Indexed Version (New)

```python
def _async_sample_step_indexed(
    mdp: MDP,
    action: int,
    state: int,
    episode_step: Scalar,
    episode_length: int,
    key: PRNGKey
) -> tuple[int, Scalar, Scalar, Scalar, int, Scalar]:
    """
    Indexed implementation - memory efficient.

    Returns integers for states, not one-hot arrays.
    """
    state_key, init_key = jrd.split(key, num=2)

    # Get transition probabilities for this (state, action)
    next_state_probs = mdp.transition[action, :, state]  # [S]

    # Sample next state (returns integer)
    next_state = jrd.choice(state_key, mdp.state_size, p=next_state_probs)

    # Get reward for this specific transition
    reward = mdp.reward[action, state, next_state]

    # Check if terminal
    terminal = mdp.terminal[next_state]

    # Update episode step
    episode_step = episode_step + 1
    timeout = episode_step >= episode_length
    done = jnp.logical_or(terminal, timeout)

    # Sample initial state if done
    init_state = jrd.choice(init_key, mdp.state_size, p=mdp.initial)

    # Reset state if done
    new_state = jax.lax.select(done, init_state, next_state)
    new_step = episode_step * (1 - done)

    return next_state, reward, terminal, timeout, new_state, new_step
```

### One-Hot Version (Existing)

```python
def _async_sample_step_onehot(
    mdp: MDP,
    action: Array,  # [A]
    state: Array,   # [S]
    episode_step: Scalar,
    episode_length: int,
    key: PRNGKey
) -> tuple[Array, Scalar, Scalar, Scalar, Array, Scalar]:
    """
    One-hot implementation - backward compatible.

    This is the current implementation (just renamed).
    """
    # Current implementation from base.py lines 410-464
    state_key, init_key = jrd.split(key, num=2)

    next_state_p = jnp.einsum("a,axs,s->x", action, mdp.transition, state)
    next_state = distrax.OneHotCategorical(
        probs=next_state_p, dtype="float"
    ).sample(seed=state_key)
    reward = jnp.einsum("asx,a,s,x->", mdp.reward, action, state, next_state)
    terminal = jnp.einsum("s,s->", mdp.terminal, next_state)

    episode_step = episode_step + 1
    timeout = episode_step >= episode_length
    done = jnp.logical_or(terminal, timeout)

    init_state = distrax.OneHotCategorical(
        probs=mdp.initial, dtype="float"
    ).sample(seed=init_key)
    new_state = next_state * (1 - done) + init_state * done
    new_step = episode_step * (1 - done)

    return next_state, reward, terminal, timeout, new_state, new_step
```

---

## Same Pattern for Other Primitives

### sample_from (Policy Sampling)

```python
def sample_from(
    policy: PiType,
    key: PRNGKey,
    state: Optional[Union[int, Array]] = None
) -> Union[int, Array]:
    """
    Sample action from policy.

    Args:
        policy: Policy distribution [A, S]
        key: PRNG key
        state: Optional state (int or one-hot). If None, samples for all states.

    Returns:
        Action (int if state is int, [A] if state is one-hot, [A,S] if state is None)

    Examples:
        # Sample for specific state (indexed)
        >>> action = sample_from(policy, key, state=3)
        >>> type(action)  # int

        # Sample for specific state (one-hot)
        >>> action = sample_from(policy, key, state=jnp.array([0,0,0,1,0]))
        >>> action.shape  # (A,)

        # Sample for all states (original behavior)
        >>> actions = sample_from(policy, key)
        >>> actions.shape  # (A, S)
    """
    if state is None:
        # Original: sample for all states
        return distrax.OneHotCategorical(
            probs=policy.T, dtype="float"
        ).sample(seed=key).T

    elif _is_scalar(state):
        # Indexed: sample single action for given state
        action_probs = policy[:, state]  # [A]
        return jrd.choice(key, len(action_probs), p=action_probs)

    else:
        # One-hot: sample action for given state
        action_probs = jnp.einsum("as,s->a", policy, state)  # [A]
        return distrax.OneHotCategorical(
            probs=action_probs, dtype="float"
        ).sample(seed=key)
```

### async_sample_step_pi

```python
def async_sample_step_pi(
    mdp: MDP,
    policy: PiType,
    state: Union[int, Array],
    episode_step: Scalar,
    episode_length: int,
    key: PRNGKey
) -> tuple[Union[int, Array], Union[int, Array], Scalar, Scalar, Scalar,
           Union[int, Array], Scalar]:
    """
    Sample step following policy - works with indices or one-hot.

    Returns same type as input state.
    """
    act_key, step_key = jrd.split(key, num=2)

    # Sample action (same type as state)
    action = sample_from(policy, key=act_key, state=state)

    # Sample transition
    next_state, reward, terminal, timeout, new_state, new_step = \
        async_sample_step(mdp, action, state, episode_step, episode_length, step_key)

    return action, next_state, reward, terminal, timeout, new_state, new_step
```

---

## Usage in Examples

### Planning Example (Unchanged)

```python
# examples/planning/algorithms.py

from jaxdp import bellman_optimality_operator

class vi:
    def update(state, mdp, step):
        # Works on full Q-table - no change needed
        next_q = bellman_optimality_operator.q(mdp, state.q_vals, state.gamma)
        return state.replace(q_vals=next_q)

# Planning doesn't use sampling primitives much, so no changes needed!
```

### Learning Example (Now Efficient)

```python
# examples/learning/algorithms.py

from jaxdp import async_sample_step_pi
import jax.numpy as jnp

class q_learning:

    def collect_transition_indexed(mdp, policy, state_idx, episode_step,
                                   episode_length, key):
        """Collect transition using indexed representation (efficient)."""
        # Pass INTEGER state → get INTEGER back
        action, next_state, reward, terminal, timeout, new_state, new_step = \
            async_sample_step_pi(
                mdp, policy,
                state_idx,  # Integer!
                episode_step, episode_length, key
            )

        # All states/actions are now integers - efficient!
        return TransitionIndexed(
            state=state_idx,
            action=action,
            reward=reward,
            next_state=next_state,
            terminal=terminal
        )

    def collect_transition_onehot(mdp, policy, state_oh, episode_step,
                                  episode_length, key):
        """Collect transition using one-hot (backward compatible)."""
        # Pass ONE-HOT state → get ONE-HOT back
        action, next_state, reward, terminal, timeout, new_state, new_step = \
            async_sample_step_pi(
                mdp, policy,
                state_oh,  # Array!
                episode_step, episode_length, key
            )

        return Transition(
            state=state_oh,
            action=action,
            reward=reward,
            next_state=next_state,
            terminal=terminal
        )
```

---

## API Summary

### Base Library (`jaxdp/base.py`)

**No change needed:**
- `bellman_optimality_operator.q(mdp, q_vals, gamma)` ✓
- `bellman_operator.q(mdp, policy, q_vals, gamma)` ✓
- `greedy_policy.q(q_vals)` ✓
- `e_greedy_policy.q(q_vals, epsilon)` ✓

**Add type polymorphism:**
- `async_sample_step(mdp, action, state, ...)` → auto-detects type
- `async_sample_step_pi(mdp, policy, state, ...)` → auto-detects type
- `sample_from(policy, key, state)` → returns same type as state

### Examples

**Planning (`examples/planning/`):**
- Uses Bellman operators → no change

**Learning (`examples/learning/`):**
- Can now use indexed sampling → efficient
- Old one-hot code still works → backward compatible

---

## Key Benefits

### 1. Single Simple API
```python
# Same function works for both!
next_s, r, ... = async_sample_step(mdp, action, state, ...)

# User just chooses input type:
action = 1        # int → indexed (fast)
action = [0,1,0]  # array → one-hot (compatible)
```

### 2. No Code Duplication
- One implementation in `base.py` with type dispatch
- Examples choose representation by input type
- No separate `_indexed` vs `_onehot` modules in the library

### 3. Backward Compatible
- Existing code with arrays → works unchanged
- New code with integers → automatically efficient

### 4. Clear Semantics
- Planning: operates on Q-tables (always dense) → use Bellman operators
- Learning: samples transitions → choose indexed (int) or one-hot (array)

---

## Implementation Checklist

### Phase 1: Core Primitives
- [ ] Add `_is_scalar()` helper
- [ ] Implement `_async_sample_step_indexed()`
- [ ] Modify `async_sample_step()` to dispatch based on input type
- [ ] Update `sample_from()` to support state argument (indexed or one-hot)
- [ ] Update `async_sample_step_pi()` to dispatch

### Phase 2: Testing
- [ ] Test indexed sampling returns correct types
- [ ] Test one-hot sampling still works (backward compat)
- [ ] Test can't mix types (error handling)
- [ ] Benchmark memory usage (indexed vs one-hot)

### Phase 3: Examples
- [ ] Update learning examples to show indexed usage
- [ ] Add performance comparison notebook
- [ ] Documentation on when to use which

### Phase 4: Advanced
- [ ] Consider JIT optimization for dispatch
- [ ] Add type hints for Union[int, Array]
- [ ] Profile performance

---

## Example: Complete Learning Loop

```python
# examples/learning/q_learning_efficient.py

from jaxdp import async_sample_step_pi, bellman_optimality_operator
from jaxdp.mdp import GridWorld
import jax.numpy as jnp
import jax.random as jrd

def train_qlearning_large_mdp():
    """Train Q-learning on large MDP using indexed sampling."""

    mdp = GridWorld(height=316, width=316)  # ~100K states
    key = jrd.PRNGKey(0)

    # Q-table is always dense [A, S]
    q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
    gamma = 0.99
    alpha = 0.1

    for episode in range(1000):
        # Sample initial state as INDEX
        key, subkey = jrd.split(key)
        state = jrd.choice(subkey, mdp.state_size, p=mdp.initial)  # int!

        episode_step = 0

        for step in range(100):
            # Get policy from Q-values
            policy = greedy_policy.q(q_vals)  # [A, S]

            # Sample transition - pass INTEGER state
            key, subkey = jrd.split(key)
            action, next_state, reward, terminal, timeout, state, episode_step = \
                async_sample_step_pi(
                    mdp, policy,
                    state,  # INTEGER (not one-hot!)
                    episode_step, 100, subkey
                )

            # Update Q-value - use INDEXED access
            curr_q = q_vals[action, state]
            max_next_q = jnp.max(q_vals[:, next_state])
            target = reward + gamma * max_next_q * (1 - terminal)
            q_vals = q_vals.at[action, state].add(alpha * (target - curr_q))

            if terminal or timeout:
                break

    return q_vals

# Memory usage for 1 transition with S=100K:
# - Old (one-hot): state=[100000], action=[10] = 400KB
# - New (indexed): state=int32, action=int32 = 8 bytes
# 50,000x savings!
```

---

## Summary

**The key insight:**
- **Bellman operators don't need to change** (they work on Q-tables)
- **Sampling primitives need type polymorphism** (accept int OR array)
- **Examples choose representation** (indexed for learning, one-hot optional)

**API is simple:**
- Use integers → indexed (efficient)
- Use arrays → one-hot (compatible)
- Same function, different behavior based on input

**No duplication:**
- Single `async_sample_step()` function
- Single `sample_from()` function
- Type detection handles the rest

This provides the **simplest possible API** while supporting **both planning and learning** efficiently.
