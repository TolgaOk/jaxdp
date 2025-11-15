# Unified API Design for Hybrid Approach

## The Challenge

**Problem:** We want two implementations (one-hot and indexed) but ONE simple API.

**Anti-pattern to avoid:**
```python
# BAD: User has to choose and remember two different APIs
from jaxdp.learning import q_learning          # One-hot version
from jaxdp.learning import q_learning_indexed  # Indexed version

# Which one do I use? How do I switch? Confusing!
```

**Goal:** User should barely notice the difference, if at all.

---

## Design Option 1: Automatic Type Dispatch ⭐ **RECOMMENDED**

### Core Idea
Single API that automatically uses the right implementation based on input type.

### Implementation

```python
# jaxdp/learning/algorithms.py

from typing import Union
import jax.numpy as jnp
from flax import struct

# Unified transition type that supports both
@struct.dataclass
class Transition:
    """
    Universal transition representation.

    Automatically uses efficient storage based on input:
    - Integer inputs: Stored as indices (memory-efficient)
    - Array inputs: Stored as one-hot (backward compatible)
    """
    state: Union[int, jnp.ndarray]
    action: Union[int, jnp.ndarray]
    reward: float
    next_state: Union[int, jnp.ndarray]
    terminal: bool

    @property
    def is_indexed(self) -> bool:
        """Check if using indexed representation."""
        return jnp.isscalar(self.state) or (
            isinstance(self.state, jnp.ndarray) and self.state.ndim == 0
        )


class q_learning:
    """
    Q-Learning with automatic dispatch.

    Works with both indexed and one-hot representations transparently.
    """

    @staticmethod
    def update(state: State, transition: Transition) -> State:
        """
        Update Q-values from transition.

        Automatically uses efficient implementation based on input type.

        Args:
            state: Algorithm state
            transition: Can be indexed (int) or one-hot (array)

        Returns:
            Updated state
        """
        if transition.is_indexed:
            return q_learning._update_indexed(state, transition)
        else:
            return q_learning._update_onehot(state, transition)

    @staticmethod
    def _update_indexed(state: State, transition: Transition) -> State:
        """Fast path for indexed transitions."""
        curr_q = state.q_vals[transition.action, transition.state]
        max_next_q = jnp.max(state.q_vals[:, transition.next_state])
        target = transition.reward + state.gamma * max_next_q * (1 - transition.terminal)
        td_error = target - curr_q

        new_q_vals = state.q_vals.at[transition.action, transition.state]\
                                  .add(state.alpha * td_error)
        return state.replace(q_vals=new_q_vals)

    @staticmethod
    def _update_onehot(state: State, transition: Transition) -> State:
        """Backward-compatible path for one-hot transitions."""
        curr_q = jnp.einsum("as,a,s->", state.q_vals,
                           transition.action, transition.state)
        q_next = jnp.einsum("as,s->a", state.q_vals, transition.next_state)
        max_next_q = jnp.max(q_next)
        target = transition.reward + state.gamma * max_next_q * (1 - transition.terminal)
        td_error = target - curr_q

        update = jnp.einsum("a,s->as", transition.action, transition.state)
        new_q_vals = state.q_vals + state.alpha * td_error * update
        return state.replace(q_vals=new_q_vals)
```

### User Experience

```python
from jaxdp.learning import q_learning, Transition

# Initialize once
state = q_learning.init(mdp, key, gamma=0.99, alpha=0.1)

# Use with indexed transitions (memory-efficient for large MDPs)
trans = Transition(
    state=3,           # Integer index
    action=1,          # Integer index
    reward=1.0,
    next_state=4,
    terminal=False
)
state = q_learning.update(state, trans)  # Automatically uses fast path

# OR use with one-hot (backward compatible)
trans = Transition(
    state=jnp.array([0,0,0,1,0]),      # One-hot
    action=jnp.array([0,1,0,0]),       # One-hot
    reward=1.0,
    next_state=jnp.array([0,0,0,0,1]),
    terminal=False
)
state = q_learning.update(state, trans)  # Automatically uses einsum path

# SAME API, different internals!
```

### Pros
- ✅ **Single import, single API**
- ✅ **Automatic optimization**
- ✅ **Backward compatible**
- ✅ **Type-safe with Union types**
- ✅ **Clear semantics**

### Cons
- ⚠️ Runtime dispatch overhead (mitigated by JIT)
- ⚠️ Slightly more complex implementation
- ⚠️ Type checkers might complain about Union types

---

## Design Option 2: Configuration-Based Selection

### Core Idea
User configures representation once, API stays the same.

### Implementation

```python
# jaxdp/learning/algorithms.py

@struct.dataclass
class QLearningConfig:
    """Configuration for Q-learning algorithm."""
    gamma: float = 0.99
    alpha: float = 0.1
    use_indexed: bool = False  # User chooses representation

class q_learning:

    @staticmethod
    def init(mdp, key, config: QLearningConfig):
        """Initialize with configuration."""
        state = State(
            q_vals=jnp.zeros((mdp.action_size, mdp.state_size)),
            gamma=config.gamma,
            alpha=config.alpha,
            _use_indexed=config.use_indexed,  # Store preference
        )
        return state

    @staticmethod
    def update(state: State, transition: Transition):
        """Update uses the configured representation."""
        if state._use_indexed:
            # Automatically convert if needed
            if not transition.is_indexed:
                transition = _to_indexed(transition)
            return q_learning._update_indexed(state, transition)
        else:
            if transition.is_indexed:
                transition = _to_onehot(transition, state.q_vals.shape)
            return q_learning._update_onehot(state, transition)
```

### User Experience

```python
from jaxdp.learning import q_learning, QLearningConfig

# Configure once based on problem size
if mdp.state_size > 10000:
    config = QLearningConfig(gamma=0.99, alpha=0.1, use_indexed=True)
else:
    config = QLearningConfig(gamma=0.99, alpha=0.1, use_indexed=False)

state = q_learning.init(mdp, key, config)

# Use same API regardless
trans = sample_transition(mdp, policy, key)  # Returns appropriate type
state = q_learning.update(state, trans)      # Just works!
```

### Pros
- ✅ **Very simple user-facing API**
- ✅ **Explicit configuration**
- ✅ **No runtime dispatch (configured once)**
- ✅ **Easy to understand**

### Cons
- ⚠️ Must configure correctly upfront
- ⚠️ Conversion overhead if mixing types
- ⚠️ Less flexible than auto-dispatch

---

## Design Option 3: Smart Constructors

### Core Idea
Same class, different constructors for different representations.

### Implementation

```python
class Transition:
    """Universal transition type."""

    @staticmethod
    def from_indices(state: int, action: int, reward: float,
                     next_state: int, terminal: bool):
        """Create indexed transition (memory-efficient)."""
        return Transition(
            state=jnp.int32(state),
            action=jnp.int32(action),
            reward=reward,
            next_state=jnp.int32(next_state),
            terminal=terminal,
            _is_indexed=True,
        )

    @staticmethod
    def from_onehot(state: Array, action: Array, reward: float,
                    next_state: Array, terminal: bool):
        """Create one-hot transition (backward compatible)."""
        return Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminal=terminal,
            _is_indexed=False,
        )

    @staticmethod
    def auto(state, action, reward, next_state, terminal):
        """Automatically detect type."""
        is_indexed = jnp.isscalar(state)
        if is_indexed:
            return Transition.from_indices(state, action, reward, next_state, terminal)
        else:
            return Transition.from_onehot(state, action, reward, next_state, terminal)
```

### User Experience

```python
# Explicit (recommended for large MDPs)
trans = Transition.from_indices(state=3, action=1, reward=1.0,
                                next_state=4, terminal=False)

# Explicit (backward compatible)
trans = Transition.from_onehot(state_onehot, action_onehot, 1.0,
                               next_state_onehot, False)

# Auto-detect
trans = Transition.auto(3, 1, 1.0, 4, False)  # Detects integers
```

---

## Design Option 4: Namespace + Unified Interface ⭐ **ALTERNATIVE**

### Core Idea
Separate implementations but identical interface (duck typing).

### Implementation

```python
# jaxdp/learning/onehot.py
class q_learning:
    @staticmethod
    def init(mdp, key, gamma, alpha): ...

    @staticmethod
    def update(state, transition): ...

# jaxdp/learning/indexed.py
class q_learning:
    @staticmethod
    def init(n_actions, n_states, gamma, alpha): ...

    @staticmethod
    def update(state, transition): ...

# jaxdp/learning/__init__.py
"""
Choose representation based on problem size:

Small MDPs (< 10K states):
    from jaxdp.learning.onehot import q_learning

Large MDPs (> 10K states):
    from jaxdp.learning.indexed import q_learning
"""
```

### User Experience

```python
# User chooses ONCE at import
if mdp.state_size > 10000:
    from jaxdp.learning.indexed import q_learning
else:
    from jaxdp.learning.onehot import q_learning

# Rest of code is IDENTICAL
state = q_learning.init(mdp, key, gamma=0.99, alpha=0.1)
state = q_learning.update(state, transition)
state = q_learning.batch_update(state, transitions)
```

### Pros
- ✅ **Cleanest separation**
- ✅ **Zero runtime overhead**
- ✅ **Same API surface**
- ✅ **Easy to maintain**
- ✅ **No Union types or dispatch**

### Cons
- ⚠️ User must know to choose
- ⚠️ Can't mix easily
- ⚠️ Two codebases to maintain (but that's already true)

---

## Comparison Table

| Design | Simplicity | Performance | Flexibility | Maintenance |
|--------|-----------|-------------|-------------|-------------|
| **Auto Dispatch** | ★★★★★ | ★★★★ (dispatch overhead) | ★★★★★ | ★★★ (complex) |
| **Config-Based** | ★★★★★ | ★★★★★ (configured) | ★★★★ | ★★★★ |
| **Smart Constructors** | ★★★★ | ★★★★★ | ★★★★★ | ★★★ |
| **Namespace** | ★★★★ | ★★★★★ (no overhead) | ★★★ | ★★★★★ (clean) |

---

## My Recommendation: **Hybrid of Option 1 + 4**

### The Best of Both Worlds

```python
# jaxdp/learning/__init__.py

from jaxdp.learning.core import q_learning  # Auto-dispatch version (Option 1)
from jaxdp.learning import indexed, onehot   # Explicit versions (Option 4)

"""
Three ways to use Q-learning:

1. Auto (recommended for most users):
   from jaxdp.learning import q_learning
   state = q_learning.update(state, transition)  # Auto-detects

2. Explicit indexed (for large MDPs, performance-critical):
   from jaxdp.learning.indexed import q_learning
   state = q_learning.update(state, transition)

3. Explicit one-hot (for backward compatibility, small MDPs):
   from jaxdp.learning.onehot import q_learning
   state = q_learning.update(state, transition)
"""
```

### Why This?

1. **Default is simple:** `from jaxdp.learning import q_learning` → auto-dispatch
2. **Experts can optimize:** Import from `indexed` or `onehot` submodules
3. **No breaking changes:** Existing code keeps working
4. **Clear migration path:** Start with auto, optimize later if needed

### Implementation Strategy

```python
# jaxdp/learning/onehot/algorithms.py
class q_learning:
    """One-hot implementation (original)."""
    # ... existing code ...

# jaxdp/learning/indexed/algorithms.py
class q_learning:
    """Indexed implementation (new)."""
    # ... optimized code ...

# jaxdp/learning/core.py
class q_learning:
    """Auto-dispatching wrapper."""

    @staticmethod
    def update(state, transition):
        # Detect type
        if _is_indexed(transition):
            from jaxdp.learning.indexed import q_learning as q_idx
            return q_idx.update(state, transition)
        else:
            from jaxdp.learning.onehot import q_learning as q_oh
            return q_oh.update(state, transition)

# jaxdp/learning/__init__.py
from jaxdp.learning.core import q_learning  # Auto version is default
from jaxdp.learning import indexed, onehot  # Explicit versions available
```

---

## Addressing Your Concern

> "It is not merely the problem of implementing, it is to provide the simplest API while being functional"

**Solution:** The hybrid approach (Option 1 + 4) provides:

1. **Simplest possible for most users:**
   ```python
   from jaxdp.learning import q_learning
   # Just works with any transition type
   ```

2. **Explicit control for experts:**
   ```python
   from jaxdp.learning.indexed import q_learning
   # Guaranteed to use fast path
   ```

3. **Backward compatibility:**
   ```python
   # Existing code continues working
   from jaxdp.learning import q_learning
   # Automatically handles one-hot
   ```

4. **Clear semantics:**
   - Default import → auto-dispatch (simple but small overhead)
   - Submodule import → explicit choice (fast but requires knowledge)
   - Documentation guides when to use which

---

## JIT Compilation Considerations

**Important:** The dispatch overhead can be eliminated by JIT:

```python
# Auto-dispatch version can be JIT-compiled
@jax.jit
def training_step(state, transition):
    return q_learning.update(state, transition)

# JIT will specialize based on input type at compile time
# First call with indexed → compiles indexed version
# Subsequent calls with indexed → uses cached version
# Call with one-hot → compiles separate version
```

So the "dispatch overhead" is **only on first JIT compilation**, not runtime!

---

## Final API Design Proposal

```python
# jaxdp/learning/__init__.py

"""
Q-Learning API with automatic optimization.

Simple usage (recommended):
    from jaxdp.learning import q_learning, Transition

    trans = Transition(state=3, action=1, ...)  # Integer = indexed
    state = q_learning.update(state, trans)     # Auto-optimized

Performance-critical usage:
    from jaxdp.learning.indexed import q_learning, Transition

    trans = Transition(state=3, action=1, ...)
    state = q_learning.update(state, trans)  # Guaranteed indexed path

Backward compatible:
    from jaxdp.learning.onehot import q_learning, Transition

    trans = Transition(state=onehot_s, action=onehot_a, ...)
    state = q_learning.update(state, trans)  # Original implementation
"""

# Default: Auto-dispatch
from jaxdp.learning.core import q_learning, Transition

# Explicit: Available as submodules
from jaxdp.learning import indexed, onehot

__all__ = ['q_learning', 'Transition', 'indexed', 'onehot']
```

This provides the **simplest possible API** while being **fully functional** and **performant**.
