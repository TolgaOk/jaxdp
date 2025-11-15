# API Usage Examples - Simple and Unified

## Goal: One API, Automatic Optimization

---

## Example 1: Default Usage (Recommended)

```python
from jaxdp.learning import q_learning, Transition
from jaxdp.mdp import GridWorld
import jax.random as jrd

# Create MDP
mdp = GridWorld(height=100, width=100)  # Large: 10,000 states
key = jrd.PRNGKey(0)

# Initialize Q-learning (same API regardless of size)
state = q_learning.init(mdp, key, gamma=0.99, alpha=0.1)

# Create transition with INTEGER indices (memory-efficient)
# Library automatically uses indexed implementation
trans = Transition(
    state=3,        # Integer → indexed representation
    action=1,
    reward=1.0,
    next_state=4,
    terminal=False
)

# Update (automatically uses fast indexed path)
state = q_learning.update(state, trans)

# User doesn't need to know about one-hot vs indexed!
```

**Key Point:** Just use integers and it's automatically optimized.

---

## Example 2: Backward Compatible (One-Hot)

```python
from jaxdp.learning import q_learning, Transition
import jax.numpy as jnp

# Existing code with one-hot vectors
state_onehot = jnp.array([0, 0, 0, 1, 0])
action_onehot = jnp.array([0, 1, 0, 0])

# Same API, one-hot inputs
trans = Transition(
    state=state_onehot,      # Array → one-hot representation
    action=action_onehot,
    reward=1.0,
    next_state=jnp.array([0, 0, 0, 0, 1]),
    terminal=False
)

# Update (automatically uses one-hot path)
state = q_learning.update(state, trans)

# No breaking changes! Existing code works!
```

**Key Point:** Backward compatible with existing one-hot code.

---

## Example 3: Batch Updates

```python
from jaxdp.learning import q_learning

# Collect batch of transitions (indexed)
transitions = []
for _ in range(1000):
    trans = Transition(
        state=sample_state_idx(),    # Integers
        action=sample_action_idx(),
        reward=sample_reward(),
        next_state=sample_next_state_idx(),
        terminal=sample_terminal()
    )
    transitions.append(trans)

# Stack into batch
import jax
batch = jax.tree_map(lambda *xs: jnp.stack(xs), *transitions)

# Batch update (automatically optimized for indexed)
state = q_learning.batch_update(state, batch)

# For 1000 transitions with S=100K:
# - One-hot: 800 MB
# - Indexed: 12 KB
# 66,000x memory savings!
```

---

## Example 4: Expert Control (Performance Critical)

If you want **guaranteed** use of indexed path (no dispatch overhead):

```python
# Explicit import from indexed submodule
from jaxdp.learning.indexed import q_learning, Transition

# Now you MUST use indexed format
trans = Transition(state=3, action=1, ...)  # Integers required

# Guaranteed fast path, no dispatch
state = q_learning.update(state, trans)
```

Or keep using one-hot explicitly:

```python
# Explicit import from onehot submodule
from jaxdp.learning.onehot import q_learning, Transition

# Now you MUST use one-hot format
trans = Transition(
    state=jnp.array([0,0,0,1,0]),
    action=jnp.array([0,1,0,0]),
    ...
)

state = q_learning.update(state, trans)
```

---

## Example 5: Sampling Integration

```python
from jaxdp.learning import q_learning
from jaxdp.sampling import sample_transition  # Auto-returns right type

# Sampling automatically returns indexed transitions for large MDPs
for episode in range(num_episodes):
    state_idx = sample_initial_state(mdp, key)

    for step in range(max_steps):
        # sample_transition detects MDP size and returns appropriate type
        action_idx = sample_epsilon_greedy(q_vals, state_idx, epsilon, key)

        # Returns indexed Transition for large MDPs
        trans = sample_transition(mdp, state_idx, action_idx, key)

        # Update just works (auto-dispatch)
        state = q_learning.update(state, trans)

        if trans.terminal:
            break

        state_idx = trans.next_state
```

---

## Example 6: JIT Compilation

```python
from jaxdp.learning import q_learning
import jax

# JIT compile the training step
@jax.jit
def training_step(q_state, transition):
    return q_learning.update(q_state, transition)

# First call with indexed transition
trans_indexed = Transition(state=3, action=1, ...)
state = training_step(state, trans_indexed)
# ^ Compiles indexed path

# Subsequent calls with indexed → uses cached version
state = training_step(state, trans_indexed)  # Fast!

# Call with one-hot → compiles separate version
trans_onehot = Transition(state=jnp.array([0,0,1]), ...)
state = training_step(state, trans_onehot)
# ^ Compiles one-hot path

# Dispatch overhead eliminated by JIT!
```

---

## Decision Guide

```
┌─────────────────────────────────────────────────────────────┐
│              WHICH API SHOULD I USE?                        │
└─────────────────────────────────────────────────────────────┘

Are you writing new code?
│
├─ YES → Use default import with INTEGER indices
│        from jaxdp.learning import q_learning, Transition
│        trans = Transition(state=3, action=1, ...)
│        → Automatically optimized
│
└─ NO (existing code) → Keep current imports
                        Everything still works!
                        from jaxdp.learning import q_learning
                        trans = Transition(state=onehot, ...)
                        → Automatically handles one-hot

Is performance absolutely critical? (> 100K states)
│
└─ YES → Use explicit indexed import
         from jaxdp.learning.indexed import q_learning
         → Guaranteed fast path, no dispatch

Are you writing library code (not application)?
│
└─ YES → Use default import for flexibility
         from jaxdp.learning import q_learning
         → Works with both types
```

---

## Complete Training Example

```python
from jaxdp.learning import q_learning, Transition
from jaxdp.mdp import GridWorld
import jax.random as jrd
import jax.numpy as jnp

def train_large_mdp():
    """Train Q-learning on large MDP with simple API."""

    # Large MDP (100K states)
    mdp = GridWorld(height=316, width=316)  # ~100K states

    # Initialize
    key = jrd.PRNGKey(0)
    state = q_learning.init(mdp, key, gamma=0.99, alpha=0.1)

    # Training loop
    for episode in range(1000):
        # Sample initial state (returns index)
        key, subkey = jrd.split(key)
        current_state = jrd.choice(subkey, mdp.state_size)

        episode_transitions = []

        # Collect episode
        for step in range(100):
            # Epsilon-greedy action
            key, subkey = jrd.split(key)
            if jrd.uniform(subkey) < epsilon:
                action = jrd.choice(subkey, mdp.action_size)
            else:
                action = jnp.argmax(state.q_vals[:, current_state])

            # Sample transition (returns indexed Transition)
            key, subkey = jrd.split(key)
            trans = sample_mdp_transition(mdp, current_state, action, subkey)

            episode_transitions.append(trans)

            if trans.terminal:
                break

            current_state = trans.next_state

        # Batch update (efficient with indexed)
        batch = jax.tree_map(
            lambda *xs: jnp.stack(xs),
            *episode_transitions
        )
        state = q_learning.batch_update(state, batch)

        if episode % 100 == 0:
            print(f"Episode {episode} complete")

    return state.q_vals


# Helper function
def sample_mdp_transition(mdp, state_idx, action_idx, key):
    """Sample transition and return as indexed Transition."""
    # Get transition probabilities
    probs = mdp.transition[action_idx, :, state_idx]

    # Sample next state
    next_state_idx = jrd.choice(key, mdp.state_size, p=probs)

    # Get reward
    reward = mdp.reward[action_idx, state_idx, next_state_idx]

    # Check terminal
    terminal = mdp.terminal[next_state_idx]

    # Return indexed Transition
    return Transition(
        state=state_idx,
        action=action_idx,
        reward=reward,
        next_state=next_state_idx,
        terminal=terminal
    )


if __name__ == "__main__":
    q_vals = train_large_mdp()
    print("Training complete!")
```

---

## Key Takeaways

### For Users
1. **Just use integers** → Automatically optimized
2. **Existing code works** → No breaking changes
3. **Same API everywhere** → Easy to learn
4. **Performance is automatic** → No manual tuning needed

### For Maintainers
1. Auto-dispatch handles 90% of cases
2. Expert users can opt into explicit paths
3. Two implementations, one interface
4. JIT eliminates dispatch overhead

### For Reviewers
1. Simple user-facing API ✓
2. Backward compatible ✓
3. Performance optimized ✓
4. Clear migration path ✓

---

## Summary

**You asked for the simplest API while being functional.**

**Answer:**
- **One import:** `from jaxdp.learning import q_learning`
- **One API:** `q_learning.update(state, transition)`
- **Automatic optimization:** Integer inputs → fast path
- **Backward compatible:** Array inputs → original path
- **Expert control available:** Explicit imports for guaranteed behavior

**User barely notices the difference. Library handles the complexity.**
