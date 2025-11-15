# Complete Function Analysis: Index Support Requirements

## All Functions in `jaxdp/base.py`

This document analyzes EVERY function to determine what needs index support.

---

## Category 1: Works on Q-Tables (NO CHANGE NEEDED) ✓

These functions operate on Q-value tables `[A, S]` which are always dense. Both planning and learning use the same Q-tables.

### 1. `bellman_optimality_operator.q(mdp, q_vals, gamma)`
```python
# Current signature
def q(mdp: MDP, value: QType, gamma: float) -> QType
```
**Analysis:**
- Input: Q-values `[A, S]`
- Output: Q-values `[A, S]`
- Operations: Einsum over full transition tensor
- **Verdict: NO CHANGE** ✓

### 2. `bellman_operator.q(mdp, policy, q_vals, gamma)`
```python
# Current signature
def q(mdp: MDP, policy: PiType, value: QType, gamma: float) -> QType
```
**Analysis:**
- Input: Policy `[A, S]`, Q-values `[A, S]`
- Output: Q-values `[A, S]`
- Operations: Einsum over full state space
- **Verdict: NO CHANGE** ✓

### 3. `bellman_operator.v(mdp, policy, v_vals, gamma)`
```python
def v(mdp: MDP, policy: PiType, value: VType, gamma: float) -> VType
```
**Analysis:**
- State values `[S]` (dense)
- **Verdict: NO CHANGE** ✓

### 4. `policy_evaluation.q(mdp, policy, gamma)`
```python
def q(mdp: MDP, policy: PiType, gamma: float) -> QType
```
**Analysis:**
- Matrix inversion on full state space
- **Verdict: NO CHANGE** ✓

### 5. `policy_evaluation.v(mdp, policy, gamma)`
```python
def v(mdp: MDP, policy: PiType, gamma: float) -> VType
```
**Analysis:**
- Matrix inversion on full state space
- **Verdict: NO CHANGE** ✓

### 6. `to_greedy_state_value(q_vals)`
```python
def to_greedy_state_value(value: QType) -> VType
```
**Analysis:**
- Input: Q-values `[A, S]`
- Output: State values `[S]`
- Operation: `jnp.max(value, axis=0)`
- **Verdict: NO CHANGE** ✓

### 7. `to_state_action_value(mdp, v_vals, gamma)`
```python
def to_state_action_value(mdp: MDP, value: VType, gamma: float) -> QType
```
**Analysis:**
- Converts state values to Q-values (full table)
- **Verdict: NO CHANGE** ✓

### 8. `markov_chain_eigen_values(mdp, policy)`
```python
def markov_chain_eigen_values(mdp: MDP, policy: PiType) -> F["S"]
```
**Analysis:**
- Computes eigenvalues of transition matrix
- **Verdict: NO CHANGE** ✓

### 9. `_markov_chain_pi(mdp, policy)`
```python
def _markov_chain_pi(mdp: MDP, policy: PiType) -> tuple[F["SS"], F["SS"]]
```
**Analysis:**
- Creates Markov chain from policy (full matrices)
- **Verdict: NO CHANGE** ✓

### 10. `stationary_distribution.q(mdp, policy, iterations)`
```python
def q(mdp: MDP, policy: PiType, iterations: int) -> F["AS"]
```
**Analysis:**
- Computes stationary distribution (full state space)
- **Verdict: NO CHANGE** ✓

### 11. `sg(array)`
```python
def sg(array: chex.Array) -> chex.Array
```
**Analysis:**
- Stop gradient utility
- **Verdict: NO CHANGE** ✓

---

## Category 2: Policy Extraction (NEEDS INDEX SUPPORT) ⚠️

These extract policies from Q-values. Currently return full policy matrix, but could benefit from indexed versions.

### 12. `greedy_policy.q(q_vals)` ⚠️

**Current:**
```python
def q(value: QType) -> PiType:
    # Returns [A, S] one-hot policy
    return jax.nn.one_hot(jnp.argmax(value, axis=0), num_classes=value.shape[0], axis=0)
```

**Issue:**
- For **single state**, we waste memory returning full `[A, S]` matrix
- Learning only needs policy for ONE state at a time

**Proposed Addition:**
```python
def q(value: QType, state: Optional[int] = None) -> Union[PiType, F["A"]]:
    """
    Extract greedy policy from Q-values.

    Args:
        value: Q-values [A, S]
        state: Optional specific state index

    Returns:
        If state is None: Full policy [A, S]
        If state is int: Policy for that state [A] (one-hot)
    """
    if state is None:
        # Original behavior: full policy
        return jax.nn.one_hot(jnp.argmax(value, axis=0),
                             num_classes=value.shape[0], axis=0)
    else:
        # New: policy for single state
        greedy_action = jnp.argmax(value[:, state])
        return jax.nn.one_hot(greedy_action, num_classes=value.shape[0])
```

**OR even simpler:**
```python
def q_action(value: QType, state: int) -> int:
    """
    Get greedy action for a specific state (indexed).

    Args:
        value: Q-values [A, S]
        state: State index

    Returns:
        Greedy action index
    """
    return jnp.argmax(value[:, state])
```

**Verdict: ADD INDEXED VERSION** ⚠️

### 13. `e_greedy_policy.q(q_vals, epsilon)` ⚠️

**Current:**
```python
def q(value: QType, epsilon: float) -> PiType:
    greedy_p = greedy_policy.q(value)
    return greedy_p * (1 - epsilon) + jnp.ones_like(value) * (epsilon / value.shape[0])
```

**Proposed Addition:**
```python
def q_action(value: QType, state: int, epsilon: float, key: PRNGKey) -> int:
    """
    Sample epsilon-greedy action for specific state.

    Args:
        value: Q-values [A, S]
        state: State index
        epsilon: Exploration parameter
        key: PRNG key

    Returns:
        Sampled action index
    """
    n_actions = value.shape[0]

    # With probability epsilon, random action
    explore = jrd.uniform(key) < epsilon

    if explore:
        return jrd.randint(key, (), 0, n_actions)
    else:
        return jnp.argmax(value[:, state])
```

**Verdict: ADD INDEXED VERSION** ⚠️

### 14. `soft_policy.q(q_vals, temperature)` ⚠️

**Current:**
```python
def q(value: QType, temperature: float) -> PiType:
    return jax.nn.softmax(value / temperature, axis=0)
```

**Proposed Addition:**
```python
def q_action(value: QType, state: int, temperature: float, key: PRNGKey) -> int:
    """
    Sample softmax action for specific state.

    Args:
        value: Q-values [A, S]
        state: State index
        temperature: Temperature parameter
        key: PRNG key

    Returns:
        Sampled action index
    """
    logits = value[:, state] / temperature
    return jrd.categorical(key, logits=logits)
```

**Verdict: ADD INDEXED VERSION** ⚠️

---

## Category 3: Sampling (NEEDS INDEX SUPPORT) ❌

These are the critical ones for learning efficiency.

### 15. `sample_from(policy, key)` ❌

**Current:**
```python
def sample_from(policy: PiType, key: PRNGKey) -> F["AS"]:
    """Sample actions for ALL states (returns [A, S])."""
    return distrax.OneHotCategorical(probs=policy.T, dtype="float").sample(seed=key).T
```

**Issue:**
- Always samples for ALL states
- Learning only needs ONE state

**Proposed:**
```python
def sample_from(
    policy: PiType,
    key: PRNGKey,
    state: Optional[Union[int, Array]] = None
) -> Union[F["AS"], F["A"], int]:
    """
    Sample actions from policy.

    Args:
        policy: Policy [A, S]
        key: PRNG key
        state: Optional state (int=indexed, array=one-hot, None=all states)

    Returns:
        If state is None: Actions for all states [A, S] (one-hot)
        If state is int: Action index (int)
        If state is array: Action one-hot [A]
    """
    if state is None:
        # Original: sample for all states
        return distrax.OneHotCategorical(probs=policy.T, dtype="float")\
                     .sample(seed=key).T

    elif jnp.ndim(state) == 0:  # int/scalar
        # Indexed: sample single action
        action_probs = policy[:, state]
        return jrd.categorical(key, logits=jnp.log(action_probs))

    else:  # array
        # One-hot: sample single action
        action_probs = jnp.einsum("as,s->a", policy, state)
        return distrax.OneHotCategorical(probs=action_probs, dtype="float")\
                     .sample(seed=key)
```

**Verdict: NEEDS INDEX SUPPORT** ❌

### 16. `async_sample_step(mdp, action, state, ...)` ❌

**Already discussed - NEEDS INDEX SUPPORT**

### 17. `async_sample_step_pi(mdp, policy, state, ...)` ❌

**Already discussed - NEEDS INDEX SUPPORT**

### 18. `sync_sample(mdp, key)` ⚠️

**Current:**
```python
def sync_sample(mdp: MDP, key: PRNGKey) -> tuple[F["AS"], F["ASS"], F["AS"]]:
    """
    Synchronously sample from ALL state-action pairs.

    Returns:
        rewards [A, S]
        next_states [A, S, S]
        terminals [A, S]
    """
```

**Analysis:**
- This is for **planning** - samples from ALL (s,a) pairs simultaneously
- Not typically used in learning
- **Verdict: PROBABLY NO CHANGE** (planning-focused)

But could add indexed version:
```python
def sync_sample_indexed(mdp: MDP, state: int, action: int, key: PRNGKey):
    """Sample from specific (state, action) pair."""
    next_state_probs = mdp.transition[action, :, state]
    next_state = jrd.categorical(key, logits=jnp.log(next_state_probs))
    reward = mdp.reward[action, state, next_state]
    terminal = mdp.terminal[next_state]
    return reward, next_state, terminal
```

**Verdict: OPTIONAL ADDITION** ⚠️

---

## Category 4: Expected Value (COULD BENEFIT FROM INDEX SUPPORT) ⚠️

### 19. `expected_value.q(mdp, q_vals)`
```python
def q(mdp: MDP, value: QType) -> chex.Scalar:
    """Expected value over initial distribution."""
    return expected_value.v(mdp, jnp.max(value, axis=0))
```

### 20. `expected_value.v(mdp, v_vals)`
```python
def v(mdp: MDP, value: VType) -> chex.Scalar:
    """Expected value over initial distribution."""
    return (value * mdp.initial).sum()
```

**Analysis:**
- These compute expected values over initial distribution
- Used for evaluation metrics
- **Verdict: NO CHANGE NEEDED** (metrics, not hot path)

---

## Category 5: Sample-Based Evaluation (NEEDS INDEX SUPPORT) ⚠️

### 21. `sample_based_policy_evaluation(mdp, policy, key, gamma, max_episode_length)`

**Current:**
```python
def sample_based_policy_evaluation(...) -> chex.Scalar:
    """
    Evaluate policy using Monte Carlo sampling.

    Currently uses one-hot states internally.
    """
```

**Issue:**
- Uses `async_sample_step_pi` internally
- Once we add index support to `async_sample_step_pi`, this gets it for free
- **Verdict: GETS INDEX SUPPORT AUTOMATICALLY** ✓

---

## Summary Table

| Function | Category | Needs Index? | Priority |
|----------|----------|--------------|----------|
| bellman_optimality_operator.q | Q-table ops | ✓ No | - |
| bellman_operator.q | Q-table ops | ✓ No | - |
| bellman_operator.v | Q-table ops | ✓ No | - |
| policy_evaluation.q | Q-table ops | ✓ No | - |
| policy_evaluation.v | Q-table ops | ✓ No | - |
| to_greedy_state_value | Q-table ops | ✓ No | - |
| to_state_action_value | Q-table ops | ✓ No | - |
| markov_chain_eigen_values | Analysis | ✓ No | - |
| _markov_chain_pi | Analysis | ✓ No | - |
| stationary_distribution.q | Analysis | ✓ No | - |
| sg | Utility | ✓ No | - |
| **greedy_policy.q** | **Policy** | **⚠️ Yes** | **HIGH** |
| **e_greedy_policy.q** | **Policy** | **⚠️ Yes** | **HIGH** |
| **soft_policy.q** | **Policy** | **⚠️ Yes** | **MEDIUM** |
| **sample_from** | **Sampling** | **❌ YES** | **CRITICAL** |
| **async_sample_step** | **Sampling** | **❌ YES** | **CRITICAL** |
| **async_sample_step_pi** | **Sampling** | **❌ YES** | **CRITICAL** |
| sync_sample | Sampling | ⚠️ Optional | LOW |
| expected_value.q | Metrics | ✓ No | - |
| expected_value.v | Metrics | ✓ No | - |
| sample_based_policy_evaluation | Eval | ✓ Auto | - |

---

## Revised Implementation Plan

### Phase 1: Critical Sampling Functions (Week 1)

**Modify these 3 functions to accept int OR array:**

1. **`async_sample_step()`**
   - Add type detection
   - Add `_async_sample_step_indexed()`
   - Add `_async_sample_step_onehot()`

2. **`async_sample_step_pi()`**
   - Uses updated `async_sample_step()`
   - Automatically gets index support

3. **`sample_from()`**
   - Add optional `state` parameter
   - Support int/array/None

### Phase 2: Policy Functions (Week 2)

**Add indexed helper methods to policy classes:**

4. **`greedy_policy.q_action(q_vals, state)`**
   - New method: returns action index for single state
   - Complements existing `.q()` method

5. **`e_greedy_policy.q_action(q_vals, state, epsilon, key)`**
   - New method: samples epsilon-greedy action for single state
   - Complements existing `.q()` method

6. **`soft_policy.q_action(q_vals, state, temperature, key)`**
   - New method: samples softmax action for single state
   - Complements existing `.q()` method

### Phase 3: Optional Additions (Week 3)

7. **`sync_sample_indexed(mdp, state, action, key)`**
   - New function: sample from specific (s,a) pair
   - Optional but useful

---

## Detailed Implementation: Policy Functions

### greedy_policy - BEFORE
```python
class greedy_policy(metaclass=StaticMeta):

    def q(value: QType) -> PiType:
        """Greedy policy for ALL states."""
        return jax.nn.one_hot(jnp.argmax(value, axis=0),
                             num_classes=value.shape[0], axis=0)
```

### greedy_policy - AFTER
```python
class greedy_policy(metaclass=StaticMeta):

    def q(value: QType) -> PiType:
        """Greedy policy for ALL states (unchanged)."""
        return jax.nn.one_hot(jnp.argmax(value, axis=0),
                             num_classes=value.shape[0], axis=0)

    def q_action(value: QType, state: int) -> int:
        """
        Get greedy action for specific state (NEW).

        Args:
            value: Q-values [A, S]
            state: State index

        Returns:
            Greedy action index

        Example:
            >>> action = greedy_policy.q_action(q_vals, state=3)
            >>> type(action)
            int
        """
        return jnp.argmax(value[:, state])

    def q_onehot(value: QType, state: int) -> F["A"]:
        """
        Get greedy action as one-hot for specific state (NEW).

        Args:
            value: Q-values [A, S]
            state: State index

        Returns:
            Greedy action as one-hot [A]
        """
        action_idx = jnp.argmax(value[:, state])
        return jax.nn.one_hot(action_idx, num_classes=value.shape[0])
```

### e_greedy_policy - AFTER
```python
class e_greedy_policy(metaclass=StaticMeta):

    def q(value: QType, epsilon: float) -> PiType:
        """Epsilon-greedy policy for ALL states (unchanged)."""
        greedy_p = greedy_policy.q(value)
        return greedy_p * (1 - epsilon) + jnp.ones_like(value) * (epsilon / value.shape[0])

    def q_action(value: QType, state: int, epsilon: float, key: PRNGKey) -> int:
        """
        Sample epsilon-greedy action for specific state (NEW).

        Args:
            value: Q-values [A, S]
            state: State index
            epsilon: Exploration parameter
            key: PRNG key

        Returns:
            Sampled action index

        Example:
            >>> action = e_greedy_policy.q_action(q_vals, state=3, epsilon=0.1, key=key)
            >>> type(action)
            int
        """
        n_actions = value.shape[0]
        key_explore, key_action = jrd.split(key)

        # With probability epsilon: random action
        explore = jrd.uniform(key_explore) < epsilon

        # Random action
        random_action = jrd.randint(key_action, (), 0, n_actions)

        # Greedy action
        greedy_action = jnp.argmax(value[:, state])

        # Choose
        return jax.lax.select(explore, random_action, greedy_action)

    def q_probs(value: QType, state: int, epsilon: float) -> F["A"]:
        """
        Get epsilon-greedy probabilities for specific state (NEW).

        Args:
            value: Q-values [A, S]
            state: State index
            epsilon: Exploration parameter

        Returns:
            Action probabilities [A]
        """
        n_actions = value.shape[0]
        greedy_action = jnp.argmax(value[:, state])
        greedy_onehot = jax.nn.one_hot(greedy_action, n_actions)
        return greedy_onehot * (1 - epsilon) + jnp.ones(n_actions) * (epsilon / n_actions)
```

### soft_policy - AFTER
```python
class soft_policy(metaclass=StaticMeta):

    def q(value: QType, temperature: float) -> PiType:
        """Softmax policy for ALL states (unchanged)."""
        return jax.nn.softmax(value / temperature, axis=0)

    def q_action(value: QType, state: int, temperature: float, key: PRNGKey) -> int:
        """
        Sample softmax action for specific state (NEW).

        Args:
            value: Q-values [A, S]
            state: State index
            temperature: Temperature parameter
            key: PRNG key

        Returns:
            Sampled action index
        """
        logits = value[:, state] / temperature
        return jrd.categorical(key, logits=logits)

    def q_probs(value: QType, state: int, temperature: float) -> F["A"]:
        """
        Get softmax probabilities for specific state (NEW).

        Args:
            value: Q-values [A, S]
            state: State index
            temperature: Temperature parameter

        Returns:
            Action probabilities [A]
        """
        return jax.nn.softmax(value[:, state] / temperature)
```

---

## Usage Examples with New API

### Learning with Indexed Actions

```python
from jaxdp import async_sample_step, greedy_policy, e_greedy_policy
from jaxdp.mdp import GridWorld
import jax.numpy as jnp
import jax.random as jrd

mdp = GridWorld(316, 316)  # 100K states
q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
key = jrd.PRNGKey(0)

# Start at state 0
state = 0

for episode in range(1000):
    # Get greedy action for this state (NEW!)
    action = greedy_policy.q_action(q_vals, state)
    # Returns int, not [A] array!

    # OR epsilon-greedy (NEW!)
    key, subkey = jrd.split(key)
    action = e_greedy_policy.q_action(q_vals, state, epsilon=0.1, key=subkey)

    # Sample transition (indexed)
    key, subkey = jrd.split(key)
    next_state, reward, terminal, timeout, state, step = \
        async_sample_step(mdp, action, state, 0, 100, subkey)

    # Update Q-value
    curr_q = q_vals[action, state]
    max_next_q = jnp.max(q_vals[:, next_state])
    target = reward + 0.99 * max_next_q
    q_vals = q_vals.at[action, state].add(0.1 * (target - curr_q))

# All operations use integers - no one-hot waste!
```

---

## Summary of Changes

### Functions to MODIFY:
1. `async_sample_step()` - Add type dispatch
2. `async_sample_step_pi()` - Use updated async_sample_step
3. `sample_from()` - Add state parameter

### Functions to ADD METHODS TO:
4. `greedy_policy` - Add `.q_action()` and `.q_onehot()`
5. `e_greedy_policy` - Add `.q_action()` and `.q_probs()`
6. `soft_policy` - Add `.q_action()` and `.q_probs()`

### Total Changes:
- **3 modified functions**
- **6 new methods** (2 per policy class)
- **~200-300 lines of code**
- **Zero breaking changes** (all additions)

This is a **complete plan** covering all base.py functions!
