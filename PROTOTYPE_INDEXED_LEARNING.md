# Indexed Learning Implementation Prototype

This document contains working prototypes for the indexed learning approach (Option C).

---

## Table of Contents

1. [Data Structures](#1-data-structures)
2. [Core Operations](#2-core-operations)
3. [Q-Learning Implementation](#3-q-learning-implementation)
4. [Sampling Functions](#4-sampling-functions)
5. [Batch Processing](#5-batch-processing)
6. [Conversion Utilities](#6-conversion-utilities)
7. [Benchmarking Code](#7-benchmarking-code)

---

## 1. Data Structures

### 1.1 TransitionIndexed

```python
# jaxdp/learning/types_indexed.py

from flax import struct
import jax.numpy as jnp
from jaxdp.typehints import F

@struct.dataclass
class TransitionIndexed:
    """
    Indexed representation of MDP transitions.

    Memory: 13 bytes per transition (vs 800KB+ for one-hot with S=100K)
    """
    state: jnp.int32       # Current state index
    action: jnp.int32      # Action index
    reward: jnp.float32    # Immediate reward
    next_state: jnp.int32  # Next state index
    terminal: jnp.bool_    # Terminal flag

    @property
    def shape(self):
        """Return shape for batched transitions."""
        return self.state.shape

    @staticmethod
    def from_onehot(state_oh, action_oh, reward, next_state_oh, terminal):
        """Convert one-hot transition to indexed."""
        return TransitionIndexed(
            state=jnp.argmax(state_oh).astype(jnp.int32),
            action=jnp.argmax(action_oh).astype(jnp.int32),
            reward=reward.astype(jnp.float32),
            next_state=jnp.argmax(next_state_oh).astype(jnp.int32),
            terminal=terminal.astype(jnp.bool_),
        )

    def to_onehot(self, n_states: int, n_actions: int):
        """Convert indexed transition to one-hot (for compatibility)."""
        from jaxdp.learning.algorithms import Transition
        return Transition(
            state=jax.nn.one_hot(self.state, n_states),
            action=jax.nn.one_hot(self.action, n_actions),
            reward=self.reward,
            next_state=jax.nn.one_hot(self.next_state, n_states),
            terminal=self.terminal,
        )
```

### 1.2 Algorithm State

```python
# Compatible with existing q_learning.State
@struct.dataclass
class QLearningState:
    """State for Q-learning algorithm (works with both representations)."""
    q_vals: F["A S"]      # Q-value table [n_actions, n_states]
    gamma: jnp.float32    # Discount factor
    alpha: jnp.float32    # Learning rate

    # Optional: tracking statistics
    update_counts: F["A S"] | None = None  # How many times each (s,a) updated
    step: int = 0
```

---

## 2. Core Operations

### 2.1 Q-Value Lookup (Indexed)

```python
def get_q_value(q_vals: F["A S"], action: int, state: int) -> float:
    """
    Get Q(s, a) using indices.

    Args:
        q_vals: Q-value table [n_actions, n_states]
        action: Action index
        state: State index

    Returns:
        Q-value for the (state, action) pair

    Performance: O(1) memory access vs O(AS) for einsum
    """
    return q_vals[action, state]


def get_state_values(q_vals: F["A S"], state: int) -> F["A"]:
    """
    Get all Q(s, a) for a given state.

    Args:
        q_vals: Q-value table
        state: State index

    Returns:
        Q-values for all actions in this state [n_actions]
    """
    return q_vals[:, state]


def get_max_q_value(q_vals: F["A S"], state: int) -> float:
    """
    Get max_a Q(s, a) for a given state.

    Args:
        q_vals: Q-value table
        state: State index

    Returns:
        Maximum Q-value for this state
    """
    return jnp.max(q_vals[:, state])


def get_greedy_action(q_vals: F["A S"], state: int) -> int:
    """
    Get greedy action for a state.

    Args:
        q_vals: Q-value table
        state: State index

    Returns:
        Action index with highest Q-value
    """
    return jnp.argmax(q_vals[:, state])
```

### 2.2 Q-Value Update (Indexed)

```python
def update_q_value_inplace(
    q_vals: F["A S"],
    action: int,
    state: int,
    delta: float
) -> F["A S"]:
    """
    Update single Q-value in-place (creates new array in JAX).

    Args:
        q_vals: Current Q-value table
        action: Action index
        state: State index
        delta: Amount to add to Q(s, a)

    Returns:
        Updated Q-value table

    Example:
        q_vals = update_q_value_inplace(q_vals, action=2, state=5, delta=0.1)
        # q_vals[2, 5] increased by 0.1
    """
    return q_vals.at[action, state].add(delta)


def set_q_value(
    q_vals: F["A S"],
    action: int,
    state: int,
    value: float
) -> F["A S"]:
    """
    Set Q-value directly (not incremental).

    Useful for initialization or direct policy evaluation.
    """
    return q_vals.at[action, state].set(value)
```

---

## 3. Q-Learning Implementation

### 3.1 Single Update

```python
# jaxdp/learning/algorithms_indexed.py

import jax
import jax.numpy as jnp
from flax import struct

from jaxdp.learning.types_indexed import TransitionIndexed, QLearningState


class q_learning_indexed:
    """
    Q-Learning with indexed state/action representation.

    Optimized for large state spaces where one-hot encoding is wasteful.
    """

    @staticmethod
    def init(
        n_actions: int,
        n_states: int,
        gamma: float,
        alpha: float,
        init_q: float = 0.0,
        track_counts: bool = False,
    ) -> QLearningState:
        """
        Initialize Q-learning state.

        Args:
            n_actions: Number of actions
            n_states: Number of states
            gamma: Discount factor
            alpha: Learning rate
            init_q: Initial Q-value
            track_counts: Whether to track update counts per (s,a)

        Returns:
            Initialized algorithm state
        """
        q_vals = jnp.full((n_actions, n_states), init_q, dtype=jnp.float32)
        update_counts = jnp.zeros((n_actions, n_states), dtype=jnp.int32) if track_counts else None

        return QLearningState(
            q_vals=q_vals,
            gamma=jnp.float32(gamma),
            alpha=jnp.float32(alpha),
            update_counts=update_counts,
            step=0,
        )

    @staticmethod
    def compute_td_error(
        state: QLearningState,
        transition: TransitionIndexed
    ) -> float:
        """
        Compute TD error for a transition.

        TD Error = R + γ max_a' Q(s', a') - Q(s, a)

        Args:
            state: Current algorithm state
            transition: Observed transition

        Returns:
            TD error (scalar)
        """
        # Current Q-value: Q(s, a)
        curr_q = state.q_vals[transition.action, transition.state]

        # Target: R + γ max_a' Q(s', a')
        if transition.terminal:
            # Terminal state: no future value
            target = transition.reward
        else:
            max_next_q = jnp.max(state.q_vals[:, transition.next_state])
            target = transition.reward + state.gamma * max_next_q

        # TD error
        return target - curr_q

    @staticmethod
    def update(
        state: QLearningState,
        transition: TransitionIndexed
    ) -> QLearningState:
        """
        Update Q-values from a single transition.

        Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]

        Args:
            state: Current algorithm state
            transition: Observed transition

        Returns:
            Updated algorithm state

        Performance: O(A) for max operation, O(1) for update
        Memory: O(1) per transition (vs O(S) for one-hot)
        """
        # Compute TD error
        td_error = q_learning_indexed.compute_td_error(state, transition)

        # Update Q-value
        new_q_vals = state.q_vals.at[transition.action, transition.state]\
                                  .add(state.alpha * td_error)

        # Update counts if tracking
        new_counts = state.update_counts
        if new_counts is not None:
            new_counts = new_counts.at[transition.action, transition.state].add(1)

        return state.replace(
            q_vals=new_q_vals,
            update_counts=new_counts,
            step=state.step + 1,
        )
```

### 3.2 Batch Update

```python
    @staticmethod
    def batch_update(
        state: QLearningState,
        transitions: TransitionIndexed
    ) -> QLearningState:
        """
        Update Q-values from a batch of transitions.

        Handles repeated (s,a) pairs by averaging their TD errors.
        This is more statistically sound than summing.

        Args:
            state: Current algorithm state
            transitions: Batch of transitions (fields have leading batch dimension)

        Returns:
            Updated algorithm state

        Performance:
            - One-hot: O(batch * S) memory, O(batch * AS) compute
            - Indexed: O(batch) memory, O(batch * A + AS) compute
        """
        batch_size = transitions.state.shape[0]

        # Compute TD errors for all transitions (vectorized)
        def compute_single_td(trans):
            return q_learning_indexed.compute_td_error(state, trans)

        td_errors = jax.vmap(compute_single_td)(transitions)  # Shape: [batch]

        # Convert (action, state) pairs to linear indices for bincount
        # linear_idx = action * n_states + state
        n_states = state.q_vals.shape[1]
        linear_indices = transitions.action * n_states + transitions.state

        # Sum TD errors for each (s,a) pair using bincount
        total_td = jnp.bincount(
            linear_indices,
            weights=td_errors,
            length=state.q_vals.size,
        ).reshape(state.q_vals.shape)

        # Count occurrences of each (s,a) pair
        counts = jnp.bincount(
            linear_indices,
            length=state.q_vals.size,
        ).reshape(state.q_vals.shape)

        # Average TD errors (avoid division by zero)
        avg_td = jnp.where(counts > 0, total_td / counts, 0.0)

        # Apply updates
        new_q_vals = state.q_vals + state.alpha * avg_td

        # Update counts if tracking
        new_counts = state.update_counts
        if new_counts is not None:
            new_counts = new_counts + counts

        return state.replace(
            q_vals=new_q_vals,
            update_counts=new_counts,
            step=state.step + batch_size,
        )

    @staticmethod
    def batch_update_sum(
        state: QLearningState,
        transitions: TransitionIndexed
    ) -> QLearningState:
        """
        Alternative: sum TD errors instead of averaging.

        This matches the current one-hot implementation more closely.
        Use batch_update() for better statistical properties.
        """
        batch_size = transitions.state.shape[0]

        # Compute all TD errors
        td_errors = jax.vmap(
            lambda t: q_learning_indexed.compute_td_error(state, t)
        )(transitions)

        # Linear indexing
        n_states = state.q_vals.shape[1]
        linear_indices = transitions.action * n_states + transitions.state

        # Sum TD errors (no normalization)
        total_td = jnp.bincount(
            linear_indices,
            weights=td_errors,
            length=state.q_vals.size,
        ).reshape(state.q_vals.shape)

        new_q_vals = state.q_vals + state.alpha * total_td

        return state.replace(q_vals=new_q_vals, step=state.step + batch_size)
```

---

## 4. Sampling Functions

### 4.1 Sample Initial State

```python
# jaxdp/sampling/indexed.py

import jax
import jax.numpy as jnp
import jax.random as jrd
import distrax

from jaxdp.mdp import MDP


def sample_initial_state(mdp: MDP, key: jrd.PRNGKey) -> int:
    """
    Sample initial state from MDP's initial distribution.

    Args:
        mdp: Markov Decision Process
        key: PRNG key

    Returns:
        Initial state index
    """
    dist = distrax.Categorical(probs=mdp.initial)
    return dist.sample(seed=key).astype(jnp.int32)
```

### 4.2 Sample Action from Policy

```python
def sample_action_epsilon_greedy(
    q_vals: F["A S"],
    state: int,
    epsilon: float,
    key: jrd.PRNGKey
) -> int:
    """
    Sample action using epsilon-greedy policy.

    Args:
        q_vals: Q-value table
        state: Current state index
        epsilon: Exploration parameter
        key: PRNG key

    Returns:
        Selected action index
    """
    key_explore, key_random = jrd.split(key)

    # With probability epsilon, choose random action
    explore = jrd.uniform(key_explore) < epsilon

    # Random action
    n_actions = q_vals.shape[0]
    random_action = jrd.randint(key_random, (), 0, n_actions)

    # Greedy action
    greedy_action = jnp.argmax(q_vals[:, state])

    # Choose
    return jax.lax.select(explore, random_action, greedy_action).astype(jnp.int32)


def sample_action_softmax(
    q_vals: F["A S"],
    state: int,
    temperature: float,
    key: jrd.PRNGKey
) -> int:
    """
    Sample action using softmax (Boltzmann) policy.

    Args:
        q_vals: Q-value table
        state: Current state index
        temperature: Temperature parameter (higher = more exploration)
        key: PRNG key

    Returns:
        Selected action index
    """
    logits = q_vals[:, state] / temperature
    dist = distrax.Categorical(logits=logits)
    return dist.sample(seed=key).astype(jnp.int32)
```

### 4.3 Sample Transition

```python
def sample_transition(
    mdp: MDP,
    state: int,
    action: int,
    key: jrd.PRNGKey
) -> TransitionIndexed:
    """
    Sample transition from MDP dynamics.

    Args:
        mdp: Markov Decision Process
        state: Current state index
        action: Action index
        key: PRNG key

    Returns:
        Sampled transition
    """
    # Get transition probabilities for (state, action)
    transition_probs = mdp.transition[action, :, state]  # Shape: [n_states]

    # Sample next state
    next_state_dist = distrax.Categorical(probs=transition_probs)
    next_state = next_state_dist.sample(seed=key).astype(jnp.int32)

    # Get reward
    reward = mdp.reward[action, state, next_state]

    # Check terminal
    terminal = mdp.terminal[next_state].astype(jnp.bool_)

    return TransitionIndexed(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        terminal=terminal,
    )
```

### 4.4 Complete Episode Rollout

```python
def rollout_episode(
    mdp: MDP,
    q_vals: F["A S"],
    epsilon: float,
    max_steps: int,
    key: jrd.PRNGKey
) -> tuple[list[TransitionIndexed], float]:
    """
    Roll out a complete episode using epsilon-greedy policy.

    Args:
        mdp: Markov Decision Process
        q_vals: Current Q-values
        epsilon: Exploration parameter
        max_steps: Maximum episode length
        key: PRNG key

    Returns:
        transitions: List of transitions
        total_reward: Cumulative reward
    """
    # Sample initial state
    key, subkey = jrd.split(key)
    state = sample_initial_state(mdp, subkey)

    transitions = []
    total_reward = 0.0

    for step in range(max_steps):
        # Sample action
        key, subkey = jrd.split(key)
        action = sample_action_epsilon_greedy(q_vals, state, epsilon, subkey)

        # Sample transition
        key, subkey = jrd.split(key)
        transition = sample_transition(mdp, state, action, subkey)

        transitions.append(transition)
        total_reward += transition.reward

        # Check termination
        if transition.terminal:
            break

        # Move to next state
        state = transition.next_state

    return transitions, total_reward
```

---

## 5. Batch Processing

### 5.1 Vectorized Rollouts

```python
def parallel_rollouts(
    mdp: MDP,
    q_vals: F["A S"],
    epsilon: float,
    max_steps: int,
    n_episodes: int,
    key: jrd.PRNGKey
) -> tuple[TransitionIndexed, F["N"]]:
    """
    Run multiple episode rollouts in parallel using vmap.

    Args:
        mdp: Markov Decision Process
        q_vals: Current Q-values
        epsilon: Exploration parameter
        max_steps: Maximum episode length
        n_episodes: Number of parallel episodes
        key: PRNG key

    Returns:
        transitions: Batched transitions (padded to max_steps)
        rewards: Total reward per episode [n_episodes]
    """
    keys = jrd.split(key, n_episodes)

    # Vmap over rollout_episode would be complex due to variable-length episodes
    # Instead, use fixed-length scan-based rollout

    def single_rollout(key):
        return rollout_fixed_length(mdp, q_vals, epsilon, max_steps, key)

    # Vectorize
    results = jax.vmap(single_rollout)(keys)

    return results


def rollout_fixed_length(
    mdp: MDP,
    q_vals: F["A S"],
    epsilon: float,
    max_steps: int,
    key: jrd.PRNGKey
) -> tuple[TransitionIndexed, float]:
    """
    Fixed-length rollout suitable for vmap.

    Continues stepping even after terminal (but with zero rewards).
    """
    key, init_key = jrd.split(key)
    initial_state = sample_initial_state(mdp, init_key)

    def step_fn(carry, _):
        state, key, done, cumulative_reward = carry

        # Split key
        key, action_key, trans_key = jrd.split(key, 3)

        # Sample action (only matters if not done)
        action = sample_action_epsilon_greedy(q_vals, state, epsilon, action_key)

        # Sample transition
        trans = sample_transition(mdp, state, action, trans_key)

        # Update cumulative reward (only if not done)
        reward = jax.lax.select(done, 0.0, trans.reward)
        cumulative_reward = cumulative_reward + reward

        # Update done flag
        done = jnp.logical_or(done, trans.terminal)

        # Next state (stay in terminal if done)
        next_state = jax.lax.select(done, state, trans.next_state)

        return (next_state, key, done, cumulative_reward), trans

    init_carry = (initial_state, key, jnp.bool_(False), 0.0)
    final_carry, transitions = jax.lax.scan(
        step_fn,
        init_carry,
        None,
        length=max_steps
    )

    _, _, _, total_reward = final_carry

    return transitions, total_reward
```

---

## 6. Conversion Utilities

```python
# jaxdp/converters.py

import jax
import jax.numpy as jnp
from jaxdp.typehints import F


def onehot_to_index(onehot: F["N"]) -> int:
    """Convert one-hot vector to index."""
    return jnp.argmax(onehot).astype(jnp.int32)


def index_to_onehot(index: int, num_classes: int) -> F["N"]:
    """Convert index to one-hot vector."""
    return jax.nn.one_hot(index, num_classes)


def batch_onehot_to_indices(onehot_batch: F["B N"]) -> F["B"]:
    """Convert batch of one-hot vectors to indices."""
    return jnp.argmax(onehot_batch, axis=1).astype(jnp.int32)


def batch_indices_to_onehot(indices: F["B"], num_classes: int) -> F["B N"]:
    """Convert batch of indices to one-hot vectors."""
    return jax.nn.one_hot(indices, num_classes)


# Conversion between Transition types

def transition_onehot_to_indexed(
    trans_onehot,
    n_states: int,
    n_actions: int
) -> TransitionIndexed:
    """Convert one-hot Transition to indexed."""
    return TransitionIndexed(
        state=onehot_to_index(trans_onehot.state),
        action=onehot_to_index(trans_onehot.action),
        reward=trans_onehot.reward,
        next_state=onehot_to_index(trans_onehot.next_state),
        terminal=trans_onehot.terminal,
    )


def transition_indexed_to_onehot(
    trans_indexed: TransitionIndexed,
    n_states: int,
    n_actions: int
):
    """Convert indexed Transition to one-hot."""
    from jaxdp.learning.algorithms import Transition
    return Transition(
        state=index_to_onehot(trans_indexed.state, n_states),
        action=index_to_onehot(trans_indexed.action, n_actions),
        reward=trans_indexed.reward,
        next_state=index_to_onehot(trans_indexed.next_state, n_states),
        terminal=trans_indexed.terminal,
    )
```

---

## 7. Benchmarking Code

### 7.1 Memory Benchmark

```python
# benchmarks/memory_comparison.py

import jax
import jax.numpy as jnp
import tracemalloc
from jaxdp.mdp import GridWorld
from jaxdp.learning.algorithms import q_learning, Transition
from jaxdp.learning.algorithms_indexed import q_learning_indexed, TransitionIndexed


def measure_memory_onehot(n_states, n_actions, batch_size):
    """Measure memory for one-hot approach."""

    tracemalloc.start()

    # Create dummy transitions
    transitions = []
    for _ in range(batch_size):
        state = jax.nn.one_hot(0, n_states)
        action = jax.nn.one_hot(0, n_actions)
        transitions.append(Transition(
            state=state,
            action=action,
            reward=0.0,
            next_state=state,
            terminal=False
        ))

    # Stack into batch
    batch = jax.tree_map(lambda *xs: jnp.stack(xs), *transitions)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak


def measure_memory_indexed(n_states, n_actions, batch_size):
    """Measure memory for indexed approach."""

    tracemalloc.start()

    # Create dummy transitions
    states = jnp.zeros(batch_size, dtype=jnp.int32)
    actions = jnp.zeros(batch_size, dtype=jnp.int32)
    rewards = jnp.zeros(batch_size, dtype=jnp.float32)

    batch = TransitionIndexed(
        state=states,
        action=actions,
        reward=rewards,
        next_state=states,
        terminal=jnp.zeros(batch_size, dtype=jnp.bool_),
    )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak


def run_memory_benchmark():
    """Run comprehensive memory benchmark."""

    state_sizes = [100, 1_000, 10_000, 100_000]
    batch_sizes = [1, 10, 100, 1_000]

    results = []

    for n_states in state_sizes:
        for batch_size in batch_sizes:
            mem_onehot = measure_memory_onehot(n_states, 10, batch_size)
            mem_indexed = measure_memory_indexed(n_states, 10, batch_size)

            results.append({
                'n_states': n_states,
                'batch_size': batch_size,
                'memory_onehot_MB': mem_onehot / 1024 / 1024,
                'memory_indexed_MB': mem_indexed / 1024 / 1024,
                'savings_ratio': mem_onehot / mem_indexed,
            })

    return results
```

### 7.2 Speed Benchmark

```python
# benchmarks/speed_comparison.py

import time
import jax
import jax.numpy as jnp
from jaxdp.learning.algorithms import q_learning
from jaxdp.learning.algorithms_indexed import q_learning_indexed


def benchmark_single_update(n_states, n_actions, n_iterations):
    """Benchmark single update speed."""

    # One-hot setup
    key = jax.random.PRNGKey(0)
    state_oh = q_learning.init(
        MDP(n_states, n_actions, ...),
        key,
        gamma=0.99,
        alpha=0.1
    )

    trans_oh = Transition(
        state=jax.nn.one_hot(0, n_states),
        action=jax.nn.one_hot(0, n_actions),
        reward=1.0,
        next_state=jax.nn.one_hot(1, n_states),
        terminal=False
    )

    # JIT compile
    update_oh = jax.jit(q_learning.update)
    _ = update_oh(state_oh, trans_oh)

    # Time one-hot
    start = time.time()
    for _ in range(n_iterations):
        state_oh = update_oh(state_oh, trans_oh)
    state_oh.q_vals.block_until_ready()  # Wait for GPU
    time_oh = time.time() - start

    # Indexed setup
    state_idx = q_learning_indexed.init(n_actions, n_states, 0.99, 0.1)
    trans_idx = TransitionIndexed(
        state=jnp.int32(0),
        action=jnp.int32(0),
        reward=1.0,
        next_state=jnp.int32(1),
        terminal=False
    )

    # JIT compile
    update_idx = jax.jit(q_learning_indexed.update)
    _ = update_idx(state_idx, trans_idx)

    # Time indexed
    start = time.time()
    for _ in range(n_iterations):
        state_idx = update_idx(state_idx, trans_idx)
    state_idx.q_vals.block_until_ready()
    time_idx = time.time() - start

    return {
        'time_onehot': time_oh,
        'time_indexed': time_idx,
        'speedup': time_oh / time_idx,
    }


def benchmark_batch_update(n_states, n_actions, batch_size, n_iterations):
    """Benchmark batch update speed."""

    # Similar structure to single update benchmark
    # ... (implement batch version)

    pass
```

---

## 8. Testing

### 8.1 Unit Tests

```python
# tests/test_indexed_learning.py

import pytest
import jax
import jax.numpy as jnp
from jaxdp.learning.algorithms_indexed import q_learning_indexed, TransitionIndexed


class TestTransitionIndexed:

    def test_from_onehot(self):
        """Test conversion from one-hot to indexed."""
        state_oh = jnp.array([0, 0, 1, 0, 0])
        action_oh = jnp.array([0, 1, 0])

        trans = TransitionIndexed.from_onehot(
            state_oh, action_oh, 1.0, state_oh, False
        )

        assert trans.state == 2
        assert trans.action == 1
        assert trans.reward == 1.0

    def test_to_onehot(self):
        """Test conversion from indexed to one-hot."""
        trans_idx = TransitionIndexed(
            state=2, action=1, reward=1.0, next_state=3, terminal=False
        )

        trans_oh = trans_idx.to_onehot(n_states=5, n_actions=3)

        assert jnp.argmax(trans_oh.state) == 2
        assert jnp.argmax(trans_oh.action) == 1


class TestQLearningIndexed:

    def test_init(self):
        """Test initialization."""
        state = q_learning_indexed.init(
            n_actions=4,
            n_states=10,
            gamma=0.99,
            alpha=0.1,
        )

        assert state.q_vals.shape == (4, 10)
        assert state.gamma == 0.99
        assert state.alpha == 0.1

    def test_compute_td_error(self):
        """Test TD error computation."""
        state = q_learning_indexed.init(4, 10, 0.99, 0.1)

        # Set up known Q-values
        state = state.replace(q_vals=jnp.zeros((4, 10)))
        state.q_vals = state.q_vals.at[1, 2].set(5.0)  # Q(s=2, a=1) = 5
        state.q_vals = state.q_vals.at[2, 3].set(10.0)  # Q(s=3, a=2) = 10

        # Transition: s=2, a=1, r=1, s'=3
        trans = TransitionIndexed(
            state=2, action=1, reward=1.0, next_state=3, terminal=False
        )

        td_error = q_learning_indexed.compute_td_error(state, trans)

        # Expected: 1 + 0.99 * 10 - 5 = 5.9
        expected = 1.0 + 0.99 * 10.0 - 5.0
        assert jnp.isclose(td_error, expected)

    def test_single_update(self):
        """Test single update modifies Q-value correctly."""
        state = q_learning_indexed.init(4, 10, 0.99, 0.1)

        trans = TransitionIndexed(
            state=2, action=1, reward=1.0, next_state=3, terminal=False
        )

        updated = q_learning_indexed.update(state, trans)

        # Q-value should have changed
        assert updated.q_vals[1, 2] != state.q_vals[1, 2]
        # Step should increment
        assert updated.step == 1

    def test_batch_update(self):
        """Test batch update handles repeated (s,a) pairs."""
        state = q_learning_indexed.init(4, 10, 0.99, 0.1)

        # Two transitions with same (s,a)
        states = jnp.array([2, 2], dtype=jnp.int32)
        actions = jnp.array([1, 1], dtype=jnp.int32)
        rewards = jnp.array([1.0, 2.0])
        next_states = jnp.array([3, 4], dtype=jnp.int32)
        terminals = jnp.array([False, False])

        batch = TransitionIndexed(
            state=states,
            action=actions,
            reward=rewards,
            next_state=next_states,
            terminal=terminals,
        )

        updated = q_learning_indexed.batch_update(state, batch)

        # Should average the two TD errors
        assert updated.step == 2

    def test_terminal_state_handling(self):
        """Test that terminal states have zero future value."""
        state = q_learning_indexed.init(4, 10, 0.99, 0.1)

        # Set high Q-values
        state = state.replace(q_vals=jnp.ones((4, 10)) * 100)

        # Terminal transition
        trans = TransitionIndexed(
            state=2, action=1, reward=1.0, next_state=3, terminal=True
        )

        td_error = q_learning_indexed.compute_td_error(state, trans)

        # Expected: 1.0 - 100 = -99 (no future value due to terminal)
        expected = 1.0 - 100.0
        assert jnp.isclose(td_error, expected)
```

---

## 9. Example Usage

```python
# examples/learning/train_qlearning_indexed.py

import jax
import jax.random as jrd
from jaxdp.mdp import GridWorld
from jaxdp.learning.algorithms_indexed import q_learning_indexed
from jaxdp.sampling.indexed import (
    sample_initial_state,
    sample_action_epsilon_greedy,
    sample_transition,
)


def train_qlearning_gridworld():
    """Train Q-learning on GridWorld using indexed representation."""

    # Create MDP
    mdp = GridWorld(height=10, width=10, slip=0.1)

    # Initialize Q-learning
    key = jrd.PRNGKey(0)
    state = q_learning_indexed.init(
        n_actions=mdp.action_size,
        n_states=mdp.state_size,
        gamma=0.99,
        alpha=0.1,
    )

    # Training loop
    n_episodes = 1000
    max_steps = 100
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    epsilon = epsilon_start

    for episode in range(n_episodes):
        # Reset environment
        key, subkey = jrd.split(key)
        current_state = sample_initial_state(mdp, subkey)

        episode_reward = 0.0

        for step in range(max_steps):
            # Choose action
            key, subkey = jrd.split(key)
            action = sample_action_epsilon_greedy(
                state.q_vals, current_state, epsilon, subkey
            )

            # Take step
            key, subkey = jrd.split(key)
            transition = sample_transition(mdp, current_state, action, subkey)

            # Update Q-values
            state = q_learning_indexed.update(state, transition)

            episode_reward += transition.reward

            if transition.terminal:
                break

            current_state = transition.next_state

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % 100 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, ε = {epsilon:.3f}")

    return state.q_vals


if __name__ == "__main__":
    q_vals = train_qlearning_gridworld()
    print("Training complete!")
    print(f"Q-values shape: {q_vals.shape}")
    print(f"Mean Q-value: {q_vals.mean():.3f}")
```

---

*End of Prototype Document*
