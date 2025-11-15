#!/usr/bin/env python3
"""
Quick benchmark: One-hot vs Indexed Q-learning

Measures ACTUAL memory and speed for TD learning updates.
"""

import jax
import jax.numpy as jnp
import jax.random as jrd
import time
import tracemalloc
from jaxdp.mdp import GridWorld


# ============================================================================
# One-Hot Implementation (Current)
# ============================================================================

def q_learning_update_onehot(q_vals, state_oh, action_oh, reward, next_state_oh,
                             terminal, alpha, gamma):
    """Q-learning update with one-hot states/actions."""
    # Extract Q(s,a) using einsum
    curr_q = jnp.einsum("as,a,s->", q_vals, action_oh, state_oh)

    # Extract max Q(s',a')
    q_next = jnp.einsum("as,s->a", q_vals, next_state_oh)
    max_next_q = jnp.max(q_next)

    # TD target
    target = reward + gamma * max_next_q * (1 - terminal)
    td_error = target - curr_q

    # Create update mask
    update = jnp.einsum("a,s->as", action_oh, state_oh)

    # Apply update
    return q_vals + alpha * td_error * update


def collect_transitions_onehot(mdp, n_samples, key):
    """Collect transitions with one-hot representation."""
    transitions = []

    # Sample initial state (one-hot)
    state = jnp.zeros(mdp.state_size)
    state = state.at[0].set(1.0)

    for i in range(n_samples):
        key, subkey = jrd.split(key)

        # Random action (one-hot)
        action_idx = jrd.randint(subkey, (), 0, mdp.action_size)
        action = jax.nn.one_hot(action_idx, mdp.action_size)

        # Sample next state
        key, subkey = jrd.split(key)
        transition_probs = jnp.einsum("a,axs,s->x", action, mdp.transition, state)
        next_state_idx = jrd.choice(subkey, mdp.state_size, p=transition_probs)
        next_state = jax.nn.one_hot(next_state_idx, mdp.state_size)

        # Get reward
        reward = jnp.einsum("asx,a,s,x->", mdp.reward, action, state, next_state)

        # Terminal
        terminal = mdp.terminal[next_state_idx]

        transitions.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'terminal': terminal
        })

        state = next_state

    return transitions


# ============================================================================
# Indexed Implementation (Proposed)
# ============================================================================

def q_learning_update_indexed(q_vals, state_idx, action_idx, reward, next_state_idx,
                              terminal, alpha, gamma):
    """Q-learning update with indexed states/actions."""
    # Direct indexing - no einsum
    curr_q = q_vals[action_idx, state_idx]

    # Max over actions for next state
    max_next_q = jnp.max(q_vals[:, next_state_idx])

    # TD target
    target = reward + gamma * max_next_q * (1 - terminal)
    td_error = target - curr_q

    # In-place update
    return q_vals.at[action_idx, state_idx].add(alpha * td_error)


def collect_transitions_indexed(mdp, n_samples, key):
    """Collect transitions with indexed representation."""
    transitions = []

    # Sample initial state (index)
    state = 0

    for i in range(n_samples):
        key, subkey = jrd.split(key)

        # Random action (index)
        action = jrd.randint(subkey, (), 0, mdp.action_size)

        # Sample next state
        key, subkey = jrd.split(key)
        transition_probs = mdp.transition[action, :, state]
        next_state = jrd.choice(subkey, mdp.state_size, p=transition_probs)

        # Get reward
        reward = mdp.reward[action, state, next_state]

        # Terminal
        terminal = mdp.terminal[next_state]

        transitions.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'terminal': terminal
        })

        state = int(next_state)

    return transitions


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_memory(mdp, n_samples, key):
    """Measure memory usage for transition collection."""

    print(f"\n{'='*70}")
    print(f"Memory Benchmark: {n_samples} transitions, {mdp.state_size} states")
    print(f"{'='*70}")

    # One-hot memory
    tracemalloc.start()
    transitions_oh = collect_transitions_onehot(mdp, n_samples, key)
    current, peak_oh = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Indexed memory
    tracemalloc.start()
    transitions_idx = collect_transitions_indexed(mdp, n_samples, key)
    current, peak_idx = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"One-hot memory:  {peak_oh / 1024 / 1024:.2f} MB")
    print(f"Indexed memory:  {peak_idx / 1024 / 1024:.2f} MB")
    print(f"Savings ratio:   {peak_oh / peak_idx:.1f}x")

    return transitions_oh, transitions_idx


def benchmark_speed(mdp, transitions_oh, transitions_idx, n_iterations=1000):
    """Measure update speed."""

    print(f"\n{'='*70}")
    print(f"Speed Benchmark: {n_iterations} updates")
    print(f"{'='*70}")

    # Initialize Q-values
    q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
    alpha = 0.1
    gamma = 0.99

    # JIT compile both versions
    update_oh_jit = jax.jit(q_learning_update_onehot, static_argnames=('alpha', 'gamma'))
    update_idx_jit = jax.jit(q_learning_update_indexed, static_argnames=('alpha', 'gamma'))

    # Warm up
    trans = transitions_oh[0]
    _ = update_oh_jit(q_vals, trans['state'], trans['action'], trans['reward'],
                     trans['next_state'], trans['terminal'], alpha, gamma)

    trans = transitions_idx[0]
    _ = update_idx_jit(q_vals, trans['state'], trans['action'], trans['reward'],
                      trans['next_state'], trans['terminal'], alpha, gamma)

    # Benchmark one-hot
    q_vals_oh = jnp.zeros((mdp.action_size, mdp.state_size))
    start = time.time()
    for i in range(n_iterations):
        trans = transitions_oh[i % len(transitions_oh)]
        q_vals_oh = update_oh_jit(q_vals_oh, trans['state'], trans['action'],
                                   trans['reward'], trans['next_state'],
                                   trans['terminal'], alpha, gamma)
    q_vals_oh.block_until_ready()  # Wait for GPU
    time_oh = time.time() - start

    # Benchmark indexed
    q_vals_idx = jnp.zeros((mdp.action_size, mdp.state_size))
    start = time.time()
    for i in range(n_iterations):
        trans = transitions_idx[i % len(transitions_idx)]
        q_vals_idx = update_idx_jit(q_vals_idx, trans['state'], trans['action'],
                                    trans['reward'], trans['next_state'],
                                    trans['terminal'], alpha, gamma)
    q_vals_idx.block_until_ready()
    time_idx = time.time() - start

    print(f"One-hot time:    {time_oh:.4f} seconds ({n_iterations/time_oh:.1f} updates/sec)")
    print(f"Indexed time:    {time_idx:.4f} seconds ({n_iterations/time_idx:.1f} updates/sec)")
    print(f"Speedup:         {time_oh / time_idx:.2f}x")

    # Verify they produce similar results
    diff = jnp.abs(q_vals_oh - q_vals_idx).max()
    print(f"\nMax Q-value diff: {diff:.6f} (should be ~0)")


def benchmark_batch_updates(mdp, transitions_oh, transitions_idx):
    """Benchmark batch updates."""

    print(f"\n{'='*70}")
    print(f"Batch Update Benchmark: {len(transitions_oh)} transitions")
    print(f"{'='*70}")

    q_vals = jnp.zeros((mdp.action_size, mdp.action_size))
    alpha = 0.1
    gamma = 0.99

    # One-hot batch
    states_oh = jnp.stack([t['state'] for t in transitions_oh])
    actions_oh = jnp.stack([t['action'] for t in transitions_oh])
    rewards = jnp.stack([t['reward'] for t in transitions_oh])
    next_states_oh = jnp.stack([t['next_state'] for t in transitions_oh])
    terminals = jnp.stack([t['terminal'] for t in transitions_oh])

    # Indexed batch
    states_idx = jnp.array([t['state'] for t in transitions_idx], dtype=jnp.int32)
    actions_idx = jnp.array([t['action'] for t in transitions_idx], dtype=jnp.int32)
    rewards_idx = jnp.stack([t['reward'] for t in transitions_idx])
    next_states_idx = jnp.array([t['next_state'] for t in transitions_idx], dtype=jnp.int32)
    terminals_idx = jnp.stack([t['terminal'] for t in transitions_idx])

    # Memory comparison
    tracemalloc.start()
    _ = (states_oh, actions_oh, rewards, next_states_oh, terminals)
    _, peak_oh = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    _ = (states_idx, actions_idx, rewards_idx, next_states_idx, terminals_idx)
    _, peak_idx = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Batch memory (one-hot): {peak_oh / 1024 / 1024:.2f} MB")
    print(f"Batch memory (indexed): {peak_idx / 1024 / 1024:.2f} MB")
    print(f"Savings ratio:          {peak_oh / peak_idx:.1f}x")


def run_all_benchmarks():
    """Run comprehensive benchmarks."""

    print("\n" + "="*70)
    print("TD LEARNING BENCHMARK: One-Hot vs Indexed")
    print("="*70)

    key = jrd.PRNGKey(0)

    # Test different MDP sizes
    configs = [
        (10, 10, 100),      # Small: 100 states
        (32, 32, 100),      # Medium: 1024 states
        (100, 100, 100),    # Large: 10,000 states
        (316, 316, 100),    # Very Large: ~100K states
    ]

    for height, width, n_samples in configs:
        print(f"\n{'#'*70}")
        print(f"# GridWorld {height}x{width} ({height*width} states)")
        print(f"{'#'*70}")

        mdp = GridWorld(height=height, width=width, slip=0.0)

        key, subkey = jrd.split(key)

        # Memory benchmark
        transitions_oh, transitions_idx = benchmark_memory(mdp, n_samples, subkey)

        # Speed benchmark (fewer iterations for large MDPs)
        n_iterations = min(1000, 10000 // (height * width // 100))
        benchmark_speed(mdp, transitions_oh, transitions_idx, n_iterations)

        # Batch benchmark
        benchmark_batch_updates(mdp, transitions_oh, transitions_idx)


if __name__ == "__main__":
    # Make sure JAX is using CPU for fair comparison (optional)
    # jax.config.update('jax_platform_name', 'cpu')

    run_all_benchmarks()
