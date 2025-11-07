"""
Demonstration: Building DP Algorithms with Compositional Language

This script shows practical examples of using the compositional DP operator
language to build custom algorithms.
"""

import jax
import jax.numpy as jnp
from jaxdp.mdp import garnet_mdp
from examples.algorithms import vi, pi
from examples.compositional_operators import (
    OptimisticVI,
    TemperatureScheduledVI,
    MomentumPI,
    OperatorComposition,
    PolicyCombinator,
    CompositionalSampling,
    DPOperator,
    DPDSL,
)


def demo_basic_composition():
    """
    Demo 1: Basic operator composition using primitives
    """
    print("=" * 70)
    print("DEMO 1: Basic Operator Composition")
    print("=" * 70)

    # Create MDP
    key = jax.random.PRNGKey(42)
    mdp = garnet_mdp(key, num_states=50, num_actions=4, branch_factor=3)

    # Initialize different algorithm states
    gamma = 0.99
    key, *subkeys = jax.random.split(key, 4)

    vi_state = vi.init(mdp, subkeys[0], gamma)
    pi_state = pi.init(mdp, subkeys[1], gamma)
    optimistic_state = OptimisticVI.init(mdp, subkeys[2], gamma, optimism_bonus=2.0)

    print(f"\n✓ Created MDP with {mdp.num_states} states, {mdp.num_actions} actions")
    print(f"✓ Initialized 3 algorithm states")

    # Run updates
    print("\n--- Running single updates ---")

    # Standard VI update
    vi_state_new = vi.update(vi_state, mdp, step=0)
    vi_delta = jnp.max(jnp.abs(vi_state_new.q_vals - vi_state.q_vals))
    print(f"VI update delta: {vi_delta:.6f}")

    # Optimistic VI (composed with bonus)
    opt_state_new = OptimisticVI.update(optimistic_state, mdp, step=0)
    opt_delta = jnp.max(jnp.abs(opt_state_new.q_vals - optimistic_state.q_vals))
    print(f"Optimistic VI update delta: {opt_delta:.6f}")
    print(f"  (Notice larger delta due to optimism bonus)")

    # Blended VI+PI update
    blended_update = OperatorComposition.blend(vi.update, pi.update, alpha=0.6)
    blended_state = blended_update(vi_state, mdp, step=0)
    blend_delta = jnp.max(jnp.abs(blended_state.q_vals - vi_state.q_vals))
    print(f"Blended (60% VI + 40% PI) delta: {blend_delta:.6f}")


def demo_policy_composition():
    """
    Demo 2: Composing policies from primitives
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Policy Composition")
    print("=" * 70)

    from jaxdp import greedy_policy, soft_policy, e_greedy_policy

    # Create simple Q-values
    q_vals = jnp.array([
        [1.0, 0.5, 0.3, 2.0, 0.8],  # Action 0
        [0.8, 2.0, 0.4, 1.5, 1.2],  # Action 1
        [0.6, 1.0, 3.0, 0.7, 1.5],  # Action 2
    ])

    print("\nQ-values (3 actions × 5 states):")
    print(q_vals)

    # Extract different policies from same Q-values
    print("\n--- Primitive Policies ---")

    greedy_pi = greedy_policy.q(q_vals)
    print("\nGreedy policy (one-hot on max Q):")
    print(greedy_pi)

    soft_pi = soft_policy.q(q_vals, temperature=1.0)
    print("\nSoft policy (temperature=1.0):")
    print(soft_pi)

    egreedy_pi = e_greedy_policy.q(q_vals, epsilon=0.2)
    print("\nEpsilon-greedy policy (ε=0.2):")
    print(egreedy_pi)

    # Composite policy
    print("\n--- Composite Policy ---")
    composite_pi = PolicyCombinator.epsilon_boltzmann(
        q_vals, epsilon=0.1, temperature=0.5
    )
    print("\nEpsilon-Boltzmann blend (70% ε-greedy + 30% softmax):")
    print(composite_pi)


def demo_operator_overloading():
    """
    Demo 3: Using operator overloading DSL
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Operator Overloading Language")
    print("=" * 70)

    # Wrap algorithms in DPOperator for symbolic composition
    vi_op = DPOperator(vi.update, vi.State)
    pi_op = DPOperator(pi.update, pi.State)

    print("\n--- Building Algorithms with Operators ---")

    # Create blended algorithm: 70% VI + 30% PI
    hybrid = (vi_op * 0.7) + (pi_op * 0.3)
    print("✓ hybrid = (vi_op * 0.7) + (pi_op * 0.3)")
    print("  → Blends 70% VI updates with 30% PI updates")

    # Create alternating algorithm
    alternating = vi_op | pi_op
    print("\n✓ alternating = vi_op | pi_op")
    print("  → Alternates between VI and PI every N steps")

    # Create pipeline: VI then PI
    pipeline = vi_op >> pi_op
    print("\n✓ pipeline = vi_op >> pi_op")
    print("  → Applies VI update, then PI update sequentially")

    # Test execution
    key = jax.random.PRNGKey(123)
    mdp = garnet_mdp(key, num_states=20, num_actions=3, branch_factor=2)
    state = vi.init(mdp, key, gamma=0.95)

    print("\n--- Executing Hybrid Operator ---")
    new_state = hybrid(state, mdp, step=0)
    print(f"✓ Executed (vi*0.7 + pi*0.3) successfully")
    print(f"  Q-value range: [{jnp.min(new_state.q_vals):.3f}, {jnp.max(new_state.q_vals):.3f}]")


def demo_higher_order_combinators():
    """
    Demo 4: Higher-order operator combinators
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Higher-Order Combinators")
    print("=" * 70)

    key = jax.random.PRNGKey(456)
    mdp = garnet_mdp(key, num_states=30, num_actions=4, branch_factor=3)

    # Conditional operator: explore first 20 steps, then exploit
    print("\n--- Conditional Operator ---")

    temp_vi = TemperatureScheduledVI
    standard_vi = vi

    adaptive_alg = OperatorComposition.conditional(
        condition=lambda step: step < 20,
        op_true=lambda s, m, t: temp_vi.update(s, m, t, total_steps=20),
        op_false=standard_vi.update,
    )

    print("✓ Created adaptive algorithm:")
    print("  • Steps 0-19:  Temperature-scheduled VI (exploration)")
    print("  • Steps 20+:   Standard VI (exploitation)")

    # Test early and late behavior
    state = temp_vi.init(mdp, key, gamma=0.99)

    print("\n--- Testing Adaptive Behavior ---")
    state_early = adaptive_alg(state, mdp, step=5)
    print(f"Step 5 (exploring):  max|Q| = {jnp.max(jnp.abs(state_early.q_vals)):.4f}")

    state_late = adaptive_alg(state_early, mdp, step=25)
    print(f"Step 25 (exploiting): max|Q| = {jnp.max(jnp.abs(state_late.q_vals)):.4f}")


def demo_custom_sampling():
    """
    Demo 5: Compositional sampling methods
    """
    print("\n" + "=" * 70)
    print("DEMO 5: Compositional Sampling")
    print("=" * 70)

    key = jax.random.PRNGKey(789)
    mdp = garnet_mdp(key, num_states=10, num_actions=3, branch_factor=2)

    # Run VI to get Q-values
    state = vi.init(mdp, key, gamma=0.95)
    for _ in range(50):
        state = vi.update(state, mdp, step=_)

    print(f"\n✓ Trained VI for 50 iterations")
    print(f"  Final Q-value range: [{jnp.min(state.q_vals):.3f}, {jnp.max(state.q_vals):.3f}]")

    # Multi-policy sampling
    print("\n--- Multi-Policy Ensemble Sampling ---")
    key, subkey = jax.random.split(key)
    all_actions, mean_actions = CompositionalSampling.multi_policy_sample(
        subkey, mdp, state.q_vals, state=0, num_samples=100
    )

    print(f"Sampled from 3 policy types × 100 samples each")
    print(f"Action distributions shape: {all_actions.shape}")
    print(f"Mean actions per policy: {mean_actions}")

    # Adaptive rollout
    print("\n--- Adaptive Rollout ---")
    key, subkey = jax.random.split(key)
    final_state, total_reward = CompositionalSampling.adaptive_rollout(
        subkey, mdp, state.q_vals, state=0, horizon=20, gamma=0.95
    )

    print(f"Rollout from state 0 → state {final_state}")
    print(f"Total discounted reward: {total_reward:.4f}")
    print("(Used soft policy for first 10 steps, greedy for last 10)")


def demo_dsl():
    """
    Demo 6: Domain-Specific Language for algorithm construction
    """
    print("\n" + "=" * 70)
    print("DEMO 6: Fluent DSL for Algorithm Construction")
    print("=" * 70)

    key = jax.random.PRNGKey(999)
    mdp = garnet_mdp(key, num_states=25, num_actions=4, branch_factor=2)

    print("\n--- Building Algorithm with Fluent API ---")
    print("\nCode:")
    print("  algorithm = (")
    print("      DPDSL(mdp)")
    print("      .with_exploration(temperature=2.0)")
    print("      .with_optimism(bonus=1.0)")
    print("      .with_momentum(beta=0.9)")
    print("      .with_exploitation()")
    print("      .build()")
    print("  )")

    algorithm = (
        DPDSL(mdp)
        .with_exploration(temperature=2.0)
        .with_optimism(bonus=1.0)
        .with_momentum(beta=0.9)
        .with_exploitation()
        .build()
    )

    print("\n✓ Built custom algorithm with compositional DSL")
    print("  Combines: exploration → optimism → momentum → exploitation")


def main():
    """Run all demonstrations"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Compositional DP Operator Language Demonstration".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    demos = [
        demo_basic_composition,
        demo_policy_composition,
        demo_operator_overloading,
        demo_higher_order_combinators,
        demo_custom_sampling,
        demo_dsl,
    ]

    for demo_fn in demos:
        try:
            demo_fn()
        except Exception as e:
            print(f"\n❌ Error in {demo_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All demonstrations completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Operators are composable functions (functional composition)")
    print("  2. Policies can be mixed, blended, and combined from primitives")
    print("  3. Higher-order combinators create complex behaviors")
    print("  4. Operator overloading adds symbolic DSL syntax")
    print("  5. Custom sampling methods compose existing samplers")
    print("  6. Fluent APIs provide readable algorithm construction")
    print()


if __name__ == "__main__":
    main()
