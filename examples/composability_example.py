"""
Composability Example: Building Complex Algorithms from Simple Primitives

This example demonstrates how to build increasingly complex DP algorithms
by composing simple primitives in a step-by-step manner.
"""

import jax
import jax.numpy as jnp
from flax import struct

# Import primitives
from jaxdp import (
    greedy_policy,
    soft_policy,
    e_greedy_policy,
    policy_evaluation,
    bellman_optimality_operator,
)
from jaxdp.mdp import garnet_mdp
from jaxdp.typehints import MDP, QType, StaticMeta


# =============================================================================
# LEVEL 1: Using Primitives Directly
# =============================================================================

def level_1_primitives_only():
    """
    Most basic: Use primitives directly without any composition.
    """
    print("\n" + "="*70)
    print("LEVEL 1: Using Primitives Directly (No Composition)")
    print("="*70)

    # Setup
    key = jax.random.PRNGKey(42)
    mdp = garnet_mdp(key, num_states=20, num_actions=3, branch_factor=2)
    q_vals = jnp.zeros((mdp.num_actions, mdp.num_states))
    gamma = 0.95

    print(f"\nMDP: {mdp.num_states} states, {mdp.num_actions} actions")

    # Use primitives one at a time
    print("\n1. Extract greedy policy from Q-values:")
    policy = greedy_policy.q(q_vals)
    print(f"   Policy shape: {policy.shape}")

    print("\n2. Evaluate that policy:")
    q_new = policy_evaluation.q(mdp, policy, gamma)
    print(f"   New Q-values range: [{jnp.min(q_new):.3f}, {jnp.max(q_new):.3f}]")

    print("\n3. Apply Bellman optimality operator:")
    q_optimal = bellman_optimality_operator.q(mdp, q_new, gamma)
    print(f"   Optimal Q-values range: [{jnp.min(q_optimal):.3f}, {jnp.max(q_optimal):.3f}]")

    print("\n✓ No composition yet - just calling primitives individually")


# =============================================================================
# LEVEL 2: Simple Sequential Composition (Policy Iteration)
# =============================================================================

class SimplePolicyIteration(metaclass=StaticMeta):
    """
    Compose two primitives sequentially:
    greedy_policy → policy_evaluation
    """

    @struct.dataclass
    class State:
        q_vals: QType
        gamma: jnp.ndarray

    @staticmethod
    def init(mdp: MDP, key, gamma: float):
        return SimplePolicyIteration.State(
            q_vals=jnp.zeros((mdp.num_actions, mdp.num_states)),
            gamma=jnp.array(gamma),
        )

    @staticmethod
    def update(state: "SimplePolicyIteration.State", mdp: MDP, step: int):
        # COMPOSITION: Chain two primitives
        policy = greedy_policy.q(state.q_vals)                      # Primitive 1
        q_vals = policy_evaluation.q(mdp, policy, state.gamma)      # Primitive 2
        return state.replace(q_vals=q_vals)


def level_2_sequential_composition():
    """
    Compose primitives sequentially to create Policy Iteration.
    """
    print("\n" + "="*70)
    print("LEVEL 2: Sequential Composition (Primitive 1 → Primitive 2)")
    print("="*70)

    # Setup
    key = jax.random.PRNGKey(42)
    mdp = garnet_mdp(key, num_states=20, num_actions=3, branch_factor=2)

    print("\nAlgorithm: Policy Iteration")
    print("Composition: greedy_policy → policy_evaluation")

    # Run algorithm
    state = SimplePolicyIteration.init(mdp, key, gamma=0.95)

    print("\nRunning 10 iterations...")
    for step in range(10):
        state = SimplePolicyIteration.update(state, mdp, step)
        if step % 3 == 0:
            max_q = jnp.max(state.q_vals)
            print(f"  Step {step:2d}: max|Q| = {max_q:8.4f}")

    print("\n✓ Built Policy Iteration by composing greedy_policy + policy_evaluation")


# =============================================================================
# LEVEL 3: Parametric Composition (Temperature-Scheduled Soft PI)
# =============================================================================

class SoftPolicyIteration(metaclass=StaticMeta):
    """
    Compose primitives with time-varying parameters:
    soft_policy(temperature=f(t)) → policy_evaluation
    """

    @struct.dataclass
    class State:
        q_vals: QType
        gamma: jnp.ndarray
        temperature: float

    @staticmethod
    def init(mdp: MDP, key, gamma: float, temperature: float = 2.0):
        return SoftPolicyIteration.State(
            q_vals=jnp.zeros((mdp.num_actions, mdp.num_states)),
            gamma=jnp.array(gamma),
            temperature=temperature,
        )

    @staticmethod
    def update(state: "SoftPolicyIteration.State", mdp: MDP, step: int):
        # COMPOSITION: Soft policy with temperature + evaluation + annealing
        policy = soft_policy.q(state.q_vals, state.temperature)     # Primitive 1 (parametric)
        q_vals = policy_evaluation.q(mdp, policy, state.gamma)      # Primitive 2
        new_temp = state.temperature * 0.95                         # Parameter schedule
        return state.replace(q_vals=q_vals, temperature=new_temp)


def level_3_parametric_composition():
    """
    Add parameter scheduling to primitive composition.
    """
    print("\n" + "="*70)
    print("LEVEL 3: Parametric Composition (Primitive with Schedule)")
    print("="*70)

    # Setup
    key = jax.random.PRNGKey(42)
    mdp = garnet_mdp(key, num_states=20, num_actions=3, branch_factor=2)

    print("\nAlgorithm: Soft Policy Iteration with Temperature Annealing")
    print("Composition: soft_policy(temp=f(t)) → policy_evaluation")

    # Run algorithm
    state = SoftPolicyIteration.init(mdp, key, gamma=0.95, temperature=5.0)

    print("\nRunning 10 iterations with temperature decay...")
    for step in range(10):
        state = SoftPolicyIteration.update(state, mdp, step)
        if step % 3 == 0:
            max_q = jnp.max(state.q_vals)
            print(f"  Step {step:2d}: max|Q| = {max_q:8.4f}, temp = {state.temperature:.3f}")

    print("\n✓ Added parameter scheduling to primitive composition")


# =============================================================================
# LEVEL 4: Parallel Composition (Blended VI + PI)
# =============================================================================

class BlendedAlgorithm(metaclass=StaticMeta):
    """
    Compose primitives in parallel and blend results:
    (bellman_optimality + greedy→evaluation) → weighted average
    """

    @struct.dataclass
    class State:
        q_vals: QType
        gamma: jnp.ndarray
        vi_weight: float

    @staticmethod
    def init(mdp: MDP, key, gamma: float, vi_weight: float = 0.6):
        return BlendedAlgorithm.State(
            q_vals=jnp.zeros((mdp.num_actions, mdp.num_states)),
            gamma=jnp.array(gamma),
            vi_weight=vi_weight,
        )

    @staticmethod
    def update(state: "BlendedAlgorithm.State", mdp: MDP, step: int):
        # COMPOSITION 1: VI branch
        q_vi = bellman_optimality_operator.q(mdp, state.q_vals, state.gamma)

        # COMPOSITION 2: PI branch
        policy = greedy_policy.q(state.q_vals)
        q_pi = policy_evaluation.q(mdp, policy, state.gamma)

        # BLEND: Combine both branches
        q_blend = state.vi_weight * q_vi + (1 - state.vi_weight) * q_pi

        return state.replace(q_vals=q_blend)


def level_4_parallel_composition():
    """
    Compose primitives in parallel and blend results.
    """
    print("\n" + "="*70)
    print("LEVEL 4: Parallel Composition (Branch 1 ⊕ Branch 2)")
    print("="*70)

    # Setup
    key = jax.random.PRNGKey(42)
    mdp = garnet_mdp(key, num_states=20, num_actions=3, branch_factor=2)

    print("\nAlgorithm: Blended VI + PI")
    print("Composition:")
    print("  Branch 1: bellman_optimality (VI)")
    print("  Branch 2: greedy_policy → policy_evaluation (PI)")
    print("  Blend: 60% VI + 40% PI")

    # Run algorithm
    state = BlendedAlgorithm.init(mdp, key, gamma=0.95, vi_weight=0.6)

    print("\nRunning 10 iterations...")
    for step in range(10):
        state = BlendedAlgorithm.update(state, mdp, step)
        if step % 3 == 0:
            max_q = jnp.max(state.q_vals)
            print(f"  Step {step:2d}: max|Q| = {max_q:8.4f}")

    print("\n✓ Combined two composition branches with weighted blending")


# =============================================================================
# LEVEL 5: Multi-Stage Composition (Explore → Optimize → Exploit)
# =============================================================================

class MultiStageAlgorithm(metaclass=StaticMeta):
    """
    Compose multiple algorithms with conditional switching:
    - Early: Soft policy (explore)
    - Middle: Epsilon-greedy (balance)
    - Late: Greedy (exploit)
    """

    @struct.dataclass
    class State:
        q_vals: QType
        gamma: jnp.ndarray
        stage: int  # 0=explore, 1=balance, 2=exploit

    @staticmethod
    def init(mdp: MDP, key, gamma: float):
        return MultiStageAlgorithm.State(
            q_vals=jnp.zeros((mdp.num_actions, mdp.num_states)),
            gamma=jnp.array(gamma),
            stage=0,
        )

    @staticmethod
    def update(state: "MultiStageAlgorithm.State", mdp: MDP, step: int):
        # Determine stage
        if step < 10:
            stage = 0  # Explore
        elif step < 20:
            stage = 1  # Balance
        else:
            stage = 2  # Exploit

        # COMPOSITION: Different primitives per stage
        if stage == 0:
            # Stage 1: Soft policy (high temperature)
            policy = soft_policy.q(state.q_vals, temperature=3.0)
        elif stage == 1:
            # Stage 2: Epsilon-greedy
            policy = e_greedy_policy.q(state.q_vals, epsilon=0.2)
        else:
            # Stage 3: Greedy
            policy = greedy_policy.q(state.q_vals)

        # All stages use same evaluation
        q_vals = policy_evaluation.q(mdp, policy, state.gamma)

        return state.replace(q_vals=q_vals, stage=stage)


def level_5_multistage_composition():
    """
    Compose primitives with multi-stage conditional logic.
    """
    print("\n" + "="*70)
    print("LEVEL 5: Multi-Stage Composition (Stage 1 → Stage 2 → Stage 3)")
    print("="*70)

    # Setup
    key = jax.random.PRNGKey(42)
    mdp = garnet_mdp(key, num_states=20, num_actions=3, branch_factor=2)

    print("\nAlgorithm: Three-Stage Curriculum")
    print("Composition:")
    print("  Steps 0-9:   soft_policy(T=3.0) → policy_evaluation  (EXPLORE)")
    print("  Steps 10-19: e_greedy_policy(ε=0.2) → policy_eval    (BALANCE)")
    print("  Steps 20+:   greedy_policy → policy_evaluation       (EXPLOIT)")

    # Run algorithm
    state = MultiStageAlgorithm.init(mdp, key, gamma=0.95)

    stage_names = ["EXPLORE", "BALANCE", "EXPLOIT"]

    print("\nRunning 30 iterations...")
    for step in range(30):
        state = MultiStageAlgorithm.update(state, mdp, step)
        if step % 5 == 0:
            max_q = jnp.max(state.q_vals)
            stage_name = stage_names[state.stage]
            print(f"  Step {step:2d}: max|Q| = {max_q:8.4f}  [{stage_name}]")

    print("\n✓ Built curriculum by composing 3 different primitive combinations")


# =============================================================================
# LEVEL 6: Higher-Order Composition (Composition of Compositions)
# =============================================================================

def create_momentum_wrapper(base_algorithm_update, momentum: float = 0.9):
    """
    Higher-order function: Takes an algorithm and adds momentum.

    This composes ANY algorithm with momentum acceleration.
    """
    def momentum_update(state, mdp, step):
        # Save previous Q-values
        q_old = state.q_vals

        # Run base algorithm (which itself is a composition!)
        new_state = base_algorithm_update(state, mdp, step)

        # Add momentum if not first step
        if step > 0:
            q_momentum = new_state.q_vals + momentum * (new_state.q_vals - q_old)
            return new_state.replace(q_vals=q_momentum)

        return new_state

    return momentum_update


def level_6_higher_order_composition():
    """
    Compose compositions: Wrap existing algorithms with momentum.
    """
    print("\n" + "="*70)
    print("LEVEL 6: Higher-Order Composition (Wrapper ∘ Algorithm)")
    print("="*70)

    # Setup
    key = jax.random.PRNGKey(42)
    mdp = garnet_mdp(key, num_states=20, num_actions=3, branch_factor=2)

    print("\nTake existing algorithms and wrap them with momentum:")
    print("  SimplePolicyIteration + Momentum")
    print("  BlendedAlgorithm + Momentum")

    # Create momentum-wrapped versions
    pi_with_momentum = create_momentum_wrapper(
        SimplePolicyIteration.update,
        momentum=0.9
    )

    blended_with_momentum = create_momentum_wrapper(
        BlendedAlgorithm.update,
        momentum=0.95
    )

    # Test PI + Momentum
    print("\n1. Policy Iteration + Momentum (β=0.9):")
    state = SimplePolicyIteration.init(mdp, key, gamma=0.95)
    for step in range(10):
        state = pi_with_momentum(state, mdp, step)
        if step % 3 == 0:
            max_q = jnp.max(state.q_vals)
            print(f"     Step {step:2d}: max|Q| = {max_q:8.4f}")

    # Test Blended + Momentum
    print("\n2. Blended Algorithm + Momentum (β=0.95):")
    state = BlendedAlgorithm.init(mdp, key, gamma=0.95)
    for step in range(10):
        state = blended_with_momentum(state, mdp, step)
        if step % 3 == 0:
            max_q = jnp.max(state.q_vals)
            print(f"     Step {step:2d}: max|Q| = {max_q:8.4f}")

    print("\n✓ Wrapped two different algorithms with same momentum operator")
    print("  This is composing compositions - higher-order composition!")


# =============================================================================
# LEVEL 7: Full Compositional Pipeline
# =============================================================================

def level_7_full_pipeline():
    """
    Combine everything: show how compositions build on each other.
    """
    print("\n" + "="*70)
    print("LEVEL 7: Full Compositional Pipeline")
    print("="*70)

    print("\nComposition Hierarchy:")
    print()
    print("  PRIMITIVES:")
    print("    ├─ greedy_policy.q()")
    print("    ├─ soft_policy.q()")
    print("    ├─ e_greedy_policy.q()")
    print("    ├─ policy_evaluation.q()")
    print("    └─ bellman_optimality_operator.q()")
    print()
    print("  LEVEL 2 (Sequential):")
    print("    └─ SimplePolicyIteration = greedy → evaluation")
    print()
    print("  LEVEL 3 (Parametric):")
    print("    └─ SoftPolicyIteration = soft(T=f(t)) → evaluation")
    print()
    print("  LEVEL 4 (Parallel):")
    print("    └─ BlendedAlgorithm = (bellman ⊕ greedy→eval)")
    print()
    print("  LEVEL 5 (Multi-Stage):")
    print("    └─ MultiStageAlgorithm = soft → e_greedy → greedy")
    print()
    print("  LEVEL 6 (Higher-Order):")
    print("    ├─ PI + Momentum = momentum(SimplePolicyIteration)")
    print("    └─ Blended + Momentum = momentum(BlendedAlgorithm)")
    print()
    print("  LEVEL 7 (Full Pipeline):")
    print("    └─ Adaptive Multi-Stage + Momentum + Blending")
    print("       = momentum(multistage(blend(primitives)))")
    print()
    print("Each level builds on the previous one through composition!")


# =============================================================================
# Main Runner
# =============================================================================

def main():
    """Run all levels of composition examples."""

    print("\n" + "╔" + "="*68 + "╗")
    print("║" + "COMPOSABILITY DEMONSTRATION".center(68) + "║")
    print("║" + "Building Complex Algorithms from Simple Primitives".center(68) + "║")
    print("╚" + "="*68 + "╝")

    # Run each level
    level_1_primitives_only()
    level_2_sequential_composition()
    level_3_parametric_composition()
    level_4_parallel_composition()
    level_5_multistage_composition()
    level_6_higher_order_composition()
    level_7_full_pipeline()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: The Compositional Ladder")
    print("="*70)
    print()
    print("1. PRIMITIVES        → Basic building blocks")
    print("2. SEQUENTIAL        → Chain primitives (A → B)")
    print("3. PARAMETRIC        → Add schedules (A(t) → B)")
    print("4. PARALLEL          → Blend branches (A ⊕ B)")
    print("5. MULTI-STAGE       → Conditional switching (if stage: A else: B)")
    print("6. HIGHER-ORDER      → Wrap algorithms (F(Algorithm))")
    print("7. FULL PIPELINE     → Combine everything")
    print()
    print("Each level composes the previous level - like building with LEGO!")
    print("This is the power of compositional design.")
    print()


if __name__ == "__main__":
    main()
