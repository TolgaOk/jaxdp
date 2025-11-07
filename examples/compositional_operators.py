"""
Compositional DP Operator Language Examples

This module demonstrates how to build a compositional "language" for DP operations
by combining primitive operators, creating new ones, and defining domain-specific
abstractions.
"""

import jax
import jax.numpy as jnp
from flax import struct
from jaxdp import (
    greedy_policy,
    soft_policy,
    e_greedy_policy,
    policy_evaluation,
    bellman_operator,
    bellman_optimality_operator,
    stationary_distribution,
    expected_value,
    to_state_action_value,
)
from jaxdp.typehints import MDP, QType, VType, PiType, StaticMeta
from typing import Callable, Tuple


# =============================================================================
# 1. COMPOSITIONAL OPERATOR BUILDING
# =============================================================================

class OptimisticVI(metaclass=StaticMeta):
    """
    Optimistic Value Iteration - combines Bellman operator with optimism bonus.
    Demonstrates: Composing bellman_optimality_operator with custom logic.
    """

    @struct.dataclass
    class State:
        q_vals: QType
        gamma: jnp.ndarray
        optimism_bonus: float

    @staticmethod
    def init(mdp: MDP, key: jax.Array, gamma: float, optimism_bonus: float = 1.0):
        return OptimisticVI.State(
            q_vals=jnp.zeros((mdp.num_actions, mdp.num_states)),
            gamma=jnp.array(gamma),
            optimism_bonus=optimism_bonus,
        )

    @staticmethod
    def update(state: "OptimisticVI.State", mdp: MDP, step: int) -> "OptimisticVI.State":
        """Compose Bellman operator with optimism bonus"""
        # Primitive: Bellman optimality operator
        greedy_q = bellman_optimality_operator.q(mdp, state.q_vals, state.gamma)

        # Add optimism bonus (decays with iterations)
        bonus = state.optimism_bonus / jnp.sqrt(step + 1)
        optimistic_q = greedy_q + bonus

        return state.replace(q_vals=optimistic_q)


class TemperatureScheduledVI(metaclass=StaticMeta):
    """
    Softmax Value Iteration with temperature annealing.
    Demonstrates: Composing soft_policy with custom scheduling.
    """

    @struct.dataclass
    class State:
        q_vals: QType
        gamma: jnp.ndarray
        init_temperature: float
        final_temperature: float

    @staticmethod
    def init(mdp: MDP, key: jax.Array, gamma: float,
             init_temp: float = 10.0, final_temp: float = 0.1):
        return TemperatureScheduledVI.State(
            q_vals=jnp.zeros((mdp.num_actions, mdp.num_states)),
            gamma=jnp.array(gamma),
            init_temperature=init_temp,
            final_temperature=final_temp,
        )

    @staticmethod
    def update(state: "TemperatureScheduledVI.State", mdp: MDP,
               step: int, total_steps: int = 1000) -> "TemperatureScheduledVI.State":
        """Compose soft_policy with annealed temperature"""
        # Temperature annealing schedule
        alpha = step / total_steps
        temperature = (1 - alpha) * state.init_temperature + alpha * state.final_temperature

        # Primitive: Soft policy extraction
        soft_pi = soft_policy.q(state.q_vals, temperature)

        # Primitive: Policy evaluation with soft policy
        q_vals = policy_evaluation.q(mdp, soft_pi, state.gamma)

        return state.replace(q_vals=q_vals)


class MomentumPI(metaclass=StaticMeta):
    """
    Policy Iteration with momentum.
    Demonstrates: Composing greedy_policy + policy_evaluation with momentum.
    """

    @struct.dataclass
    class State:
        q_vals: QType
        prev_q: QType
        gamma: jnp.ndarray
        momentum: float

    @staticmethod
    def init(mdp: MDP, key: jax.Array, gamma: float, momentum: float = 0.9):
        init_q = jnp.zeros((mdp.num_actions, mdp.num_states))
        return MomentumPI.State(
            q_vals=init_q,
            prev_q=init_q,
            gamma=jnp.array(gamma),
            momentum=momentum,
        )

    @staticmethod
    def update(state: "MomentumPI.State", mdp: MDP, step: int) -> "MomentumPI.State":
        """Compose greedy extraction, evaluation, and momentum"""
        # Primitive: Extract greedy policy
        policy = greedy_policy.q(state.q_vals)

        # Primitive: Evaluate policy
        new_q = policy_evaluation.q(mdp, policy, state.gamma)

        # Add momentum term
        q_vals = new_q + state.momentum * (state.q_vals - state.prev_q)

        return state.replace(q_vals=q_vals, prev_q=state.q_vals)


# =============================================================================
# 2. HIGHER-ORDER OPERATOR COMBINATORS
# =============================================================================

class OperatorComposition(metaclass=StaticMeta):
    """
    Higher-order combinators for building operator pipelines.
    This is the "language" layer - combinators that compose primitives.
    """

    @staticmethod
    def alternate(op1_update: Callable, op2_update: Callable, period: int = 10):
        """
        Alternate between two operators every 'period' steps.

        Example: Alternate between VI and PI every 10 iterations.
        """
        def alternating_update(state, mdp, step):
            use_op1 = (step // period) % 2 == 0
            return jax.lax.cond(
                use_op1,
                lambda s: op1_update(s, mdp, step),
                lambda s: op2_update(s, mdp, step),
                state
            )
        return alternating_update

    @staticmethod
    def blend(op1_update: Callable, op2_update: Callable, alpha: float = 0.5):
        """
        Blend outputs of two operators with weight alpha.

        Example: Combine VI and PI updates with weighted average.
        """
        def blended_update(state, mdp, step):
            state1 = op1_update(state, mdp, step)
            state2 = op2_update(state, mdp, step)

            blended_q = alpha * state1.q_vals + (1 - alpha) * state2.q_vals
            return state1.replace(q_vals=blended_q)

        return blended_update

    @staticmethod
    def conditional(condition: Callable[[int], bool],
                   op_true: Callable,
                   op_false: Callable):
        """
        Choose operator based on runtime condition.

        Example: Use aggressive exploration early, exploitation later.
        """
        def conditional_update(state, mdp, step):
            return jax.lax.cond(
                condition(step),
                lambda s: op_true(s, mdp, step),
                lambda s: op_false(s, mdp, step),
                state
            )
        return conditional_update


# =============================================================================
# 3. POLICY COMPOSITION LANGUAGE
# =============================================================================

class PolicyCombinator(metaclass=StaticMeta):
    """
    Combinators for building composite policies from primitives.
    """

    @staticmethod
    def mixture(policies: list[PiType], weights: jnp.ndarray) -> PiType:
        """Mix multiple policies with given weights"""
        return sum(w * pi for w, pi in zip(weights, policies))

    @staticmethod
    def cascade(policies: list[PiType], thresholds: jnp.ndarray) -> PiType:
        """
        Cascade policy: use greedy if confidence > threshold, else soft.

        Example compositional policy from primitives.
        """
        def build_cascade(q_vals: QType) -> PiType:
            # Primitive 1: Greedy policy
            greedy_pi = greedy_policy.q(q_vals)

            # Primitive 2: Soft policy as fallback
            soft_pi = soft_policy.q(q_vals, temperature=1.0)

            # Confidence = max probability
            max_q = jnp.max(q_vals, axis=0, keepdims=True)
            confidence = jnp.exp(max_q) / jnp.sum(jnp.exp(q_vals), axis=0, keepdims=True)

            # Use greedy where confident, soft otherwise
            use_greedy = confidence > thresholds[..., None]
            return jnp.where(use_greedy, greedy_pi, soft_pi)

        return build_cascade

    @staticmethod
    def epsilon_boltzmann(q_vals: QType, epsilon: float, temperature: float) -> PiType:
        """
        Combine epsilon-greedy with Boltzmann exploration.
        Demonstrates: Building new policy from two primitives.
        """
        # Primitive 1: Epsilon-greedy
        eps_greedy = e_greedy_policy.q(q_vals, epsilon)

        # Primitive 2: Soft (Boltzmann)
        boltzmann = soft_policy.q(q_vals, temperature)

        # Weighted combination
        return 0.7 * eps_greedy + 0.3 * boltzmann


# =============================================================================
# 4. CUSTOM SAMPLING METHODS BUILT FROM PRIMITIVES
# =============================================================================

class CompositionalSampling(metaclass=StaticMeta):
    """
    Build new sampling methods by composing primitive sampling operations.
    """

    @staticmethod
    def multi_policy_sample(key: jax.Array,
                           mdp: MDP,
                           q_vals: QType,
                           state: int,
                           num_samples: int = 10) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample from multiple policy types and return ensemble statistics.

        Composes: greedy_policy, soft_policy, e_greedy_policy + sampling
        """
        from jaxdp import sample_from

        keys = jax.random.split(key, 3)

        # Sample from each primitive policy
        greedy_pi = greedy_policy.q(q_vals)
        soft_pi = soft_policy.q(q_vals, temperature=1.0)
        egreedy_pi = e_greedy_policy.q(q_vals, epsilon=0.1)

        # Vectorized sampling
        def sample_policy(k, pi):
            return sample_from(k, pi, state)

        greedy_actions = jax.vmap(sample_policy, in_axes=(0, None))(
            jax.random.split(keys[0], num_samples), greedy_pi
        )
        soft_actions = jax.vmap(sample_policy, in_axes=(0, None))(
            jax.random.split(keys[1], num_samples), soft_pi
        )
        egreedy_actions = jax.vmap(sample_policy, in_axes=(0, None))(
            jax.random.split(keys[2], num_samples), egreedy_pi
        )

        # Return action distributions
        all_actions = jnp.stack([greedy_actions, soft_actions, egreedy_actions])
        return all_actions, jnp.mean(all_actions, axis=1)

    @staticmethod
    def adaptive_rollout(key: jax.Array,
                        mdp: MDP,
                        q_vals: QType,
                        state: int,
                        horizon: int,
                        gamma: float) -> Tuple[jnp.ndarray, float]:
        """
        Adaptive rollout: switch policy based on value uncertainty.

        Composes: soft_policy (explore) + greedy_policy (exploit) + sync_sample
        """
        from jaxdp import sync_sample

        def policy_selector(s: int, step: int) -> PiType:
            # Early in rollout: explore with soft policy
            # Later: exploit with greedy
            exploration_phase = step < horizon // 2

            return jax.lax.cond(
                exploration_phase,
                lambda: soft_policy.q(q_vals, temperature=2.0),
                lambda: greedy_policy.q(q_vals),
            )

        # Manually implement adaptive sampling (simplified)
        def rollout_step(carry, step):
            state, total_reward, k = carry
            k, subkey = jax.random.split(k)

            # Get adaptive policy
            pi = policy_selector(state, step)

            # Sample action
            action = sample_from(subkey, pi, state)

            # Get next state and reward
            k, k_next = jax.random.split(k)
            next_state = jax.random.choice(
                k_next,
                mdp.num_states,
                p=mdp.transitions[action, state, :]
            )
            reward = mdp.rewards[action, state, next_state]

            discounted_reward = (gamma ** step) * reward

            return (next_state, total_reward + discounted_reward, k), None

        init_carry = (state, 0.0, key)
        (final_state, total_reward, _), _ = jax.lax.scan(
            rollout_step,
            init_carry,
            jnp.arange(horizon)
        )

        return final_state, total_reward


# =============================================================================
# 5. OPERATOR OVERLOADING EXTENSION (Optional Enhancement)
# =============================================================================

class DPOperator:
    """
    Wrapper class to enable operator overloading for DP operations.

    This adds a "language layer" with symbolic operators like +, *, |, etc.
    """

    def __init__(self, update_fn: Callable, state_type: type):
        self.update_fn = update_fn
        self.state_type = state_type

    def __add__(self, other: "DPOperator") -> "DPOperator":
        """+ operator: Blend two operators with equal weight"""
        return DPOperator(
            update_fn=OperatorComposition.blend(self.update_fn, other.update_fn, alpha=0.5),
            state_type=self.state_type
        )

    def __mul__(self, alpha: float) -> "DPOperator":
        """* operator: Scale operator influence"""
        original_update = self.update_fn

        def scaled_update(state, mdp, step):
            new_state = original_update(state, mdp, step)
            # Interpolate between old and new
            blended_q = (1 - alpha) * state.q_vals + alpha * new_state.q_vals
            return new_state.replace(q_vals=blended_q)

        return DPOperator(scaled_update, self.state_type)

    def __or__(self, other: "DPOperator") -> "DPOperator":
        """| operator: Alternate between operators"""
        return DPOperator(
            update_fn=OperatorComposition.alternate(self.update_fn, other.update_fn),
            state_type=self.state_type
        )

    def __rshift__(self, other: "DPOperator") -> "DPOperator":
        """>> operator: Sequential composition (pipeline)"""
        def sequential_update(state, mdp, step):
            intermediate = self.update_fn(state, mdp, step)
            return other.update_fn(intermediate, mdp, step)

        return DPOperator(sequential_update, self.state_type)

    def __call__(self, state, mdp, step):
        """Execute the operator"""
        return self.update_fn(state, mdp, step)


# =============================================================================
# 6. EXAMPLE USAGE: BUILDING ALGORITHMS WITH THE "LANGUAGE"
# =============================================================================

def example_compositional_algorithm():
    """
    Demonstrate building complex algorithms from primitive operators.
    """

    # Example 1: Create blended VI/PI algorithm using combinators
    from examples.algorithms import vi, pi

    blended_vi_pi = OperatorComposition.blend(
        vi.update,
        pi.update,
        alpha=0.7  # 70% VI, 30% PI
    )

    # Example 2: Adaptive algorithm - explore early, exploit late
    adaptive_algorithm = OperatorComposition.conditional(
        condition=lambda step: step < 50,  # First 50 iterations
        op_true=TemperatureScheduledVI.update,  # Explore
        op_false=vi.update  # Exploit
    )

    # Example 3: Using operator overloading DSL
    vi_op = DPOperator(vi.update, vi.State)
    pi_op = DPOperator(pi.update, pi.State)

    # Create complex operator with symbolic syntax
    hybrid_op = (vi_op * 0.8) + (pi_op * 0.2)  # 80% VI + 20% PI
    alternating_op = vi_op | pi_op  # Alternate between VI and PI
    pipeline_op = vi_op >> pi_op  # VI then PI (sequential)

    return {
        'blended': blended_vi_pi,
        'adaptive': adaptive_algorithm,
        'hybrid': hybrid_op,
        'alternating': alternating_op,
        'pipeline': pipeline_op,
    }


# =============================================================================
# 7. DOMAIN-SPECIFIC LANGUAGE EXAMPLE
# =============================================================================

class DPDSL:
    """
    Domain-Specific Language for DP algorithm construction.

    Provides fluent API for building algorithms compositionally.
    """

    def __init__(self, mdp: MDP):
        self.mdp = mdp
        self.operators = []

    def with_exploration(self, temperature: float = 1.0):
        """Add soft policy exploration phase"""
        self.operators.append(('explore', temperature))
        return self

    def with_exploitation(self):
        """Add greedy exploitation phase"""
        self.operators.append(('exploit', None))
        return self

    def with_optimism(self, bonus: float = 1.0):
        """Add optimism bonus"""
        self.operators.append(('optimism', bonus))
        return self

    def with_momentum(self, beta: float = 0.9):
        """Add momentum acceleration"""
        self.operators.append(('momentum', beta))
        return self

    def build(self) -> DPOperator:
        """Build the final composed operator"""
        # Translate DSL to actual operator composition
        # This is a simplified example - you'd implement full logic

        def composed_update(state, mdp, step):
            q = state.q_vals

            for op_type, param in self.operators:
                if op_type == 'explore':
                    pi = soft_policy.q(q, temperature=param)
                    q = policy_evaluation.q(mdp, pi, state.gamma)
                elif op_type == 'exploit':
                    q = bellman_optimality_operator.q(mdp, q, state.gamma)
                elif op_type == 'optimism':
                    q = q + param / jnp.sqrt(step + 1)
                # ... handle other operators

            return state.replace(q_vals=q)

        return DPOperator(composed_update, None)


def example_dsl_usage(mdp: MDP):
    """
    Example: Building algorithm with fluent DSL
    """

    # Build custom algorithm: Explore → Optimize → Exploit
    my_algorithm = (
        DPDSL(mdp)
        .with_exploration(temperature=2.0)
        .with_optimism(bonus=1.0)
        .with_momentum(beta=0.9)
        .with_exploitation()
        .build()
    )

    return my_algorithm
