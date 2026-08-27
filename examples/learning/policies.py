import jax
import jax.numpy as jnp
from flax import struct


class epsilon_greedy:
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Epsilon-Greedy Exploration Policy

    Selects random action with probability epsilon, greedy action otherwise.
    Epsilon decays over time: epsilon = max(epsilon * decay, min)
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    class State(struct.PyTreeNode):
        epsilon: jax.Array  # Exploration rate (scalar)
        eps_decay: jax.Array  # Decay factor (scalar)
        eps_min: jax.Array  # Minimum epsilon (scalar)

    @staticmethod
    def init(
        epsilon: float = 1.0, eps_decay: float = 0.997, eps_min: float = 0.1
    ) -> "epsilon_greedy.State":
        return epsilon_greedy.State(
            epsilon=jnp.array(epsilon), eps_decay=jnp.array(eps_decay), eps_min=jnp.array(eps_min)
        )

    @staticmethod
    def update(state: "epsilon_greedy.State", done: jax.Array) -> "epsilon_greedy.State":
        """Decay epsilon after each episode"""
        new_epsilon = jnp.maximum(state.epsilon * state.eps_decay, state.eps_min)
        epsilon = jnp.where(done, new_epsilon, state.epsilon)
        return state.replace(epsilon=epsilon)

    @staticmethod
    def get_policy(q_vals: jax.Array, state: "epsilon_greedy.State") -> jax.Array:
        """Get epsilon-greedy policy from Q-values"""
        greedy = jax.nn.one_hot(jnp.argmax(q_vals, axis=0), q_vals.shape[0], axis=0)
        return (1 - state.epsilon) * greedy + state.epsilon / q_vals.shape[0]


class soft_policy:
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Soft Policy (Boltzmann Exploration)

    Selects actions with probability proportional to exp(Q(s,a)/temperature).
    Temperature decays over time: temp = max(temp * decay, min)
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    class State(struct.PyTreeNode):
        temperature: jax.Array  # Temperature parameter (scalar)
        temp_decay: jax.Array  # Decay factor (scalar)
        temp_min: jax.Array  # Minimum temperature (scalar)

    @staticmethod
    def init(
        temperature: float = 1.0, temp_decay: float = 0.995, temp_min: float = 0.01
    ) -> "soft_policy.State":
        return soft_policy.State(
            temperature=jnp.array(temperature),
            temp_decay=jnp.array(temp_decay),
            temp_min=jnp.array(temp_min),
        )

    @staticmethod
    def update(state: "soft_policy.State", done: jax.Array) -> "soft_policy.State":
        """Decay temperature after each episode"""
        new_temp = jnp.maximum(state.temperature * state.temp_decay, state.temp_min)
        temperature = jnp.where(done, new_temp, state.temperature)
        return state.replace(temperature=temperature)

    @staticmethod
    def get_policy(q_vals: jax.Array, state: "soft_policy.State") -> jax.Array:
        """Get soft policy from Q-values"""
        scaled_q = q_vals / state.temperature
        exp_q = jnp.exp(scaled_q - jnp.max(scaled_q, axis=0, keepdims=True))
        policy = exp_q / jnp.sum(exp_q, axis=0, keepdims=True)
        return policy
