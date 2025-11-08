
import jax.numpy as jnp
import jax.random as jrd
from flax import struct

from jaxdp.base import e_greedy_policy
from jaxdp.typehints import QType, StaticMeta


class epsilon_greedy(metaclass=StaticMeta):
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Epsilon-Greedy Exploration Policy

    Selects random action with probability epsilon, greedy action otherwise.
    Epsilon decays over time: epsilon = max(epsilon * decay, min)
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    @struct.dataclass
    class State:
        epsilon: jnp.ndarray
        eps_decay: jnp.ndarray
        eps_min: jnp.ndarray

    def init(epsilon: float = 1.0, eps_decay: float = 0.997,
             eps_min: float = 0.1) -> "epsilon_greedy.State":
        return epsilon_greedy.State(
            epsilon=jnp.array(epsilon),
            eps_decay=jnp.array(eps_decay),
            eps_min=jnp.array(eps_min)
        )

    def update(state: "epsilon_greedy.State", done: jnp.ndarray) -> "epsilon_greedy.State":
        """Decay epsilon after each episode"""
        new_epsilon = jnp.maximum(state.epsilon * state.eps_decay, state.eps_min)
        epsilon = jnp.where(done, new_epsilon, state.epsilon)
        return state.replace(epsilon=epsilon)

    def get_policy(q_vals: QType, state: "epsilon_greedy.State"):
        """Get epsilon-greedy policy from Q-values"""
        return e_greedy_policy.q(q_vals, state.epsilon)


class softmax(metaclass=StaticMeta):
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Softmax (Boltzmann) Exploration Policy

    Selects actions with probability proportional to exp(Q(s,a)/temperature).
    Temperature decays over time: temp = max(temp * decay, min)
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    @struct.dataclass
    class State:
        temperature: jnp.ndarray
        temp_decay: jnp.ndarray
        temp_min: jnp.ndarray

    def init(temperature: float = 1.0, temp_decay: float = 0.995,
             temp_min: float = 0.01) -> "softmax.State":
        return softmax.State(
            temperature=jnp.array(temperature),
            temp_decay=jnp.array(temp_decay),
            temp_min=jnp.array(temp_min)
        )

    def update(state: "softmax.State", done: jnp.ndarray) -> "softmax.State":
        """Decay temperature after each episode"""
        new_temp = jnp.maximum(state.temperature * state.temp_decay, state.temp_min)
        temperature = jnp.where(done, new_temp, state.temperature)
        return state.replace(temperature=temperature)

    def get_policy(q_vals: QType, state: "softmax.State"):
        """Get softmax policy from Q-values"""
        # Compute softmax probabilities: exp(Q/T) / sum(exp(Q/T))
        # Q-values shape: (n_actions, n_states)
        # For each state, compute softmax over actions
        scaled_q = q_vals / state.temperature
        exp_q = jnp.exp(scaled_q - jnp.max(scaled_q, axis=0, keepdims=True))  # Numerical stability
        policy = exp_q / jnp.sum(exp_q, axis=0, keepdims=True)
        return policy
