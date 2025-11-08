
import jax.numpy as jnp
import jax.random as jrd
from flax import struct

from jaxdp import async_sample_step_pi
from jaxdp.base import e_greedy_policy
from jaxdp.mdp import MDP
from jaxdp.typehints import QType, StaticMeta


class q_learning(metaclass=StaticMeta):
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning: Off-policy TD Control

    Update rule: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    @struct.dataclass
    class State:
        q_vals: QType
        gamma: jnp.ndarray
        alpha: jnp.ndarray
        epsilon: jnp.ndarray
        eps_decay: jnp.ndarray
        eps_min: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray,
             alpha: jnp.ndarray, epsilon: jnp.ndarray,
             eps_decay: jnp.ndarray = 0.995, eps_min: jnp.ndarray = 0.01,
             init_q: jnp.ndarray = 0.0) -> "q_learning.State":
        q_vals = jnp.full((mdp.action_size, mdp.state_size), init_q)

        return q_learning.State(
            q_vals=q_vals,
            gamma=gamma,
            alpha=alpha,
            epsilon=epsilon,
            eps_decay=eps_decay,
            eps_min=eps_min
        )

    def update(alg_state: "q_learning.State", mdp_state: jnp.ndarray,
               action: jnp.ndarray, next_s: jnp.ndarray, reward: jnp.ndarray,
               term: jnp.ndarray, done: jnp.ndarray) -> "q_learning.State":
        """
        Update Q-values based on a single transition.

        Args:
            alg_state: Current algorithm state (Q-values and parameters)
            mdp_state: State where action was taken
            action: Action taken (one-hot vector)
            next_s: Next state reached
            reward: Reward received
            term: Terminal flag
            done: Episode done flag (terminal or timeout)
        """
        # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        curr_q = jnp.sum(alg_state.q_vals * action[:, None] * mdp_state[None, :])
        max_next_q = jnp.max(jnp.sum(alg_state.q_vals * next_s[None, :], axis=1))
        td_target = reward + alg_state.gamma * max_next_q * (1.0 - term)
        td_error = td_target - curr_q
        next_q = alg_state.q_vals + alg_state.alpha * td_error * action[:, None] * mdp_state[None, :]

        # Decay epsilon after each episode
        new_epsilon = jnp.maximum(alg_state.epsilon * alg_state.eps_decay, alg_state.eps_min)
        epsilon = jnp.where(done, new_epsilon, alg_state.epsilon)

        return alg_state.replace(q_vals=next_q, epsilon=epsilon)
