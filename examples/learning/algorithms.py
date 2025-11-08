
import jax.numpy as jnp
import jax.random as jrd
from flax import struct

from jaxdp import async_sample_step_pi
from jaxdp.base import e_greedy_policy
from jaxdp.mdp import MDP
from jaxdp.typehints import QType, StaticMeta


@struct.dataclass
class Transition:
    """
    Protocol dataclass for MDP transitions.

    Contains all information about a single transition (s, a, r, s', done).
    """
    state: jnp.ndarray  # Current state (one-hot, shape: [n_states])
    action: jnp.ndarray  # Action taken (one-hot, shape: [n_actions])
    reward: jnp.ndarray  # Reward received
    next_state: jnp.ndarray  # Next state reached (one-hot, shape: [n_states])
    terminal: jnp.ndarray  # Terminal flag


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

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray,
             alpha: jnp.ndarray, init_q: jnp.ndarray = 0.0) -> "q_learning.State":
        q_vals = jnp.full((mdp.action_size, mdp.state_size), init_q)

        return q_learning.State(
            q_vals=q_vals,
            gamma=gamma,
            alpha=alpha
        )

    def update(state: "q_learning.State", transition: Transition) -> "q_learning.State":
        """
        Update Q-values based on a single transition.

        Args:
            state: Current algorithm state (Q-values and parameters)
            transition: Transition dataclass containing (s, a, r, s', done)
        """
        # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        # Use einsum for efficient computation
        # Q: [n_actions, n_states], action: [n_actions], state: [n_states]
        curr_q = jnp.einsum('as,a,s->', state.q_vals, transition.action, transition.state)

        # Compute Q-values for next state, then take max
        q_next = jnp.einsum('as,s->a', state.q_vals, transition.next_state)
        max_next_q = jnp.max(q_next)

        # TD target and error
        td_target = transition.reward + state.gamma * max_next_q * (1.0 - transition.terminal)
        td_error = td_target - curr_q

        # Update: Q += α * δ * (action ⊗ state)
        update = jnp.einsum('a,s->as', transition.action, transition.state)
        next_q = state.q_vals + state.alpha * td_error * update

        return state.replace(q_vals=next_q)
