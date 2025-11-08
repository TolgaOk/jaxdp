
import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct

from jaxdp import async_sample_step_pi
from jaxdp.base import e_greedy_policy
from jaxdp.mdp import MDP
from jaxdp.typehints import F, QType, StaticMeta


@struct.dataclass
class Transition:
    """
    Protocol dataclass for MDP transitions.

    Contains all information about a single transition (s, a, r, s', done).
    """
    state: F["S"]  # Current state (one-hot)
    action: F["A"]  # Action taken (one-hot)
    reward: F[""]  # Reward received (scalar)
    next_state: F["S"]  # Next state reached (one-hot)
    terminal: F[""]  # Terminal flag (scalar)


class q_learning(metaclass=StaticMeta):
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning: Off-policy TD Control

    Update rule: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    @struct.dataclass
    class State:
        q_vals: QType  # Q-values [A, S]
        gamma: F[""]  # Discount factor (scalar)
        alpha: F[""]  # Learning rate (scalar)

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: float,
             alpha: float, init_q: float = 0.0) -> "q_learning.State":
        q_vals = jnp.full((mdp.action_size, mdp.state_size), init_q)

        return q_learning.State(
            q_vals=q_vals,
            gamma=jnp.array(gamma),
            alpha=jnp.array(alpha)
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

    def batch_update(state: "q_learning.State", transitions: Transition) -> "q_learning.State":
        """
        Update Q-values based on a batch of transitions.

        Properly handles repeated state-action pairs by dividing the summed updates
        by the number of occurrences of each (s,a) pair. This ensures that frequently
        visited (s,a) pairs don't get disproportionately large updates.

        Args:
            state: Current algorithm state (Q-values and parameters)
            transitions: Batch of Transition objects (fields have batch dimension)

        Returns:
            Updated algorithm state with properly normalized updates
        """
        # Vmap the regular update to compute all individual updates
        vmap_update = jax.vmap(lambda t: q_learning.update(state, t))
        updated_states = vmap_update(transitions)

        # Compute deltas: change in Q-values for each transition
        # Shape: [batch, n_actions, n_states]
        deltas = updated_states.q_vals - state.q_vals

        # Sum all deltas across the batch
        total_delta = jnp.sum(deltas, axis=0)  # Shape: [n_actions, n_states]

        # Count how many times each (s,a) pair appears in the batch
        # Use einsum: sum over batch dimension to get occurrence counts
        total_counts = jnp.einsum('ba,bs->as', transitions.action, transitions.state)

        # Divide summed updates by occurrence count for each (s,a)
        # Avoid division by zero (where count=0, delta should also be 0)
        safe_counts = jnp.maximum(total_counts, 1.0)
        normalized_delta = total_delta / safe_counts

        # Apply the normalized updates
        new_q_vals = state.q_vals + normalized_delta

        return state.replace(q_vals=new_q_vals)
