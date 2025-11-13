from typing import Protocol

import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct

from jaxdp import async_sample_step_pi
from jaxdp.base import e_greedy_policy
from jaxdp.mdp import MDP
from jaxdp.typehints import F, QType, StaticMeta


class Transition(Protocol):
    """
    Protocol for MDP transition objects.

    Defines the interface that transition objects must implement.
    Actual implementation is sampler.Transition in benchmark.py.
    """

    state: F["S"]
    action: F["A"]
    reward: F[""]
    next_state: F["S"]
    terminal: F[""]


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

    def init(
        mdp: MDP, key: jrd.PRNGKey, gamma: float, alpha: float, init_q: float = 0.0
    ) -> "q_learning.State":
        q_vals = jnp.full((mdp.action_size, mdp.state_size), init_q)

        return q_learning.State(q_vals=q_vals, gamma=jnp.array(gamma), alpha=jnp.array(alpha))

    def _compute_delta(state: "q_learning.State", transition: Transition) -> F["A S"]:
        """
        Compute Q-value update delta for a single transition without alpha scaling.

        Args:
            state: Current algorithm state (Q-values and parameters)
            transition: Transition dataclass containing (s, a, r, s', done)

        Returns:
            Delta matrix before alpha multiplication [n_actions, n_states]
        """
        curr_q = jnp.einsum("as,a,s->", state.q_vals, transition.action, transition.state)

        q_next = jnp.einsum("as,s->a", state.q_vals, transition.next_state)
        max_next_q = jnp.max(q_next)

        td_target = transition.reward + state.gamma * max_next_q * (1.0 - transition.terminal)
        td_error = td_target - curr_q

        update = jnp.einsum("a,s->as", transition.action, transition.state)
        delta = td_error * update

        return delta

    def update(state: "q_learning.State", transition: Transition) -> "q_learning.State":
        """
        Update Q-values based on a single transition.

        Args:
            state: Current algorithm state (Q-values and parameters)
            transition: Transition dataclass containing (s, a, r, s', done)

        Returns:
            Updated algorithm state with new Q-values
        """
        delta = q_learning._compute_delta(state, transition)
        return state.replace(q_vals=state.q_vals + state.alpha * delta)

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
        vmap_delta = jax.vmap(lambda t: q_learning._compute_delta(state, t))
        deltas = vmap_delta(transitions)

        total_delta = jnp.sum(deltas, axis=0)

        total_counts = jnp.einsum("ba,bs->as", transitions.action, transitions.state)

        safe_counts = jnp.maximum(total_counts, 1.0)
        normalized_delta = total_delta / safe_counts

        return state.replace(q_vals=state.q_vals + state.alpha * normalized_delta)
