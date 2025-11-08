
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
        state: jnp.ndarray
        ep_step: jnp.ndarray
        ep_return: jnp.ndarray
        last_return: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray,
             alpha: jnp.ndarray, epsilon: jnp.ndarray) -> "q_learning.State":
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
        init_state = mdp.init_state(key)

        return q_learning.State(
            q_vals=q_vals,
            gamma=gamma,
            alpha=alpha,
            epsilon=epsilon,
            state=init_state,
            ep_step=jnp.array(0.0),
            ep_return=jnp.array(0.0),
            last_return=jnp.array(0.0)
        )

    def update(state: "q_learning.State", mdp: MDP, step: int,
               max_ep_len: int, key: jrd.PRNGKey) -> "q_learning.State":
        policy = e_greedy_policy.q(state.q_vals, state.epsilon)

        action, next_s, reward, term, timeout, s, ep_step = async_sample_step_pi(
            mdp, policy, state.state, state.ep_step, max_ep_len, key
        )

        # Q-learning update
        curr_q = jnp.sum(state.q_vals * action[:, None] * s[None, :])
        max_next_q = jnp.max(jnp.sum(state.q_vals * next_s[None, :], axis=1))
        td_target = reward + state.gamma * max_next_q * (1.0 - term)
        td_error = td_target - curr_q
        next_q = state.q_vals + state.alpha * td_error * action[:, None] * s[None, :]

        # Track episode return
        new_return = state.ep_return + reward
        done = term + timeout > 0

        last_return = jnp.where(done, new_return, state.last_return)
        ep_return = jnp.where(done, 0.0, new_return)

        return state.replace(
            q_vals=next_q,
            state=s,
            ep_step=ep_step,
            ep_return=ep_return,
            last_return=last_return
        )
