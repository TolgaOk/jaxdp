
import jax.numpy as jnp
import jax.random as jrd
from flax import struct

from jaxdp import async_sample_step_pi
from jaxdp.base import e_greedy_policy
from jaxdp.mdp import MDP
from jaxdp.mdp.sampler.mdp import State as SamplerState
from jaxdp.mdp.sampler.mdp import init_sampler_state
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
        last_state: jnp.ndarray
        episode_step: jnp.ndarray
        current_episode_reward: jnp.ndarray  # Accumulates reward for current episode
        last_episode_reward: jnp.ndarray  # Stores reward from last completed episode
        episode_count: jnp.ndarray  # Total number of completed episodes

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray,
             alpha: jnp.ndarray, epsilon: jnp.ndarray) -> "q_learning.State":
        """Initialize Q-learning state."""
        # Initialize Q-values to zero
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))

        # Get initial state
        init_state = mdp.init_state(key)

        return q_learning.State(
            q_vals=q_vals,
            gamma=gamma,
            alpha=alpha,
            epsilon=epsilon,
            last_state=init_state,
            episode_step=jnp.array(0.0),
            current_episode_reward=jnp.array(0.0),
            last_episode_reward=jnp.array(0.0),
            episode_count=jnp.array(0.0)
        )

    def update(state: "q_learning.State", mdp: MDP, step: int,
               max_episode_len: int = 1000,
               key: jrd.PRNGKey = None) -> "q_learning.State":
        """Perform one Q-learning update step."""
        # Generate epsilon-greedy policy
        policy = e_greedy_policy.q(state.q_vals, state.epsilon)

        # Sample one transition
        action, next_state_vec, reward, terminal, timeout, state_vec, episode_step = \
            async_sample_step_pi(
                mdp, policy,
                state.last_state,
                state.episode_step,
                max_episode_len,
                key
            )

        # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        current_q = jnp.sum(state.q_vals * action[:, None] * state_vec[None, :])
        max_next_q = jnp.max(jnp.sum(state.q_vals * next_state_vec[None, :], axis=1))
        td_target = reward + state.gamma * max_next_q * (1.0 - terminal)
        td_error = td_target - current_q
        q_update = state.alpha * td_error * action[:, None] * state_vec[None, :]
        next_q = state.q_vals + q_update

        # Track episodic rewards
        new_episode_reward = state.current_episode_reward + reward
        episode_ended = terminal + timeout  # Episode ends on terminal or timeout

        # If episode ended, store the total reward and reset counter, increment episode count
        last_ep_reward = jnp.where(episode_ended > 0, new_episode_reward, state.last_episode_reward)
        current_ep_reward = jnp.where(episode_ended > 0, 0.0, new_episode_reward)
        new_episode_count = jnp.where(episode_ended > 0, state.episode_count + 1, state.episode_count)

        return state.replace(
            q_vals=next_q,
            last_state=state_vec,  # Use state_vec (properly reset) not next_state_vec!
            episode_step=episode_step,
            current_episode_reward=current_ep_reward,
            last_episode_reward=last_ep_reward,
            episode_count=new_episode_count
        )
