import chex
import distrax
import jax
import jax.numpy as jnp
import jax.random as jrd

from jaxdp.mdp.mdp import MDP
from jaxdp.typehints import F, PiType


def sample_from(policy: PiType, key: chex.PRNGKey) -> F["AS"]:
    r"""
    Sample from a policy. The samples will be one-hot vectors.

    Args:
        policy (PiType): Policy distribution
        key (chex.PRNGKey): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        Array: Sampled actions in the one-hot vector form for each state.

    """
    return distrax.OneHotCategorical(probs=policy.T, dtype=policy.dtype).sample(seed=key).T


def sample_based_policy_evaluation(
    mdp: MDP, policy: PiType, key: chex.PRNGKey, gamma: float, max_episode_length: int
) -> chex.Scalar:
    """
    Evaluate policy using sample-based Monte Carlo estimation.

    Args:
        mdp (MDP): Markov Decision Process
        policy (PiType): Policy distribution
        key (chex.PRNGKey): State of the JAX pseudorandom number generators (PRNGs)
        gamma (float): Discount factor
        max_episode_length (int): Maximum length of episode for sampling

    Returns:
        chex.Scalar: Estimated value of the policy

    """
    # TODO: Add test
    episode_step = jnp.zeros((1,))
    state = mdp.initial
    episode_rewards = jnp.full((max_episode_length,), jnp.nan)
    is_terminated = jnp.array(False).astype("bool")

    def _step(index, _data):
        _episode_step, _key, _episode_rewards, _state, _is_terminated = _data
        _key, step_key = jrd.split(_key)
        (act, next_state, reward, terminal, timeout, _state, _episode_step) = async_sample_step_pi(
            mdp, policy, _state, _episode_step, max_episode_length, step_key
        )
        reward = (1 - _is_terminated) * reward * (gamma**index)
        _is_terminated = jnp.logical_or(terminal, _is_terminated)
        _episode_rewards = _episode_rewards.at[index].set(reward)

        return _episode_step, _key, _episode_rewards, _state, _is_terminated

    _, _, episode_rewards, _, _ = jax.lax.fori_loop(
        0, max_episode_length, _step, (episode_step, key, episode_rewards, state, is_terminated)
    )
    return episode_rewards.sum()


def sync_sample(mdp: MDP, key: chex.PRNGKey) -> tuple[F["AS"], F["ASS"], F["AS"]]:
    r"""
    Synchronously sample starting from each state action pair in the given MDP

    Args:
        mdp (MDP): Markov Decision Process
        key (chex.PRNGKey): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        tuple[chex.Array, chex.Array, chex.Array]: Rewards, Next states, Termination condition

    """
    next_state = distrax.OneHotCategorical(
        probs=jnp.einsum("axs->asx", mdp.transition), dtype=mdp.transition.dtype
    ).sample(seed=key)
    terminal = jnp.einsum("asx,x->as", next_state, mdp.terminal)
    reward = jnp.einsum("asx,asx->as", mdp.reward, next_state)

    return reward, next_state, terminal


def async_sample_step(
    mdp: MDP,
    action: F["A"],
    state: F["S"],
    episode_step: jax.Array,
    episode_length: int,
    key: chex.PRNGKey,
) -> tuple[F["S"], jax.Array, jax.Array, jax.Array, F["S"], jax.Array]:
    r"""
    Asynchronously sample from the given MDP by following the given action. The starting state
    is given by the <state> argument. Similar to stateless version of the env.step() function
    from "gym", this function iterates the MDP only one step and returns the transition artifacts
    and the stepped MDP states (state & episode_length). The <episode_length> argument limits
    the maximum episode length (artificially). If an episode is terminated by reaching the maximum
    episodic length, this function sets <timeout> value to "True" while <terminal> may leave as
    "False".
        Note that: The terminated MDP state is automatically set to initial state.

    Args:
        mdp (MDP): Markov Decision Process
        action (Array): One hot action
        state (Array): Current state of the MDP
        episode_step (Scalar): Step count of the MDP
        episode_length (int): Maximum allowed episode length
        key (chex.PRNGKey): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        chex.Array: Next states of the transition (not necessarily equal to stepped State)
        chex.Scalar: Rewards of the transition
        chex.Scalar: termination condition (either 0 or 1) of the transition
        chex.Scalar: timeout condition (either 0 or 1) of the transition
        chex.Array: Stepped state
        chex.Scalar: Stepped step count

    """
    state_key, init_key = jrd.split(key, num=2)

    next_state_p = jnp.einsum("a,axs,s->x", action, mdp.transition, state)
    next_state = distrax.OneHotCategorical(
        probs=next_state_p,
        dtype=next_state_p.dtype,
    ).sample(seed=state_key)
    reward = jnp.einsum("asx,a,s,x->", mdp.reward, action, state, next_state)
    terminal = jnp.einsum("s,s->", mdp.terminal, next_state)

    episode_step = episode_step + 1
    timeout = episode_step >= episode_length
    terminal = jnp.einsum("s,s->", mdp.terminal, next_state)
    done = jnp.logical_or(terminal, timeout)

    init_state = mdp.init_state(init_key)
    state = next_state * (1 - done) + init_state * done
    episode_step = episode_step * (1 - done)

    return next_state, reward, terminal, timeout, state, episode_step


def async_sample_step_pi(
    mdp: MDP,
    policy: PiType,
    state: F["S"],
    episode_step: jax.Array,
    episode_length: int,
    key: chex.PRNGKey,
) -> tuple[F["A"], F["S"], jax.Array, jax.Array, jax.Array, F["S"], jax.Array]:
    r"""
    Asynchronously sample from the given MDP by following the given policy.

    Args:
        mdp (MDP): Markov Decision Process
        policy (PiType): Policy distribution
        state (Array): Current state of the MDP
        episode_step (Scalar): Step count of the MDP
        episode_length (int): Maximum allowed episode length
        key (chex.PRNGKey): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        chex.Array: Action of the transition
        chex.Array: Next states of the transition (not necessarily equal to stepped State)
        chex.Scalar: Rewards of the transition
        chex.Scalar: termination condition (either 0 or 1) of the transition
        chex.Scalar: timeout condition (either 0 or 1) of the transition
        chex.Array: Stepped state
        chex.Scalar: Stepped step count

    """
    act_key, step_key = jrd.split(key, num=2)
    policy_p = jnp.einsum("as,s->a", policy, state)
    action = sample_from(policy_p, key=act_key)

    return action, *async_sample_step(
        mdp=mdp,
        action=action,
        state=state,
        episode_step=episode_step,
        episode_length=episode_length,
        key=step_key,
    )


def sg(array: chex.Array) -> chex.Array:
    """Stop Gradient function"""
    return jax.lax.stop_gradient(array)
