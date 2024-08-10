from atexit import register
from typing import Optional, Callable, Tuple, Dict, Union, Any
import jax.numpy as jnp
import jax.random as jrd
from jax.typing import ArrayLike
import jax
import distrax
from jaxtyping import Array, Float
from jax.typing import ArrayLike

from jaxdp.mdp.mdp import MDP
from jaxdp.typehints import QType, VType, PiType, F
from jaxdp.utils import register_as


@register_as("q")
def greedy_policy(value: QType) -> PiType:
    r"""
    Greedy policy distribution from Q values.

    Args:
        value (QType): Q Value array

    Returns:
        PiType: Policy distribution as one hot vectors

    """
    return jax.nn.one_hot(jnp.argmax(value, axis=0, keepdims=False),
                          num_classes=value.shape[0],
                          axis=0)


@register_as("v")
def greedy_policy(mdp: MDP, value: VType, gamma: float) -> PiType:
    # TODO: Add docstring
    # TODO: Add test
    return greedy_policy.q(to_state_action_value(mdp, value, gamma))


@register_as("q")
def soft_policy(value: QType, temperature: float) -> PiType:
    r"""
    Softmax policy distribution.

    Args:
        value (QType): Q Value array
        temperature (float): Temperature parameter of softmax. Lower values will result in
            uniform policy distribution while higher values will result a distribution closed
            to greedy policy.

    Returns:
        PiType: Policy distribution

    """
    return jax.nn.softmax(value * temperature, axis=0)


@register_as("v")
def soft_policy(value: VType, temperature: float) -> PiType:
    raise NotImplementedError


@register_as("q")
def e_greedy_policy(value: QType, epsilon: float, ) -> PiType:
    r"""
    Epsilon greedy policy distribution.

    Args:
        value (QType): Q Value array
        epsilon (float): Epsilon parameter. The policy takes the greedy action with 
            (1 - epsilon + epsilon/|A|) probability while take a non-greedy action with 
            (epsilon / |A|) probability where |A| is the dimension of the action space.

    Returns:
        PiType: Policy distribution

    """
    greedy_policy.p = greedy_policy.q(value)
    return greedy_policy.p * (1 - epsilon) + jnp.ones_like(value) * (epsilon / value.shape[0])


@register_as("v")
def e_greedy_policy(value: VType, epsilon: float, ) -> PiType:
    raise NotImplementedError


@register_as("q")
def sample_from(policy: QType, key: ArrayLike) -> F["A S"]:
    r"""
    Sample from a policy. The samples will be one-hot vectors.

    Args:
        policy (PiType): Policy distribution
        key (ArrayLike): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        F["A S"]: Sampled actions in the one-hot vector form for each state.

    """
    return distrax.OneHotCategorical(probs=policy.T, dtype=jnp.float32).sample(seed=key).T


@register_as("v")
def expected_value(mdp: MDP, value: VType) -> F[""]:
    r"""
    Expected value of the state-values over initial distribution.

    .. math::
        \underset{s_0 \sim \rho}{\mathbb{E}} [V(S)]

    Args:
        mdp (MDP): Markov Decision Process
        value (F[S"]): Value array

    Returns:
        F[""]: Expected value

    """
    return (value * mdp.initial).sum()


@register_as("q")
def expected_value(mdp: MDP, value: QType) -> F[""]:
    r"""
    Expected value of the state-action values (Q) over initial distribution.

    .. math::
        \underset{\substack{s_0 \sim \rho \\ a \sim \pi^Q}}{\mathbb{E}} [Q(S, A)]

    Args:
        mdp (MDP): Markov Decision Process
        value (QType): Q Value array

    Returns:
        F[""]: Expected value

    """
    return expected_value.v(mdp, jnp.max(value, axis=0))


@register_as("v")
def policy_evaluation(mdp: MDP, policy: PiType, gamma: float) -> VType:
    r"""
    Evaluate the policy for each state using the true MDP

    .. math::
        \eta(\pi)(s_i) = \big[(\mathrm{I}_n - \gamma P^\pi)r^\pi\big]_i

    Args:
        mdp (MDP): Markov Decision Process
        policy (PiType): Policy distribution
        gamma (float): Discount factor

    Returns:
        VType: Cumulative discounted reward of each state

    """
    transition_pi, reward_pi = _markov_chain_pi(mdp, policy)

    return (jnp.linalg.inv(jnp.eye(mdp.state_size) - gamma * transition_pi.T) @
            jnp.einsum("sx,sx->s", transition_pi.T, reward_pi))


@register_as("q")
def policy_evaluation(mdp: MDP, policy: PiType, gamma: float) -> QType:
    r"""
    Evaluate the policy for each state-action pair using the true MDP

    .. math::
        \eta(\pi)(s_i, a_j) = \big[r^{a_j} + \gamma P^\pi (\mathrm{I}_n - \gamma P^\pi)r^\pi\big]_i

    Args:
        mdp (MDP): Markov Decision Process
        policy (PiType): Policy distribution
        gamma (float): Discount factor

    Returns:
        QType: Cumulative discounted reward of each state-action pair

    """
    mc_state_values = policy_evaluation.v(mdp, policy, gamma)
    reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
    return (reward + gamma * jnp.einsum("axs,x->as", mdp.transition, mc_state_values))


@register_as("v")
def bellman_operator(mdp: MDP, policy: PiType, value: VType, gamma: float) -> VType:
    r"""
    Evaluate the Bellman policy operator for each state

    .. math::
        \mathcal{B}(V)(s_i) = \big[r^{a_j} + \gamma 
            \underset{s^+ \sim P^{a_j}}{\mathbb{E}}[Q(s^+, a_j)]\big]_i - Q(s_i, a_j))

    Args:
        mdp (MDP): Markov Decision Process
        policy (PiType): Policy distribution
        value (Float[Array,"S"]): State value array
        gamma (float): Discount factor

    Returns:
        VType: Target value

    """
    # TODO: Update docstring
    # TODO: Add test
    target_values = jnp.einsum("axs,x,x->as",
                               mdp.transition, value, (1 - mdp.terminal))
    reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
    return jnp.einsum("as,as->s", reward + gamma * target_values, policy)


@register_as("q")
def bellman_operator(mdp: MDP, policy: PiType, value: QType, gamma: float) -> QType:
    r"""
    Evaluate the Bellman policy operator for each state-action pair

    .. math::
        \mathcal{B}(Q)(s_i, a_j) = \big[r^{a_j} + \gamma 
            \underset{s^+ \sim P^{a_j}}{\mathbb{E}}[Q(s^+, a_j)]\big]_i - Q(s_i, a_j))

    Args:
        mdp (MDP): Markov Decision Process
        policy (PiType): Policy distribution
        value (QType): Q Value array
        gamma (float): Discount factor

    Returns:
        QType: Target value

    """
    target_values = jnp.einsum("axs,ux,ux,x->as",
                               mdp.transition, value, policy, (1 - mdp.terminal))
    reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
    return reward + gamma * target_values


@register_as("q")
def bellman_optimality_operator(mdp: MDP, value: QType, gamma: float) -> QType:
    # TODO: Update docstring
    # TODO: Add test
    target_values = jnp.einsum("axs,x->as", mdp.transition, jnp.max(value, axis=0, keepdims=False))
    rewards = jnp.einsum("axs,asx->as", mdp.transition, mdp.reward)
    return rewards + gamma * target_values


def to_greedy_state_value(value: QType) -> VType:
    # TODO: Add docstring
    # TODO: Add test
    return jnp.max(value, axis=0)


def to_state_action_value(mdp: MDP, value: VType, gamma: float) -> QType:
    # TODO: Add docstring
    # TODO: Add test
    return (jnp.einsum("asx,axs->as", mdp.reward, mdp.transition) +
            gamma * jnp.einsum("axs,x->as", mdp.transition, value))


def _markov_chain_pi(mdp: MDP, policy: PiType) -> Tuple[F["S S"], F["S S"]]:
    r"""
    Make Markov Chain of an MDP by fixing the policy.

    .. math::
        P^\pi = \underset{a \sim \pi}{\mathbb{E}}[P^a]
        r^\pi = \underset{a \sim \pi}{\mathbb{E}}[r^a]

    Args:
        mdp (MDP): Markov Decision Process
        policy (PiType): Policy distribution

    Returns:
        F["S S"]: Transition matrix
        F["S S"]: Reward matrix

    """
    transition_pi = jnp.einsum("as,axs->xs", policy, mdp.transition)
    reward_pi = jnp.einsum("as,asx->sx", policy, mdp.reward)
    return transition_pi, reward_pi


def sample_based_policy_evaluation(mdp: MDP,
                                   policy: PiType,
                                   key: ArrayLike,
                                   gamma: float,
                                   max_episode_length: int
                                   ) -> F[""]:
    # TODO: Add test
    # TODO: Add docstring
    episode_step = jnp.zeros((1,))
    state = mdp.initial
    episode_rewards = jnp.full((max_episode_length,), jnp.nan)
    is_terminated = jnp.array(False).astype("bool")

    def _step(index, _data):
        _episode_step, _key, _episode_rewards, _state, _is_terminated = _data
        _key, step_key = jrd.split(_key)
        (act, next_state, reward, terminal, timeout, _state, _episode_step
         ) = async_sample_step_pi(
            mdp, policy, _state, _episode_step, max_episode_length, step_key)
        reward = (1 - _is_terminated) * reward * (gamma ** index)
        _is_terminated = jnp.logical_or(terminal, _is_terminated)
        _episode_rewards = _episode_rewards.at[index].set(reward)

        return _episode_step, _key, _episode_rewards, _state, _is_terminated

    _, _, episode_rewards, _, _ = jax.lax.fori_loop(0, max_episode_length, _step,
                                                    (episode_step, key, episode_rewards, state, is_terminated))
    return episode_rewards.sum()


def sync_sample(mdp: MDP, key: ArrayLike) -> Tuple[F["A S"], F["A S S"], F["A S"]]:
    r"""
    Synchronously sample starting from each state action pair in the given MDP

    Args:
        mdp (MDP): Markov Decision Process
        key (ArrayLike): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        VType: Rewards
        F["S S"]: Next states
        VType]: Termination condition (either 0 or 1)

    """
    next_state = distrax.OneHotCategorical(
        probs=jnp.einsum("axs->asx", mdp.transition), dtype=jnp.float32).sample(seed=key)
    terminal = jnp.einsum("asx,x->as", next_state, mdp.terminal)
    reward = jnp.einsum("asx,asx->as", mdp.reward, next_state)

    return reward, next_state, terminal


def async_sample_step(mdp: MDP,
                      action: F["A"],
                      state: F["S"],
                      episode_step: F[""],
                      episode_length: int,
                      key: ArrayLike
                      ) -> Tuple[F["S"], F[""], F[""], F[""], F["S"], F[""]]:
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
        action (F["A"]): One hot action
        state (F["S"]): Current state of the MDP
        episode_step (F[""]): Step count of the MDP
        episode_length (int): Maximum allowed episode length
        key (ArrayLike): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        VType: Next states of the transition (not necessarily equal to stepped State)
        F[""]: Rewards of the transition
        F[""]]: termination condition (either 0 or 1) of the transition
        F[""]]: timeout condition (either 0 or 1) of the transition
        VType: Stepped state
        F[""]]: Stepped step count

    """
    state_key, init_key = jrd.split(key, num=2)

    next_state_p = jnp.einsum(
        "a,axs,s->x", action, mdp.transition, state)
    next_state = distrax.OneHotCategorical(
        probs=next_state_p, dtype=jnp.float32).sample(seed=state_key)
    reward = jnp.einsum("asx,a,s,x->", mdp.reward, action, state, next_state)
    terminal = jnp.einsum("s,s->", mdp.terminal, next_state)

    episode_step = episode_step + 1
    timeout = episode_step >= episode_length
    terminal = jnp.einsum("s,s->", mdp.terminal, next_state)
    done = jnp.logical_or(terminal, timeout)

    init_state = distrax.OneHotCategorical(
        probs=mdp.initial, dtype=jnp.float32).sample(seed=init_key)
    state = next_state * (1 - done) + init_state * done
    episode_step = episode_step * (1 - done)

    return next_state, reward, terminal, timeout, state, episode_step


def async_sample_step_pi(mdp: MDP,
                         policy: PiType,
                         state: F["S"],
                         episode_step: F[""],
                         episode_length: int,
                         key: ArrayLike
                         ) -> Tuple[F["S"], F["A"], F[""], F[""], F[""], F["S"], F[""]]:
    r"""
    Asynchronously sample from the given MDP by following the given policy.

    Args:
        mdp (MDP): Markov Decision Process
        policy (PiType): Policy distribution
        state (F["S"]): Current state of the MDP
        episode_step (F[""]): Step count of the MDP
        episode_length (int): Maximum allowed episode length
        key (ArrayLike): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        F["A"]: Action of the transition
        VType: Next states of the transition (not necessarily equal to stepped State)
        F[""]: Rewards of the transition
        F[""]]: termination condition (either 0 or 1) of the transition
        F[""]]: timeout condition (either 0 or 1) of the transition
        VType: Stepped state
        F[""]]: Stepped step count

    """
    act_key, step_key = jrd.split(key, num=2)
    policy_p = jnp.einsum("as,s->a", policy, state)
    action = sample_from(policy_p, key=act_key)

    return action, *async_sample_step(mdp=mdp,
                                      action=action,
                                      state=state,
                                      episode_step=episode_step,
                                      episode_length=episode_length,
                                      key=step_key)


def sg(array: F["..."]) -> F["..."]:
    """Stop Gradient function"""
    return jax.lax.stop_gradient(array)
