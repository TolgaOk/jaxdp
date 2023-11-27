from typing import Optional, Callable, Tuple, Dict, Union, Any
from dataclasses import dataclass
import jax.numpy as jnp
import jax.random as jrd
import jax
import distrax
from jaxtyping import Array, Float

from jaxdp.mdp.mdp import MDP


def greedy_policy(value: Float[Array, "A S"]) -> Float[Array, "A S"]:
    r"""
    Greedy policy distribution.

    Args:
        value (Float[Array,"A S"]): Q Value array

    Returns:
        Float[Array, "A S"]: Policy distribution as one hot vectors

    """
    return jax.nn.one_hot(jnp.argmax(value, axis=0, keepdims=False),
                          num_classes=value.shape[0],
                          axis=0)


def soft_policy(value: Float[Array, "A S"],
                temperature: float,
                ) -> Float[Array, "A S"]:
    r"""
    Softmax policy distribution.

    Args:
        value (Float[Array,"A S"]): Q Value array
        temperature (float): Temperature parameter of softmax. Lower values will result in
            uniform policy distribution while higher values will result a distribution closed
            to greedy policy.

    Returns:
        Float[Array, "A S"]: Policy distribution

    """
    return jax.nn.softmax(value * temperature, axis=0)


def e_greedy_policy(value: Float[Array, "A S"],
                    epsilon: float,
                    ) -> Float[Array, "A S"]:
    r"""
    Epsilon greedy policy distribution.

    Args:
        value (Float[Array,"A S"]): Q Value array
        epsilon (float): Epsilon parameter. The policy takes the greedy action with 
            (1 - epsilon + epsilon/|A|) probability while take a non-greedy action with 
            (epsilon / |A|) probability where |A| is the dimension of the action space.

    Returns:
        Float[Array, "A S"]: Policy distribution

    """
    greedy_policy_p = greedy_policy(value)
    return greedy_policy_p * (1 - epsilon) + jnp.ones_like(value) * (epsilon / value.shape[0])


def sample_from(policy: Float[Array, "A S"],
                key: jrd.KeyArray,
                ) -> Float[Array, "A S"]:
    r"""
    Sample from a policy. The samples will be one-hot vectors.

    Args:
        policy (Float[Array,"A S"]): Policy distribution
        key (jrd.KeyArray): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        Float[Array, "A S"]: Sampled actions in the one-hot vector form for each state.

    """
    return distrax.OneHotCategorical(probs=policy.T).sample(seed=key).T


def to_state_value(mdp: MDP, value: Float[Array, "A S"]) -> Float[Array, "S"]:
    # TODO: Add docstring
    # TODO: Add test
    pass


def to_state_action_value(mdp: MDP, value: Float[Array, "S"]) -> Float[Array, "A S"]:
    # TODO: Add docstring
    # TODO: Add test
    pass


def expected_state_value(mdp: MDP, value: Float[Array, "S"]) -> Float[Array, ""]:
    r"""
    Expected value of the state-values over initial distribution.

    .. math::
        \underset{s_0 \sim \rho}{\mathbb{E}} [V(S)]

    Args:
        mdp (MDP): Markov Decision Process
        value (Float[Array,"S"]): Value array

    Returns:
        Float[Array, ""]: Expected value

    """
    return (value * mdp.initial).sum()


def expected_q_value(mdp: MDP, value: Float[Array, "A S"]) -> Float[Array, ""]:
    r"""
    Expected value of the state-action values (Q) over initial distribution.

    .. math::
        \underset{\substack{s_0 \sim \rho \\ a \sim \pi^Q}}{\mathbb{E}} [Q(S, A)]

    Args:
        mdp (MDP): Markov Decision Process
        value (Float[Array,"A S"]): Q Value array

    Returns:
        Float[Array, ""]: Expected value

    """
    return expected_state_value(mdp, jnp.max(value, axis=0))


def _markov_chain_pi(mdp: MDP, policy: Float[Array, "A S"]
                     ) -> Tuple[Float[Array, "S S"], Float[Array, "S"]]:
    r"""
    Make Markov Chain of an MDP by fixing the policy.

    .. math::
        P^\pi = \underset{a \sim \pi}{\mathbb{E}}[P^a]
        r^\pi = \underset{a \sim \pi}{\mathbb{E}}[r^a]

    Args:
        mdp (MDP): Markov Decision Process
        policy (Float[Array,"A S"]): Policy distribution

    Returns:
        Float[Array, "S S"]: Transition matrix
        Float[Array, "S"]: Reward vector

    """
    transition_pi = jnp.einsum("as,axs->xs", policy, mdp.transition)
    reward_pi = jnp.einsum("as,as->s", policy, mdp.reward)
    return transition_pi, reward_pi


def sample_based_policy_evaluation(mdp: MDP,
                                   policy: Float[Array, "A S"],
                                   key: jrd.KeyArray
                                   ) -> Float[Array, "S"]:
    # TODO: Implement sample based evaluation
    # TODO: Add test
    # TODO: Add docstring
    pass


def policy_evaluation(mdp: MDP,
                      policy: Float[Array, "A S"],
                      gamma: float
                      ) -> Float[Array, "S"]:
    r"""
    Evaluate the policy for each state using the true MDP

    .. math::
        \eta(\pi)(s_i) = \big[(\mathrm{I}_n - \gamma P^\pi)r^\pi\big]_i

    Args:
        mdp (MDP): Markov Decision Process
        policy (Float[Array,"A S"]): Policy distribution
        gamma (float): Discount factor

    Returns:
        Float[Array, "S"]: Cumulative discounted reward of each state

    """
    transition_pi, reward_pi = _markov_chain_pi(mdp, policy)
    target_state_values = jnp.linalg.inv(
        jnp.eye(mdp.state_size) - gamma * transition_pi.T) @ (reward_pi * (1 - mdp.terminal))
    return target_state_values


def q_policy_evaluation(mdp: MDP,
                        policy: Float[Array, "A S"],
                        gamma: float,
                        ) -> Float[Array, "A S"]:
    r"""
    Evaluate the policy for each state-action pair using the true MDP

    .. math::
        \eta(\pi)(s_i, a_j) = \big[r^{a_j} + \gamma P^\pi (\mathrm{I}_n - \gamma P^\pi)r^\pi\big]_i

    Args:
        mdp (MDP): Markov Decision Process
        policy (Float[Array,"A S"]): Policy distribution
        gamma (float): Discount factor

    Returns:
        Float[Array, "S"]: Cumulative discounted reward of each state-action pair

    """
    mc_state_values = policy_evaluation(mdp, policy, gamma)
    return (mdp.reward * (1 - mdp.terminal).reshape(1, -1) +
            gamma * jnp.einsum("axs,x", mdp.transition, mc_state_values))


def bellman_operator(mdp: MDP,
                     policy: Float[Array, "A S"],
                     value: Float[Array, "A S"],
                     gamma: float
                     ) -> Float[Array, "A S"]:
    r"""
    Evaluate the Bellman operator for each state-action pair

    .. math::
        \mathcal{B}(Q)(s_i, a_j) = \big[r^{a_j} + \gamma 
            \underset{s^+ \sim P^{a_j}}{\mathbb{E}}[Q(s^+, a_j)]\big]_i - Q(s_i, a_j))

    Args:
        mdp (MDP): Markov Decision Process
        policy (Float[Array,"A S"]): Policy distribution
        value (Float[Array,"A S"]): Q Value array
        gamma (float): Discount factor

    Returns:
        Float[Array, "A S"]: Target values

    """
    target_values = jnp.einsum("axs,ux,ux,x->as",
                               mdp.transition, value, policy, (1 - mdp.terminal))
    return (mdp.reward + gamma * target_values - value) * (1 - mdp.terminal).reshape(1, -1)


def sync_sample(mdp: MDP,
                policy: Float[Array, "A S"],
                key: jrd.KeyArray
                ) -> Tuple[Float[Array, "S S"],
                           Float[Array, "S A"],
                           Float[Array, "S"],
                           Float[Array, "S S"],
                           Float[Array, "S"]]:
    r"""
    Synchronously sample starting from each state in the given MDP by following the given policy

    Args:
        mdp (MDP): Markov Decision Process
        policy (Float[Array,"AS"]): Policy distribution
        key (jrd.KeyArray): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        Float[Array, "S S"]: States
        Float[Array, "S A"]: Actions
        Float[Array, "S"]: Rewards
        Float[Array, "S S"]: Next states
        Float[Array, "S"]]: Termination condition (either 0 or 1)

    """
    transition_pi, reward_pi = _markov_chain_pi(mdp, policy)

    state = jnp.eye(mdp.state_size)
    action = sample_from(policy, key).T
    reward = reward_pi * (1 - mdp.terminal)
    next_state = distrax.OneHotCategorical(
        probs=transition_pi.T).sample(seed=key)
    terminal = jnp.einsum("sx,x->s", next_state, mdp.terminal)

    return state, action, reward, next_state, terminal


def async_sample_step(mdp: MDP,
                      action: Float[Array, "A"],
                      state: Float[Array, "S"],
                      episode_step: Float[Array, ""],
                      episode_length: int,
                      key: jrd.KeyArray
                      ) -> Tuple[Float[Array, "S"],
                                 Float[Array, ""],
                                 Float[Array, ""],
                                 Float[Array, ""],
                                 Float[Array, "S"],
                                 Float[Array, ""]]:
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
        action (Float[Array,"A"]): One hot action
        state (Float[Array,"S"]): Current state of the MDP
        episode_step (Float[Array,""]): Step count of the MDP
        episode_length (int): Maximum allowed episode length
        key (jrd.KeyArray): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        Float[Array, "S"]: Next states of the transition (not necessarily equal to stepped State)
        Float[Array, ""]: Rewards of the transition
        Float[Array, ""]]: termination condition (either 0 or 1) of the transition
        Float[Array, ""]]: timeout condition (either 0 or 1) of the transition
        Float[Array, "S"]: Stepped state
        Float[Array, ""]]: Stepped step count

    """
    state_key, init_key = jrd.split(key, num=2)

    next_state_p = jnp.einsum(
        "a,axs,s->x", action, mdp.transition, state)
    next_state = distrax.OneHotCategorical(
        probs=next_state_p).sample(seed=state_key)
    reward = jnp.einsum("as,a,s->", mdp.reward, action, state)
    terminal = jnp.einsum("s,s->", mdp.terminal, next_state)

    episode_step = episode_step + 1
    timeout = episode_step >= episode_length
    terminal = jnp.einsum("s,s->", mdp.terminal, next_state)
    done = jnp.logical_or(terminal, timeout)

    init_state = distrax.OneHotCategorical(
        probs=mdp.initial).sample(seed=init_key)
    state = next_state * (1 - done) + init_state * done
    episode_step = episode_step * (1 - done)

    return next_state, reward, terminal, timeout, state, episode_step


def async_sample_step_pi(mdp: MDP,
                         policy: Float[Array, "A S"],
                         state: Float[Array, "S"],
                         episode_step: Float[Array, ""],
                         episode_length: int,
                         key: jrd.KeyArray
                         ) -> Tuple[Float[Array, "S"],
                                    Float[Array, "A"],
                                    Float[Array, ""],
                                    Float[Array, ""],
                                    Float[Array, ""],
                                    Float[Array, "S"],
                                    Float[Array, ""]]:
    r"""
    Asynchronously sample from the given MDP by following the given policy.

    Args:
        mdp (MDP): Markov Decision Process
        policy (Float[Array,"A S"]): Policy distribution
        state (Float[Array,"S"]): Current state of the MDP
        episode_step (Float[Array,""]): Step count of the MDP
        episode_length (int): Maximum allowed episode length
        key (jrd.KeyArray): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        Float[Array, "A"]: Action of the transition
        Float[Array, "S"]: Next states of the transition (not necessarily equal to stepped State)
        Float[Array, ""]: Rewards of the transition
        Float[Array, ""]]: termination condition (either 0 or 1) of the transition
        Float[Array, ""]]: timeout condition (either 0 or 1) of the transition
        Float[Array, "S"]: Stepped state
        Float[Array, ""]]: Stepped step count

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
