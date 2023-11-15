from typing import Optional, Callable, Tuple, Dict, Union, Any
from dataclasses import dataclass
import jax.numpy as jnp
import jax.random as jrd
import jax
import distrax
from jaxtyping import Array, Float

from jaxdp.mdp.mdp import MDP
from jaxdp import TransitionType


def greedy_policy(value: Float[Array, "A S"]) -> Float[Array, "A S"]:
    # TODO: Add docstring
    return jax.nn.one_hot(jnp.argmax(value, axis=0, keepdims=False),
                          num_classes=value.shape[0],
                          axis=0)


def soft_policy(value: Float[Array, "A S"],
                temperature: float,
                ) -> Float[Array, "A S"]:
    # TODO: Add docstring
    return jax.nn.softmax(value * temperature, axis=0)


def e_greedy_policy(value: Float[Array, "A S"],
                    epsilon: float,
                    ) -> Float[Array, "A S"]:
    # TODO: Add docstring
    greedy_policy_p = greedy_policy(value)
    return greedy_policy_p * (1 - epsilon) + jnp.ones_like(value) * (epsilon / value.shape[0])


def sample_from(policy: Float[Array, "A S"],
                key: jrd.KeyArray,
                ) -> Float[Array, "A S"]:
    # TODO: Add docstring
    return distrax.OneHotCategorical(probs=policy.T).sample(seed=key).T


def expected_state_value(mdp: MDP, value: Float[Array, "S"]) -> float:
    # TODO: Add docstring
    return (value * mdp.initial).sum()


def expected_q_value(mdp: MDP, value: Float[Array, "A S"]) -> float:
    # TODO: Add docstring
    return expected_state_value(mdp, jnp.max(value, axis=0))


def _markov_chain_pi(mdp: MDP, policy: Float[Array, "A S"]
                     ) -> Tuple[Float[Array, "S S"], Float[Array, "S"]]:
    # TODO: Add docstring
    transition_pi = jnp.einsum("as,axs->xs", policy, mdp.transition)
    reward_pi = jnp.einsum("as,as->s", policy, mdp.reward)
    return transition_pi, reward_pi


def sample_based_policy_evaluation():
    # TODO: Implement sample based evaluation
    # TODO: Add test
    # TODO: Add docstring
    pass


def policy_evaluation(mdp: MDP,
                      policy: Float[Array, "A S"],
                      gamma: float
                      ) -> Float[Array, "S"]:
    # TODO: Add docstring
    transition_pi, reward_pi = _markov_chain_pi(mdp, policy)
    mc_state_values = jnp.linalg.inv(
        jnp.eye(mdp.state_size) - gamma * transition_pi.T) @ (reward_pi * (1 - mdp.terminal))
    return mc_state_values


def q_policy_evaluation(mdp: MDP,
                        policy: Float[Array, "A S"],
                        gamma: float,
                        ) -> Float[Array, "A S"]:
    # TODO: Add docstring
    mc_state_values = policy_evaluation(mdp, policy, gamma)
    return (mdp.reward * (1 - mdp.terminal).reshape(1, -1) +
            gamma * jnp.einsum("axs,x", mdp.transition, mc_state_values))


def bellman_error(mdp: MDP,
                  policy: Float[Array, "A S"],
                  value: Float[Array, "A S"],
                  gamma: float
                  ) -> Float[Array, "A S"]:
    # TODO: Add docstring
    target_values = jnp.einsum("axs,ux,ux,x->as",
                               mdp.transition, value, policy, (1 - mdp.terminal))
    return (mdp.reward + gamma * target_values - value) * (1 - mdp.terminal).reshape(1, -1)


def sync_sample_step(mdp: MDP,
                     policy: Float[Array, "A S"],
                     key: jrd.KeyArray
                     ) -> Tuple[Float[Array, "S S"],
                                Float[Array, "S A"],
                                Float[Array, "S"],
                                Float[Array, "S S"],
                                Float[Array, "S"]]:
    # TODO: Add docstring
    transition_pi, reward_pi = _markov_chain_pi(mdp, policy)

    state = jnp.eye(mdp.state_size)
    action = sample_from(policy, key).T
    reward = reward_pi * (1 - mdp.terminal)
    next_state = distrax.OneHotCategorical(
        probs=transition_pi.T).sample(seed=key)
    terminal = jnp.einsum("sx,x->s", next_state, mdp.terminal)

    return state, action, reward, next_state, terminal


def async_sample_step(mdp: MDP,
                      policy: Float[Array, "A S"],
                      state: Float[Array, "S"],
                      episode_step: Float[Array, ""],
                      episode_length: int,
                      key: jrd.KeyArray
                      ) -> TransitionType:
    """ Multiplication-based async sample """
    # TODO: Add test
    # TODO: Add docstring
    act_key, state_key, init_key = jrd.split(key, num=3)
    policy_p = jnp.einsum("as,s->a", policy, state)
    action = sample_from(policy_p, key=act_key)

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

    return state, action, reward, next_state, terminal, timeout, episode_step
