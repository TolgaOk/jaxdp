from typing import Tuple, Any, Union, Type
import jax.numpy as jnp
import jax.random as jrd
import jax

from jaxdp.mdp import MDP


def delayed_reward_mdp(delay: int, action_size: int, reward_std: float, key: jrd.KeyArray) -> MDP:
    # TODO: Add test
    # TODO: Add documentation
    state_size = int((action_size ** (delay + 1) - 1) / (action_size - 1))
    n_leafs = action_size ** delay
    n_pre_leaf = action_size ** (delay - 1)
    leaf_transitions = jnp.tile(jnp.eye(state_size), (action_size, 1, 1)
                                )[:, :, -n_leafs:]
    non_leaf_transitions = jax.nn.one_hot(
        x=(jnp.arange(state_size - n_leafs) * action_size).reshape(1, -1) +
        (jnp.arange(action_size) + 1).reshape(-1, 1),
        num_classes=state_size,
        axis=-2)
    transition = jnp.concatenate(
        [non_leaf_transitions, leaf_transitions], axis=-1)

    reward_means = jnp.concatenate([jnp.ones((n_pre_leaf,)),
                                    -jnp.ones((n_pre_leaf * (action_size - 1),))])
    reward_stds = jrd.normal(key, (n_leafs,)) * reward_std
    reward = jnp.zeros((action_size, state_size))
    reward = reward.at[:, -(n_pre_leaf + n_leafs):-n_leafs].set(
        (reward_means + reward_stds).reshape(action_size, -1, order="F"))

    initial = jnp.concatenate([jnp.ones((1,)), jnp.zeros((state_size - 1))])
    terminal = jnp.concatenate(
        [jnp.zeros((state_size - n_leafs,)), jnp.ones((n_leafs,))])

    return MDP(transition, reward, initial, terminal, f"DelayedRewardMDP(delay=f{delay})")
