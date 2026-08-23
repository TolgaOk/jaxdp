"""Fixed graph MDP from convergence studies of Q-learning."""

import jax.numpy as jnp

from jaxdp.mdp import Mdp

_EDGES = {
    0: (0, 4),
    1: (1, 3, 5),
    2: (2, 3),
    3: (1, 2, 3, 4),
    4: (0, 3, 4, 5),
    5: (1, 4, 5),
}
_STATE_SIZE = 6


def graph_mdp() -> Mdp:
    """Create the fixed six-state graph MDP."""
    transition = jnp.zeros((_STATE_SIZE, _STATE_SIZE, _STATE_SIZE))
    for state, actions in _EDGES.items():
        for action in actions:
            alternatives = tuple(candidate for candidate in actions if candidate != action)
            probability = (
                jnp.zeros(_STATE_SIZE)
                .at[action]
                .set(0.8)
                .at[jnp.array(alternatives)]
                .set(0.2 / len(alternatives))
            )
            transition = transition.at[action, :, state].set(probability)

        fallback = transition[state, :, state]
        for action in set(range(_STATE_SIZE)) - set(actions):
            transition = transition.at[action, :, state].set(fallback)

    reward_by_action_state = (jnp.eye(_STATE_SIZE) - jnp.ones((_STATE_SIZE, _STATE_SIZE))) * 0.05
    reward_by_action_state = reward_by_action_state.at[_STATE_SIZE - 1, :].set(1.0)
    reward_by_action_state = reward_by_action_state.at[4, 3].set(-1.0)
    reward = jnp.broadcast_to(
        reward_by_action_state[..., None],
        (_STATE_SIZE, _STATE_SIZE, _STATE_SIZE),
    )
    initial = jnp.full(_STATE_SIZE, 1 / _STATE_SIZE)
    terminal = jnp.zeros(_STATE_SIZE)
    return Mdp(transition, reward, initial, terminal)


__all__ = ["graph_mdp"]
