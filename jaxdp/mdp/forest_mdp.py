import jax.numpy as jnp
from jaxdp.mdp import MDP


def forest_mdp(rotation: int) -> MDP:
    """
    Constructs a simple Forest MDP for forest management decisions.

    - State space: Forest age (0 to rotation).
    - Actions:
         0: Wait (allow forest to grow).
         1: Harvest (which resets the forest age to 0).
    - Transitions:
         * For action 0: if current age s < rotation, then s -> s+1; if s == rotation, remain at rotation.
         * For action 1: regardless of current state, transition to state 0.
    - Rewards:
         * For action 0: reward is 0.
         * For action 1: reward is proportional to the current forest age (e.g. revenue = s).
    - Initial state: Forest age 0.
    - Terminal: No explicit terminal (all states are nonterminal).

    Args:
        rotation (int): Maximum forest age (rotation period).

    Returns:
        MDP: The constructed Forest MDP.
    """
    n_states = rotation + 1
    n_actions = 2

    transition = (
        jnp.zeros((n_actions, n_states, n_states))
        .at[0,
            jnp.clip(jnp.arange(n_states) + 1, 0, n_states - 1),
            jnp.arange(n_states)
            ].set(1.0)
        .at[1, 0, :].set(1)
    )
    reward = (
        jnp.zeros((n_actions, n_states, n_states))
        .at[1, jnp.arange(n_states), 0].set(jnp.arange(n_states).astype("float"))
    )
    initial = (
        jnp.zeros(n_states)
        .at[0].set(1.0)
    )
    terminal = jnp.zeros(n_states)

    return MDP(transition, reward, initial, terminal, name=f"ForestMDP[rotation={rotation}]")
