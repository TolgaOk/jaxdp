from loop import loop, LoopState, LoopArgs
import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct
from typing import Dict, Tuple
from dataclasses import dataclass

from jaxdp.mdp import MDP
from jaxdp import bellman_optimality_operator as bellman_op
from jaxdp.utils import StaticMeta

from loop import loop, LoopArgs, LoopState

# By default JAX set float types into float32. The line below enables
# float64 data type.
jax.config.update("jax_enable_x64", True)


class vi(metaclass=StaticMeta):

    @struct.dataclass
    class State:
        q_values: jnp.ndarray
        key: jrd.PRNGKey

    @dataclass(frozen=True)
    class Args:
        gamma: float = 0.99

    def init(mdp: MDP, key: jrd.PRNGKey, args: "vi.Args") -> "vi.State":
        q_values = jrd.uniform(key, (mdp.action_size, mdp.state_size),
                               dtype="float", minval=0.0, maxval=1.0)
        return vi.State(q_values=q_values, key=key)

    def update(state: "vi.State", mdp: MDP, step: int, args: "vi.Args") -> "vi.State":
        key, subkey = jrd.split(state.key)
        prev_q_values = state.q_values
        new_q_values = bellman_op.q(mdp, prev_q_values, args.gamma)
        return state.replace(q_values=new_q_values, key=key)

