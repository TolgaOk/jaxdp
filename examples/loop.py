from typing import Callable, Any, Tuple, Optional
import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct
from dataclasses import dataclass

from jaxdp.mdp import MDP
from jaxdp.base import bellman_optimality_operator as bellman_op
from jax.typing import ArrayLike as KeyType


@dataclass(frozen=True)
class LoopArgs:
    seed: int
    max_iters: int
    gamma: float
    tolerance: float


@struct.dataclass
class LoopState:
    l1: jnp.ndarray
    l2: jnp.ndarray
    linf: jnp.ndarray
    bellman_err: jnp.ndarray
    iteration: jnp.ndarray


def loop(mdp: MDP,
         update_fn: Callable[[Any, MDP, jnp.ndarray, Any], Any],
         init_fn: Callable[[MDP, KeyType, Any], Any],
         args: LoopArgs,
         alg_args: Any,
         progress_callback: Optional[Callable[[int], None]] = None,
         ) -> Tuple[Any, LoopState]:
    alg_state = init_fn(mdp, jrd.PRNGKey(args.seed), alg_args)

    init_carry = (alg_state, jnp.array(False))

    def scan_body(carry: Tuple[Any, jnp.ndarray], iter_idx: jnp.ndarray) -> Tuple[Tuple[Any, jnp.ndarray], LoopState]:
        current_alg_state, converged = carry

        # Call progress callback using JAX debug callback if provided
        if progress_callback is not None:
            jax.debug.callback(progress_callback, iter_idx)

        next_alg_state = jax.lax.cond(
            converged,
            lambda: current_alg_state,
            lambda: update_fn(current_alg_state, mdp, iter_idx, alg_args)
        )

        l1 = jnp.sum(jnp.abs(next_alg_state.q_values - current_alg_state.q_values))
        l2 = jnp.sqrt(
            jnp.sum(jnp.square(next_alg_state.q_values - current_alg_state.q_values)))
        linf = jnp.max(jnp.abs(next_alg_state.q_values - current_alg_state.q_values))

        bellman_err = jnp.max(jnp.abs(current_alg_state.q_values -
                                      bellman_op.q(mdp, current_alg_state.q_values, alg_args.gamma)))

        loop_state = LoopState(l1=l1, l2=l2, linf=linf,
                               bellman_err=bellman_err, iteration=iter_idx + 1)

        next_converged = converged | (loop_state.linf < args.tolerance)

        return (next_alg_state, next_converged), loop_state

    final_carry, all_loop_states_stacked = jax.lax.scan(
        scan_body,
        init_carry,
        jnp.arange(args.max_iters)
    )

    final_alg_state, _ = final_carry

    return final_alg_state, all_loop_states_stacked
