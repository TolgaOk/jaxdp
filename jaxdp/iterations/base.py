from typing import Dict, NamedTuple, Union, List, Tuple, Callable, Type, Any
from abc import abstractmethod
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float
from jax.typing import ArrayLike
import jax
from jax.experimental.host_callback import call

import jaxdp
from jaxdp import MDP
from jaxdp.mdp import MDP


class IterationMetrics(NamedTuple):

    expected_value: Float[Array, "N"]
    policy_evaluation: Float[Array, "N"]
    bellman_error: Float[Array, "N"]
    value_delta: Float[Array, "N"]
    policy_delta: Float[Array, "N"]

    @staticmethod
    def initialize(step_size: int) -> "IterationMetrics":
        return IterationMetrics(*[jnp.full((step_size,), jnp.nan) for _ in range(5)])

    def write(self, index: int, values: Dict[str, float]) -> "IterationMetrics":
        self_dict = self._asdict()
        return IterationMetrics(
            *[self_dict[name].at[index].set(value)
              for name, value in values.items()]
        )


ValueArray = Float[Array, "A S"]


def train(
    mdp: MDP,
    init_value: ValueArray,
    n_iterations: int,
    gamma: float,
    update_fn: Callable[[MDP, ValueArray, Any, float], Tuple[ValueArray, Any]],
    update_state: Any,
    verbose: bool = True
) -> Tuple[IterationMetrics, ValueArray]:

    metrics = IterationMetrics.initialize(n_iterations)
    value = init_value
    policy = jaxdp.greedy_policy(value)

    def print_log(data):
        step, info = data
        print(f"Progress: {step:5d}")

    def step_fn(index, step_data):
        metrics, value, policy, update_state = step_data

        next_value, update_state = update_fn(mdp, value, update_state, gamma)
        next_policy = jaxdp.greedy_policy(next_value)

        step_info = {
            "expected_value": jaxdp.expected_q_value(mdp, value),
            "policy_evaluation": (jaxdp.policy_evaluation(mdp, policy, gamma) * mdp.initial).sum(),
            "bellman_error": jnp.abs(value - jaxdp.bellman_q_operator(mdp, policy, value, gamma)).max(),
            "value_delta": jnp.max(jnp.abs(next_value - value)),
            "policy_delta": (1 - jnp.all(jnp.isclose(next_policy, policy), axis=0)).sum(),
        }

        metrics = metrics.write(index, step_info)
        if verbose:
            call(print_log, (index + 1, step_info))
        return metrics, next_value, next_policy, update_state

    metrics, value, policy, update_state = jax.lax.fori_loop(
        0, n_iterations, step_fn, (metrics, value, policy, update_state))

    return metrics, value, update_state
