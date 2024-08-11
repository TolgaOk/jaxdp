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
from jaxdp.typehints import QType, VType, F


class PlanningMetrics(NamedTuple):
    expected_value: F["N"]
    policy_evaluation: F["N"]
    bellman_error: F["N"]
    value_delta: F["N"]
    policy_delta: F["N"]
    value_error: F["N"]

    @staticmethod
    def initialize(step_size: int) -> "PlanningMetrics":
        return PlanningMetrics(*[jnp.full((step_size,), jnp.nan) for _ in range(6)])

    def write(self, index: int, values: Dict[str, float]) -> "PlanningMetrics":
        self_dict = self._asdict()
        return PlanningMetrics(
            *[self_dict[name].at[index].set(value)
              for name, value in values.items()]
        )


def train(mdp: MDP,
          init_value: QType,
          update_state: Any,
          n_iterations: int,
          gamma: float,
          value_star: QType,
          update_fn: Callable[[MDP, QType, Any, float], Tuple[QType, Any]],
          verbose: bool = True
          ) -> Tuple[PlanningMetrics, QType, Any]:
    # TODO: Add docstring

    metrics = PlanningMetrics.initialize(n_iterations)
    value = init_value
    policy = jaxdp.greedy_policy.q(value)

    def print_log(data):
        step, info = data
        print(f"Progress: {step:5d}")

    def step_fn(index, step_data):
        metrics, value, policy, update_state = step_data

        next_value, update_state = update_fn(mdp, value, update_state, gamma)
        next_policy = jaxdp.greedy_policy.q(next_value)

        step_info = {
            "expected_value": jaxdp.expected_value.q(mdp, value),
            "policy_evaluation": (jaxdp.policy_evaluation.v(mdp, policy, gamma) * mdp.initial).sum(),
            "bellman_error": jnp.abs(value - jaxdp.bellman_operator.q(mdp, policy, value, gamma)).max(),
            "value_delta": jnp.max(jnp.abs(next_value - value)),
            "policy_delta": (1 - jnp.all(jnp.isclose(next_policy, policy), axis=0)).sum(),
            "value_error": jnp.abs(value - value_star).max(),
        }

        metrics = metrics.write(index, step_info)
        if verbose:
            call(print_log, (index + 1, step_info))
        return metrics, next_value, next_policy, update_state

    metrics, value, policy, update_state = jax.lax.fori_loop(
        0, n_iterations, step_fn, (metrics, value, policy, update_state))

    return metrics, value, update_state


def no_update_state(update_fn: Callable):
    """ Update function wrapper to ignore update_state in the train
    """
    def wrapper(mdp, value, update_state, gamma, *args, **kwargs):
        return update_fn(mdp, value, gamma, *args, **kwargs), update_state

    return wrapper
