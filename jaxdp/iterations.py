from typing import Optional, Callable, Dict, Tuple, Any, Union, List
from abc import abstractmethod
import jax.numpy as jnp
import jax.random as jrd
import jax
from jaxtyping import Array, Float

import jaxdp
from jaxdp import MDP
from jaxdp.mdp import MDP
from jaxdp.runner import BaseRunner


def q_iteration_step(mdp: MDP, value: Float[Array, "A S"], gamma: float) -> Float[Array, "A S"]:
    # TODO: Add docstring
    # TODO: Add test
    non_done = (1 - mdp.terminal)
    return mdp.reward * non_done.reshape(1, -1) + gamma * jnp.einsum(
        "axs,x,x->as",
        mdp.transition,
        jnp.max(value, axis=0),
        non_done)


def value_iteration_step(mdp: MDP, value: Float[Array, "S"], gamma: float) -> Float[Array, "S"]:
    # TODO: Add docstring
    # TODO: Add test
    non_done = (1 - mdp.terminal)
    return jnp.max(mdp.reward + gamma * jnp.einsum(
        "axs,x,x->as",
        mdp.transition,
        value,
        non_done),
        axis=0) * non_done


def policy_iteration_step(mdp: MDP, value: Float[Array, "A S"], gamma: float) -> Float[Array, "A S"]:
    # TODO: Add docstring
    # TODO: Add test
    policy_pi = jaxdp.greedy_policy(value)
    return jaxdp.q_policy_evaluation(mdp, policy_pi, gamma)


def accelerated_value_iteration_step():
    pass


def momentum_value_iteration_step():
    pass


def relaxed_value_iteration_step():
    pass


class RunIteration(BaseRunner):

    def __init__(self, state_size: int, action_size: int, seed: int) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.key, self.init_key = jrd.split(jrd.PRNGKey(seed), num=2)
        self.value = self.initialize_value()

    @abstractmethod
    def initialize_value(self) -> Float[Array, "... S"]:
        pass

    def run(self, mdp: MDP, n_steps: int, gamma: float
            ) -> List[Dict[str, Float[Array, "..."]]]:

        metrics = []
        for _ in range(n_steps):
            next_value = self.step_value(mdp, gamma)
            metrics.append(self.gather_step_info(next_value, mdp, gamma))
            self.value = next_value

        return metrics

    def gather_step_info(self,
                         next_value: Float[Array, "... S"],
                         mdp: MDP,
                         gamma: float
                         ) -> Dict[str, Float[Array, "..."]]:
        greedy_policy = self.greedy_policy(mdp, gamma)
        return {
            "expected_value": jaxdp.sg(self.expected_value(mdp)),
            "policy_evaluation": jaxdp.sg((
                self.policy_evaluation(mdp, greedy_policy, gamma) *
                mdp.initial
            ).sum()),
            "bellman_error": jaxdp.sg((
                self.bellman_operator(mdp, greedy_policy, gamma) - self.value
            ).max()),
            "value_delta": jaxdp.sg((self.value - next_value).max()),
            "value": jaxdp.sg(self.value),
            "policy": greedy_policy,
        }

    def metrics(self,
                history: List[Dict[str, Float[Array, "..."]]]
                ) -> Dict[str, Union[float, int, bool]]:
        return {
            "policy_norm": [None] + [jnp.abs(
                jnp.argmax(prev_info["policy"], 0) !=
                jnp.argmax(next_info["policy"], 0)
            ).sum(-1).item() for prev_info, next_info in zip(history[:-1], history[1:])],
            "value_delta": [(next_info["value"] - prev_info["value"]).max().item()
                            for prev_info, next_info in zip(history[:-1], history[1:])] + [None],
            **{name: [step_info[name].item() for step_info in history]
               for name in ("policy_evaluation", "expected_value", "bellman_error")},
        }

    def policy_evaluation(self, mdp: MDP, policy: Float[Array, "A S"], gamma: float
                          ) -> Float[Array, "S"]:
        return jaxdp.policy_evaluation(mdp, policy, gamma)

    @abstractmethod
    def bellman_operator(self, mdp: MDP, policy: Float[Array, "A S"], gamma: float
                         ) -> Float[Array, "S"]:
        pass

    @abstractmethod
    def greedy_policy(self, mdp: MDP, gamma: float) -> Float[Array, "A S"]:
        pass

    @abstractmethod
    def expected_value(self, mdp: MDP) -> Float[Array, ""]:
        pass

    @abstractmethod
    def step_value(self, mdp: MDP, gamma: float) -> Float[Array, "... S"]:
        pass


class QIteration(RunIteration):

    value_expr: str = "q"

    @property
    def expected_value_expr(self) -> str:
        return fr"${self.init_dist_expr}^T (\max_a {self.value_expr}_k^a)$"

    def expected_value(self, mdp: MDP) -> Float[Array, ""]:
        return jaxdp.expected_q_value(mdp, self.value)

    def greedy_policy(self, *_) -> Float[Array, "A S"]:
        return jaxdp.greedy_policy(self.value)

    def bellman_operator(self, mdp: MDP, policy: Float[Array, "A S"], gamma: float
                         ) -> Float[Array, "S"]:
        return jaxdp.bellman_q_operator(mdp, policy, self.value, gamma)

    def initialize_value(self) -> Float[Array, "A S"]:
        return jrd.uniform(self.init_key, shape=(self.action_size, self.state_size))

    def step_value(self, mdp: MDP, gamma: float) -> Float[Array, "A S"]:
        return q_iteration_step(mdp, self.value, gamma)


class ValueIteration(RunIteration):

    value_expr: str = "v"

    @property
    def expected_value_expr(self) -> str:
        return fr"${self.init_dist_expr}^T {self.value_expr}_k$"

    def expected_value(self, mdp: MDP) -> Float[Array, ""]:
        return jaxdp.expected_state_value(mdp, self.value)

    def greedy_policy(self, mdp: MDP, gamma: float) -> Float[Array, "A S"]:
        return jaxdp.greedy_policy_from_v(mdp, self.value, gamma)

    def bellman_operator(self, mdp: MDP, policy: Float[Array, "A S"], gamma: float
                         ) -> Float[Array, "S"]:
        return jaxdp.bellman_v_operator(mdp, policy, self.value, gamma)

    def initialize_value(self) -> Float[Array, "S"]:
        return jrd.uniform(self.init_key, shape=(self.state_size,))

    def step_value(self, mdp: MDP, gamma: float) -> Float[Array, "S"]:
        return value_iteration_step(mdp, self.value, gamma)


class PolicyIteration():
    pass
