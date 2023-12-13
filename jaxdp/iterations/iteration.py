from typing import Dict, Union, List
from abc import abstractmethod
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float

import jaxdp
from jaxdp import MDP
from jaxdp.mdp import MDP
from jaxdp.algorithm import BaseAlgorithm


class BaseIteration(BaseAlgorithm):

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
            "expected_value": self.expected_value(mdp),
            "policy_evaluation": (
                self.policy_evaluation(mdp, greedy_policy, gamma) *
                mdp.initial
            ).sum(),
            "bellman_error": (
                self.bellman_operator(mdp, greedy_policy, gamma) - self.value
            ).max(),
            "value_delta": (self.value - next_value).max(),
            "value": self.value,
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
