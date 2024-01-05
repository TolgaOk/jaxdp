from typing import Dict, Union, List, NamedTuple
from abc import abstractmethod
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float

from jaxdp.algorithm import BaseAlgorithm
from jaxdp.learning.sampler import Sampler, RolloutSample


class BaseLearning(BaseAlgorithm):

    def __init__(self, state_size: int, action_size: int, seed: int) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.key, self.init_key = jrd.split(jrd.PRNGKey(seed), num=2)
        self.value = self.initialize_value()

    @abstractmethod
    def initialize_value(self) -> Float[Array, "... S"]:
        pass

    @abstractmethod
    def policy(self) -> Float[Array, "A S"]:
        pass

    @abstractmethod
    def value_update(self, rollout: RolloutSample, gamma: float) -> Float[Array, "... S"]:
        pass

    def run(self, sampler: Sampler, n_steps: int,  gamma: float
            ) -> List[Dict[str, Float[Array, "..."]]]:

        metrics = []
        for _ in range(n_steps):
            self.key, step_key = jrd.split(self.key, 2)
            rollout_sample = sampler.rollout_sample(self.policy(), step_key)
            next_value = self.value_update(rollout_sample, gamma)
            metrics.append(self.gather_step_info(next_value, sampler, gamma))
            self.value = next_value

        return metrics

    def gather_step_info(self,
                         next_value: Float[Array, "... S"],
                         sampler: Sampler,
                         gamma: float
                         ) -> Dict[str, Float[Array, "..."]]:
        return {
            "episode_reward": sampler.recent_episode_rewards,
            "episode_length": sampler.recent_episode_lengths,
            "value_delta": (self.value - next_value).max(),
            "value": self.value,
        }

    def metrics(self,
                history: List[Dict[str, Float[Array, "..."]]]
                ) -> Dict[str, Union[float, int, bool]]:
        return {
            "value_delta": [(next_info["value"] - prev_info["value"]).max().item()
                            for prev_info, next_info in zip(history[:-1], history[1:])] + [None],
            "mean_episode_reward": [np.mean(step_info["episode_reward"]) for step_info in history],
            "mean_episode_length": [np.mean(step_info["episode_length"]) for step_info in history],
        }
