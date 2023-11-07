
from typing import Optional, Callable
import jax.numpy as jnp
import jax.random as jrd
import jax
from jaxtyping import Array, Float

from jaxdp.base import BaseDP


class ValueIteration(BaseDP):

    def sync_learn(self, steps: int, gamma: float = 0.99) -> Float[Array, "S A"]:
        values = jnp.zeros((self.action_size, self.state_size))
        info = {
            "performance": [],
            "values": []
        }

        for step in range(steps):
            target_values = (gamma * jnp.einsum(
                "ux,axs,x->uas", values, self.transition, (1 - self.terminal)))
            values = self.reward * (1 - self.terminal).reshape(1, -1) + \
                jnp.max(target_values, axis=0)
            info["performance"].append(self.evaluate(values))
            info["values"].append(values)

        return values, info

    def async_learn(self,
                    n_trajectories: int,
                    trajectory_length: int,
                    reset_trajectory_at_each_iteration: bool,
                    n_steps: int,
                    gamma: float):
        values = jnp.zeros((self.action_size, self.state_size))
        info = {
            "performance": [],
            "values": []
        }

        trajectories = self.async_sample(
            value=values,
            
        )
