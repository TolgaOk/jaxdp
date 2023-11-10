from typing import Tuple, Any, Dict, Optional
import json
import jax.numpy as jnp
from jaxtyping import Array, Float


class MDP():
    """Markov Decision Process (MDP) definition"""

    def __init__(self,
                 transition: Float[Array, "A S S"],
                 reward: Float[Array, "A S"],
                 initial: Float[Array, "S"],
                 terminal: Float[Array, "S"],
                 name: str = "MDP"):

        self.name = name
        self.transition = transition
        self.reward = reward
        self.initial = initial
        self.terminal = terminal

        self.state_size = self.transition.shape[-2]
        self.action_size = self.transition.shape[-3]

        if not jnp.allclose(transition.sum(axis=-2), 1.0, atol=1e-5):
            raise ValueError("Transition matrix must be column stochastic!")

        if not jnp.allclose(jnp.sum(initial), 1.0, atol=1e-5):
            raise ValueError(
                "Initial distribution must be a stochastic vector!")

        if jnp.count_nonzero(self.terminal) + jnp.count_nonzero(1 - self.terminal) != self.state_size:
            raise ValueError(
                "Terminal array should only contain boolean values!")

    @staticmethod
    def array_names() -> Tuple[str]:
        return ("transition", "reward", "initial", "terminal")

    def __repr__(self) -> str:
        return f"jaxdp.{self.name}(state_size={self.state_size}, action_size={self.action_size})"

    @staticmethod
    def load_mdp_from_json(file_path: str) -> "MDP":
        with open(file_path, "r") as fobj:
            mdp_data = json.load(fobj)

        array_data = {}
        for array_name in MDP.array_names():
            array_data[array_name] = jnp.array(mdp_data.pop(array_name))

        return MDP(**array_data, **mdp_data)

    def save_mdp_as_json(self, file_path: str) -> None:
        with open(file_path, "w") as fobj:
            json.dump({
                "name": self.name,
                **{getattr(self, array_name).tolist() for array_name in self.array_names()},
            }, fobj)


def flatten_mdp(container) -> Tuple[Float[Array, ""], Dict[str, Any]]:
    """Returns an iterable over container contents."""
    flat_contents = [
        container.transition,
        container.reward,
        container.initial,
        container.terminal,
    ]
    return flat_contents, {
        "state_size": container.state_size,
        "action_size": container.action_size,
        "name": container.name
    }


def unflatten_mdp(aux_data: Dict[str, Any], flat_contents: Float[Array, ""]) -> MDP:
    """Converts flat contents into a MDP."""
    return MDP(*flat_contents, **aux_data)
