from typing import Tuple, Any, Dict, Optional, Union
import json
import jax.numpy as jnp
import jax
from jax.typing import ArrayLike as KeyType
import distrax

from jaxdp.typehints import F


class MDP():
    """ Base Markov Decision Process (MDP) class

    Args:
        transition (F["... A S S"]): Column stochastic matrix that
            describes the transition dynamics of the MDP
        reward (F["... A S S"]): Reward matrix
        initial (F["... S"]): Stochastic vector for the initial state distribution
        terminal (F["... S"]): A boolean-valued vector for terminal states
        features (F["... S F"]): State features
        name (str, optional): Name of the MDP. Defaults to "MDP".
        validate (bool, optional): If true, validate the input matrices and vectors. Defaults to 
            True.
    """

    def __init__(self,
                 transition: F["... A S S"],
                 reward: F["... A S S"],
                 initial: F["... S"],
                 terminal: F["... S"],
                 features: Optional[F["... S F"]] = None,
                 name: str = "MDP",
                 validate: bool = True):
        self.name = name
        self.transition = transition
        self.reward = reward
        self.initial = initial
        self.terminal = terminal
        if features is None:
            features = jnp.eye(self.state_size)
        self.features = features

        if validate and not isinstance(self.transition, jax.core.Tracer):
            self.validate()

    def init_state(self, key: KeyType) -> F["... S"]:
        return distrax.OneHotCategorical(
            probs=self.initial, dtype="float").sample(seed=key)

    def validate(self) -> None:
        """ Validate the MDP matrices and vectors

        Raises:
            ValueError: If transition matrices are not column stochastic
            ValueError: If initial state distribution is not stochastic
            ValueError: If termination vector is not boolean-valued
            ValueError: If terminal rewards are non-zero
            ValueError: If terminal transitions are not self pointing
        """
        if not jnp.allclose(self.transition.sum(axis=-2), 1.0, atol=1e-5):
            raise ValueError(
                "Transition matrix must be column stochastic!")

        if not jnp.allclose(jnp.sum(self.initial, axis=-1), 1.0, atol=1e-5):
            raise ValueError(
                "Initial distribution must be a stochastic vector!")

        if not jnp.allclose(jnp.count_nonzero(self.terminal, axis=-1) +
                            jnp.count_nonzero(1 - self.terminal, axis=-1), self.state_size):
            raise ValueError(
                "Terminal array should only contain boolean values!")

        if not jnp.allclose(self.reward * self.terminal.reshape(1, -1, 1), 0):
            raise ValueError("Terminal rewards must be zero!")

        if not jnp.allclose(
                jnp.einsum("axs,s->as",
                           (self.transition == jnp.expand_dims(
                               jnp.eye(self.state_size), 0)),
                           self.terminal
                           ) / self.state_size,
                self.terminal.reshape(1, -1)):
            raise ValueError("Terminal transitions must be self pointing!")

    @property
    def state_size(self) -> int:
        return self.transition.shape[-2]

    @property
    def feature_size(self) -> int:
        return self.features.shape[-1]

    @property
    def action_size(self) -> int:
        return self.transition.shape[-3]

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return self.transition.shape[:-3]

    @staticmethod
    def array_names() -> Tuple[str, ...]:
        """Return the name of the arrays that describe the MDP"""
        return ("transition", "reward", "initial", "terminal")

    def __repr__(self) -> str:
        prefix = f"batch_shape={self.batch_shape}, " if len(
            self.batch_shape) > 0 else ""
        return (f"jaxdp.{self.name}({prefix}state_size={self.state_size},"
                f" action_size={self.action_size})")

    @staticmethod
    def load_mdp_from_json(file_path: str) -> "MDP":
        """ Load an MDP from the given json file path

        Args:
            file_path (str): JSON file path of the MDP definition

        Returns:
            MDP: Initiated MDP
        """
        with open(file_path, "r") as fobj:
            mdp_data = json.load(fobj)

        array_data = {}
        for array_name in MDP.array_names():
            array_data[array_name] = jnp.array(
                mdp_data.pop(array_name)).astype("float")

        return MDP(**array_data, **mdp_data)

    def save_mdp_as_json(self, file_path: str) -> None:
        """ Save an MDP (or stacked MDPs) as a json file. 

        Args:
            file_path (str): JSON file path of the MDP definition
        """
        with open(file_path, "w") as fobj:
            json.dump({
                "name": self.name,
                **{array_name: getattr(self, array_name).tolist() for array_name in self.array_names()},
            }, fobj)


def flatten_mdp(container) -> Tuple[F[""], Dict[str, Any]]:
    """Returns an iterable over container contents for registering as a Pytree"""
    flat_contents = [
        container.transition,
        container.reward,
        container.initial,
        container.terminal,
        container.features,
    ]
    return flat_contents, {
        "name": container.name,
        "validate": False,
    }


def unflatten_mdp(aux_data: Dict[str, Any], flat_contents: F[""]) -> MDP:
    """Converts flat contents into a MDP for registering as a Pytree"""
    return MDP(*flat_contents, **aux_data)
