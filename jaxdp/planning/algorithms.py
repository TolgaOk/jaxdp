import jax.numpy as jnp
import jax.random as jrd
from jax.typing import ArrayLike as KeyType

import jaxdp
from jaxdp import MDP
from jaxdp.mdp import MDP
from jaxdp.typehints import QType, VType, F
from jaxdp.utils import StaticMeta


class value_iteration(metaclass=StaticMeta):

    class update(metaclass=StaticMeta):

        def q(mdp: MDP, value: QType, gamma: float) -> QType:
            # TODO: Add docstring
            # TODO: Add test
            # TODO: Add citation
            non_done = (1 - mdp.terminal)
            reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
            return reward * non_done.reshape(1, -1) + gamma * jnp.einsum(
                "axs,x,x->as",
                mdp.transition,
                jnp.max(value, axis=0),
                non_done)

        def v(mdp: MDP, value: VType, gamma: float) -> VType:
            # TODO: Add docstring
            # TODO: Add test
            # TODO: Add citation
            non_done = (1 - mdp.terminal)
            reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
            return jnp.max(reward + gamma * jnp.einsum(
                "axs,x,x->as",
                mdp.transition,
                value,
                non_done),
                axis=0) * non_done


class policy_iteration(metaclass=StaticMeta):

    class update(metaclass=StaticMeta):

        def q(mdp: MDP, value: QType, gamma: float) -> QType:
            # TODO: Add docstring
            # TODO: Add test
            # TODO: Add citation
            policy_pi = jaxdp.greedy_policy.q(value)
            return jaxdp.policy_evaluation.q(mdp, policy_pi, gamma)


class nesterov_vi(metaclass=StaticMeta):

    class update(metaclass=StaticMeta):

        def q(mdp: MDP, value: QType, prev_value: QType, gamma: float) -> QType:
            # TODO: Add test
            # TODO: Add docstring
            # TODO: Add citation
            r"""
            Evaluate the policy for each state-action pair using the true MDP
            via Nesterov accelerated VI
            """
            alpha = 1 / (1 + gamma)
            beta = (1 - jnp.sqrt(1 - gamma ** 2)) / gamma

            momentum = value + beta * (value - prev_value)
            delta = jaxdp.bellman_optimality_operator.q(mdp, momentum, gamma) - momentum

            return (momentum + alpha * delta.reshape(*momentum.shape)), value

    class init(metaclass=StaticMeta):

        def q(mdp: MDP, init_value: QType, gamma: float, key: KeyType):
            return init_value


class anderson_vi(metaclass=StaticMeta):

    class update(metaclass=StaticMeta):

        def q(mdp: MDP, value: QType, prev_value: QType, gamma: float,) -> QType:
            # TODO: Add test
            # TODO: Add citation
            r"""
            Evaluate the policy for each state-action pair using the true MDP
            via Anderson accelerated VI
            """
            value_diff = value - prev_value
            bellman_value = jaxdp.bellman_optimality_operator.q(mdp, value, gamma)
            bellman_prev_value = jaxdp.bellman_optimality_operator.q(mdp, prev_value, gamma)
            bellman_value_diff = bellman_value - bellman_prev_value

            delta_numerator = jnp.einsum("as,as->", value_diff, bellman_value - value)
            delta_denumerator = jnp.einsum("as,as->", value_diff, bellman_value_diff - value_diff)

            condition = jnp.isclose(delta_denumerator, 0, atol=1e-2).astype("float")
            delta = (delta_numerator * (1 - condition)) / (delta_denumerator + condition)

            return (1 - delta) * bellman_value + delta * bellman_prev_value, value

    class init(metaclass=StaticMeta):

        def q(mdp: MDP, init_value: QType, gamma: float, key: KeyType):
            return init_value


class quasi_policy_iteration(metaclass=StaticMeta):

    class update(metaclass=StaticMeta):

        def v(mdp: MDP, value: VType, gamma: float) -> VType:
            # TODO: Implement
            # TODO: Add test
            # TODO: Add docstring
            # TODO: Add citation
            raise NotADirectoryError

    class init(metaclass=StaticMeta):

        def v(mdp: MDP, init_value: VType, gamma: float, key: KeyType):
            raise NotADirectoryError
