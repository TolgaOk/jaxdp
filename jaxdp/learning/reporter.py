from typing import Callable, Dict, Union, List, Tuple, NamedTuple, Type, Any, NewType
import jax
import jax.numpy as jnp
import jax.random as jrd
from jax.typing import ArrayLike as KeyType
from flax.struct import dataclass

import jaxdp
import jaxdp.mdp.sampler as sampler
from jaxdp.mdp.mdp import MDP
from jaxdp.learning.algorithms import SyncSample
from jaxdp.typehints import QType, VType, PiType, F, I
from jaxdp.utils import StaticMeta


def log(print_data,):
    step, index, report = print_data
    title = "Training metrics - Iteration"
    print("=" * 50)
    print(f"{title:^40} {step + 1}")
    print("")
    for name in report.__dataclass_fields__:
        val = getattr(report, name)[index].item()
        formatted_name = name.replace("_", " ").title()
        print(f"{formatted_name:<30} : {val:>15.4f}")


def eval_policy(mdp: MDP, policy: PiType, gamma: float) -> float:
    return (jaxdp.policy_evaluation.v(mdp, policy, gamma) * mdp.initial).sum()


class synchronous(metaclass=StaticMeta):

    @dataclass
    class ReportData:
        policy_evaluation: F["N"]
        l_inf_value_norm: F["N"]
        bellman_error: F["N"]
        expected_value: F["N"]
        value_error: F["N"]

    def init_report(train_steps: int, eval_period: int) -> ReportData:
        n_report = train_steps // eval_period
        return synchronous.ReportData(
            *(jnp.full(n_report, jnp.nan)
              for _ in synchronous.ReportData.__dataclass_fields__))

    def record(sampler_state: sampler.State,
               report: ReportData,
               mdp: MDP,
               value: QType,
               next_value: QType,
               value_star: QType,
               gamma: float,
               step: int,
               log_period: int,
               verbose: bool = True
               ) -> ReportData:
        target_policy = jaxdp.greedy_policy.q(next_value)

        expected_value = jnp.einsum("as,as,s->", target_policy, value, mdp.initial)
        l_inf_value_norm = jnp.abs(next_value - value).max()
        bellman_error = jnp.abs(jaxdp.bellman_operator.q(
            mdp, target_policy, value, gamma) - value).max()
        value_error = jnp.linalg.norm(value - value_star, ord=2)
        policy_evaluation = eval_policy(mdp, target_policy, gamma)

        index = step // log_period

        report = jax.tree.map(
            lambda r, d: r.at[index].set(d),
            report,
            synchronous.ReportData(
                policy_evaluation,
                l_inf_value_norm,
                bellman_error,
                expected_value,
                value_error)
        )

        if verbose:
            jax.debug.callback(log, (step, index, report))
        return report


class asynchronous(metaclass=StaticMeta):

    @dataclass
    class ReportData:
        mean_behavior_reward: F["N"]
        mean_behavior_length: F["N"]
        std_behavior_reward: F["N"]
        std_behavior_length: F["N"]
        bellman_error: F["N"]
        l_inf_value_norm: F["N"]
        policy_evaluation: F["N"]
        value_error: F["N"]

    def init_report(train_steps: int, eval_period: int) -> ReportData:
        n_report = train_steps // eval_period
        return asynchronous.ReportData(
            *(jnp.full(n_report, jnp.nan)
              for _ in asynchronous.ReportData.__dataclass_fields__))

    def record(sampler_state: sampler.State,
               report: ReportData,
               mdp: MDP,
               value: QType,
               next_value: QType,
               value_star: QType,
               gamma: float,
               step: int,
               log_period: int,
               verbose: bool = True
               ) -> ReportData:
        target_policy = jaxdp.greedy_policy.q(next_value)

        sampled_rewards = sampler_state.episode_reward_queue
        sampled_lengths = sampler_state.episode_length_queue

        bellman_error = jnp.abs(jaxdp.bellman_operator.q(
            mdp, target_policy, value, gamma) - value).max()
        l_inf_value_norm = jnp.abs(next_value - value).max()
        policy_evaluation = eval_policy(mdp, target_policy, gamma)
        value_error = jnp.linalg.norm(value - value_star, ord=2)

        index = step // log_period

        report = jax.tree.map(
            lambda r, d: r.at[index].set(d),
            report,
            asynchronous.ReportData(
                jnp.nanmean(sampled_rewards),
                jnp.nanmean(sampled_lengths),
                jnp.nanstd(sampled_rewards),
                jnp.nanstd(sampled_lengths),
                bellman_error,
                l_inf_value_norm,
                policy_evaluation,
                value_error)
        )

        if verbose:
            jax.debug.callback(log, (step, index, report))
        return report
