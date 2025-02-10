"""Implementation of tabular Q-learning in an mjnax (MuJoCo on gymnax) environment.

    Disclaimer:
    This example is not intended to be imported. If you want to use this implementation,
    we suggest copying the source code. Although this approach may seem counterintuitive
    from a software development perspective, we find it more flexible for research purposes.
"""

from typing import Tuple, Dict, Any
import os
from itertools import chain
import pandas as pd
import json
from flax import struct
from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax.random as jrd
import jax
from jax.typing import ArrayLike as KeyType
from gymnax.environments.environment import Environment
from nestedtuple import nestedtuple

import jaxdp
from jaxdp.learning.algorithms import q_learning, StepSample, reducer
import jaxdp.mdp.sampler.gym as sampler
from jaxdp.typehints import F, QType
from mjnax.pendulum import MjxModelType, DiscretizedPendulum


# By default JAX set float types into float32. The line below enables
# float64 data type.
jax.config.update("jax_enable_x64", True)


@nestedtuple
class Arg:
    seed: int = 42                     # Initial seeds
    n_env: int = 4                     # Number of parallel environments for sampling

    class policy_fn:
        epsilon: float = 0.15          # Epsilon-greedy parameter

    class update_fn:
        alpha: float = 0.10            # Step size (a.k.a learning rate)

    class train_loop:
        gamma: float = 0.99            # Discount factor
        n_steps: int = 1000            # Number of steps

    class evaluation:
        period: int = 50               # Evaluation period (in terms of <n_steps>)
        n_env: int = 10
        n_step: int = 100
        queue_size: int = 10
        max_episode_len: int = 100

    class sampler_init:
        queue_size: int = 50           # Queue size of the sampler for the metrics

    class sampler_fn:
        max_episode_len: int = 100     # Maximum length of an episode allowed by the sampler
        rollout_len: int = 10          # Length of a rollout

    class value_init:
        minval: float = -1.0           # Minimum value of the uniform distribution
        maxval: float = 1.0            # Maxiumum value of the uniform distribution


@struct.dataclass
class ReportData:
    """ Report of the training. N is the number of epochs. """
    mean_behavior_reward: F["N"]
    mean_behavior_length: F["N"]
    std_behavior_reward: F["N"]
    std_behavior_length: F["N"]
    mean_target_reward: F["N"]
    mean_target_length: F["N"]
    std_target_reward: F["N"]
    td_error: F["N"]
    l1_value_diff_norm: F["N"]
    l0_value_diff_norm: F["N"]


@struct.dataclass
class MetricData:
    """ Training metric of an epoch. N is the number of steps in an epoch. """
    td_error: F["N"]
    l1_value_diff_norm: F["N"]
    l0_value_diff_norm: F["N"]


@struct.dataclass
class RunState():
    """ State of the training run. """
    key: jax.Array
    sampler: sampler.State
    value: QType
    env_model: MjxModelType
    metric: MetricData
    report: ReportData


@struct.dataclass
class RunStatic():
    """ Static objects of the training run. """
    env: Environment


@partial(jax.vmap, in_axes=(0, None, None, None), out_axes=RunState(0, 0, None, None, None, None))
def init(sampler_key: KeyType, value_key: KeyType, static: RunStatic, arg: Arg) -> RunState:
    """ Initialize training state for n parallel rollout samplers.

    Args:
        sampler_key (KeyType): rng
        value_key (KeyType): rng
        static (RunStatic): train statics
        arg (Arg): train args

    Returns:
        RunState: initialized training state
    """

    key, env_key = jrd.split(sampler_key, 2)
    env_model = static.env.default_params
    obs, env_state = static.env.reset(env_key, env_model)

    metric_len = arg.evaluation.period
    report_len = arg.train_loop.n_steps // arg.evaluation.period

    return RunState(
        key,
        sampler.init_sampler_state(obs, env_state, arg.sampler_init.queue_size),
        jrd.uniform(value_key, (static.env.num_actions, static.env.num_states,),
                    dtype="float", **arg.value_init._asdict()),
        env_model,
        MetricData(*((jnp.full(metric_len, jnp.nan)) for _ in MetricData.__dataclass_fields__)),
        ReportData(*(jnp.full(report_len, jnp.nan) for _ in ReportData.__dataclass_fields__)),
    )


@partial(jax.vmap,
         in_axes=(RunState(0, 0, None, None, None, None), None, None),
         out_axes=(0, RunState(0, 0, None, None, None, None)))
def sample_rollout(state: RunState,
                   static: RunStatic,
                   arg: Arg
                   ) -> Tuple[sampler.RolloutData, RunState]:
    """ Collect n rollouts.

    Args:
        state (RunState): train state
        static (RunStatic): train statics
        arg (Arg): train arguments

    Returns:
        Tuple[sampler.RolloutData, RunState]:
            - rollout data
            - updated training state
    """
    key, step_key = jrd.split(state.key)
    step_keys = jrd.split(step_key, arg.sampler_fn.rollout_len)
    rollout = sampler.init_rollout(
        static.env.num_states,
        static.env.num_actions,
        arg.sampler_fn.rollout_len)

    def policy(key: KeyType, obs: F["S"]) -> F["A"]:
        """ Epsilon-greedy policy """
        pi = jaxdp.e_greedy_policy.q(state.value, arg.policy_fn.epsilon)
        policy_p = jnp.einsum("as,s->a", pi, obs)
        act = jaxdp.sample_from(policy_p, key)
        return act

    def step(i: int, payload):
        rollout, sampler_state = payload
        act_key, sample_key = jrd.split(step_keys[i], 2)
        act = policy(act_key, sampler_state.last_obs)
        step_data, sampler_state = sampler.step(
            sample_key,
            act,
            sampler_state,
            state.env_model,
            static.env,
            arg.sampler_fn.max_episode_len)
        rollout = jax.tree.map(lambda x, y: x.at[i].set(y), rollout, step_data)
        return rollout, sampler_state

    rollout, sampler_state = jax.lax.fori_loop(
        0, arg.sampler_fn.rollout_len, step, (rollout, state.sampler))

    return rollout, state.replace(sampler=sampler_state, key=key)


def update(rollout: sampler.RolloutData,
           state: RunState,
           static: RunStatic,
           arg: Arg
           ) -> Tuple[RunState, Dict[str, Any]]:
    """ Tabular Q-learning value update.

    Args:
        rollout (sampler.RolloutData): rollout data
        state (RunState): train state
        static (RunStatic): train statics
        arg (Arg): train arguments

    Returns:
        Tuple[RunState, Dict[str, Any]]:
            - updated train state
            - metrics for the update
    """

    @partial(jax.vmap, in_axes=(0, None, None))
    @partial(jax.vmap, in_axes=(0, None, None))
    def value_step(sample: sampler.RolloutData, value: QType, gamma: float):
        return q_learning.asynchronous.step(sample, value, gamma)

    rollout = StepSample(rollout.obs, rollout.next_obs, rollout.action,
                         rollout.reward, rollout.terminal, rollout.timeout)
    batch_next_value = value_step(rollout, state.value, arg.train_loop.gamma)
    target_value = reducer.every_visit(rollout, batch_next_value)
    next_value = q_learning.update(state.value, target_value, alpha=arg.update_fn.alpha)

    return next_value, {
        "td_error": jnp.abs(batch_next_value).mean(),
        "l1_value_diff_norm": jnp.abs(next_value - state.value).mean(),
        "l0_value_diff_norm": jnp.abs(next_value != state.value).mean(),
    }


def eval_policy(key: KeyType, state: RunState, static: RunStatic, arg: Arg) -> Dict[str, F[""]]:
    """ Run n eval episodes with the target(greedy) policy. """

    @jax.vmap
    def init_sampler(key: KeyType) -> sampler.State:
        """ Initialize sampler for evaluation. """
        env_key, sampler_key = jrd.split(key, 2)
        obs, env_state = static.env.reset_env(env_key, state.env_model)
        sampler_state = sampler.init_sampler_state(
            obs, env_state, arg.evaluation.queue_size)
        return sampler_state, sampler_key

    sampler_state, sampler_key = init_sampler(jrd.split(key, arg.evaluation.n_env))
    _, updated_state = sample_rollout(
        state.replace(
            key=sampler_key,
            sampler=sampler_state,
        ),
        static,
        arg._replace(
            policy_fn=arg.policy_fn._replace(epsilon=0),
            sampler_fn=arg.sampler_fn._replace(
                rollout_len=arg.evaluation.n_step,
                max_episode_len=arg.evaluation.max_episode_len),
            sampler_init=arg.sampler_init._replace(queue_size=arg.evaluation.queue_size)
        )
    )

    return {"mean_target_reward": jnp.nanmean(updated_state.sampler.episode_reward_queue),
            "std_target_reward": jnp.nanstd(updated_state.sampler.episode_reward_queue),
            "mean_target_length": jnp.nanmean(updated_state.sampler.episode_length_queue)}


def log(state: RunState, i_epoch: int, arg: Arg) -> None:
    """ Print the reports of an epoch.

    Args:
        state (RunState): train state
        i_epoch (int): epoch index
        arg (Arg): train arguments
    """
    title = "Training Metrics - Step"
    step_report_data = jax.tree.map(lambda x: x[i_epoch], state.report)
    print("=" * 43)
    print(f"{title:^40} {(i_epoch + 1) * arg.evaluation.period}")
    print("-" * 43)
    for name, val in step_report_data.__dict__.items():
        formatted_name = name.replace("_", " ").title()
        print(f"{formatted_name:<25} | {val:>15.4f}")


@partial(jax.jit, static_argnames=["static", "arg"])
def train(key: KeyType, state: RunState, static: RunStatic, arg: Arg) -> RunState:
    """ Run q-learning.

    Args:
        key (KeyType): rng for evaluation
        state (RunState): train state
        static (RunStatic): train statics
        arg (Arg): train arguments

    Returns:
        RunState: final train state
    """

    n_epochs = arg.train_loop.n_steps // arg.evaluation.period
    eval_keys = jrd.split(key, n_epochs)

    def train_step(i: int, state: RunState) -> RunState:
        """ Singe training step. """
        rollout, state = sample_rollout(state, static, arg)
        next_value, metric_dict = update(rollout, state, static, arg)

        return state.replace(
            metric=jax.tree.map(
                lambda x, y: x.at[i % arg.evaluation.period].set(y),
                state.metric,
                MetricData(**metric_dict)
            ),
            value=next_value
        )

    def report(i_epoch: int, state: RunState):
        """ Record the epoch overview to report. """
        eval_dict = eval_policy(eval_keys[i_epoch], state, static, arg)

        step_report_data = ReportData(
            jnp.nanmean(state.sampler.episode_reward_queue),
            jnp.nanmean(state.sampler.episode_length_queue),
            jnp.nanstd(state.sampler.episode_reward_queue),
            jnp.nanstd(state.sampler.episode_length_queue),
            **eval_dict,
            **jax.tree.map(lambda x: jnp.nanmean(x), state.metric).__dict__
        )
        report_data = jax.tree.map(lambda y, x: y.at[i_epoch].set(
            x), state.report, step_report_data)

        # Refresh metric and sampler queues
        sampler_state = sampler.refresh_queues(state.sampler)
        metric = jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), state.metric)

        state = state.replace(
            sampler=sampler_state,
            metric=metric,
            report=report_data
        )
        return state

    def epoch(i_epoch: int, state: RunState) -> RunState:
        """ Run one epoch. """
        n_train = arg.evaluation.period

        state = jax.lax.fori_loop(i_epoch * n_train, (i_epoch + 1) * n_train, train_step, state)
        state = report(i_epoch, state)
        jax.debug.callback(log, state, i_epoch, arg=arg)
        return state

    return jax.lax.fori_loop(0, n_epochs, epoch, state)


def report_to_dataframe(report: ReportData, percentile: int = 25) -> pd.DataFrame:
    """ Make a dataframe from training report.

    Args:
        report (ReportData): training report
        percentile (int, optional): percentile of the half tube width. Defaults to 25.

    Returns:
        pd.DataFrame: resulting data frame
    """
    length = len(report.mean_behavior_reward)

    index = pd.MultiIndex.from_product(
        [["pendulum"], ["q-learning"], list(range(length))],
        names=["ENV", "ALG", "STEP"])
    columns = pd.MultiIndex.from_product(
        [report.__dataclass_fields__, ["low", "med", "high"]],
        names=["METRIC", "PERCENTILE"])

    data = []
    for name in report.__dataclass_fields__:
        values = getattr(report, name)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        percentiles = jnp.nanpercentile(
            values, q=jnp.array([percentile, 50, 100 - percentile]), axis=0)
        data.append(percentiles)

    return pd.DataFrame(data=jnp.stack(list(chain(*data)), axis=1), columns=columns, index=index)


def save_run(experiment_name: str, arg: Arg, report: ReportData, result_dir: str = "./results") -> None:
    """ Save the train arguments and report.

    Args:
        experiment_name (str): name of the experiment (used as the folder name)
        arg (Arg): train arguments
        report (ReportData): train report
        result_dir (str, optional): Main directory to save. Defaults to "./results".
    """
    dir_path = os.path.join(result_dir, experiment_name)
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path, f"arg.json"), "w") as fp:
        json.dump(arg._asdict(), fp)

    df = report_to_dataframe(report, percentile=25)
    df.to_parquet(os.path.join(dir_path, f"report.parquet"))


if __name__ == "__main__":
    arg = Arg(
        seed=42,
        n_env=32,
        sampler_init=Arg.sampler_init(queue_size=100),
        policy_fn=Arg.policy_fn(epsilon=0.15),
        sampler_fn=Arg.sampler_fn(rollout_len=50),
        train_loop=Arg.train_loop(n_steps=5000, gamma=0.99),
        evaluation=Arg.evaluation(period=100, n_env=1),
        update_fn=Arg.update_fn(alpha=0.01)
    )
    key = jrd.PRNGKey(arg.seed)
    train_key, sampler_key, value_key = jrd.split(key, 3)

    static = RunStatic(DiscretizedPendulum())
    state = init(jrd.split(sampler_key, arg.n_env), value_key, static, arg)
    final_state = jax.block_until_ready(
        train(train_key, state, static, arg)
    )

    script_name = __file__.split("/")[-1].split(".py")[0]
    save_run(script_name, arg, final_state.report)

    # TODO: Make a rendering if possible
    # TODO: (Maybe) Save final state instead of df
    # TODO: (Low priority) Add CLI for args
