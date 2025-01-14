"""Implementation of tabular Q-learning in a finite MDP.

    Disclaimer:
    This example is not intended to be imported. If you want to use this implementation,
    we suggest copying the source code. Although this approach may seem counterintuitive
    from a software development perspective, we find it more flexible for research purposes.
"""

from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax.random as jrd
import jax
from flax.struct import dataclass
from nestedtuple import nestedtuple

import jaxdp
import jaxdp.mdp.sampler.mdp as sampler
from jaxdp.mdp.mdp import MDP
from jaxdp.learning.algorithms import q_learning, reducer
import jaxdp.learning.reporter as reporter
from jaxdp.typehints import QType
from jax.typing import ArrayLike as KeyType


# By default JAX set float types into float32. The line below enables
# float64 data type.
jax.config.update("jax_enable_x64", True)


# Define the arguments
@nestedtuple
class Args:
    seed: int = 42                     # Initial seeds
    n_env: int = 8                     # Number of parallel environments for sampling

    class policy_fn:
        epsilon: float = 0.15          # Epsilon-greedy parameter

    class update_fn:
        alpha: float = 0.10            # Step size (a.k.a learning rate)

    class train_loop:
        gamma: float = 0.99            # Discount factor
        n_steps: int = 1000            # Number of steps
        eval_period: int = 50          # Evaluation period (in terms of <n_steps>)

    class sampler_init:
        queue_size: int = 50           # Queue size of the sampler for the metrics

    class sampler_fn:
        max_episode_len: int = 15      # Maximum length of an episode allowed by the sampler
        rollout_len: int = 16          # Length of a rollout

    class value_init:
        minval: float = 0.0            # Minimum value of the uniform distribution
        maxval: float = 1.0            # Maximum value of the uniform distribution

    class mdp_init:
        p_slip: float = 0.15           # Probability of slipping
        board: Tuple[str] = ("#####",  # The board of the grid-world
                             "#  @#",
                             "#  X#",
                             "#P  #",
                             "#####")


@dataclass
class State:
    """ State of the training run. """
    key: KeyType
    sampler: sampler.State
    value: QType
    mdp: MDP
    report: reporter.asynchronous.ReportData


@partial(jax.vmap,
         in_axes=(State(0, 0, None, None, None), None),
         out_axes=(0, State(0, 0, None, None, None)))
def rollout_sample(state: State, arg: Args) -> Tuple[sampler.RolloutData, State]:
    """ Collect n rollouts.

    Args:
        state (State): train state
        arg (Args): train arguments

    Returns:
        Tuple[sampler.RolloutData, State]:
            - rollout data
            - updated train state
    """
    key, sample_key = jrd.split(state.key, 2)
    policy = jaxdp.e_greedy_policy.q(state.value, arg.policy_fn.epsilon)
    rollout, sampler_state = sampler.rollout(
        sample_key,
        state.sampler,
        policy,
        state.mdp,
        arg.sampler_fn.rollout_len,
        arg.sampler_fn.max_episode_len)

    return rollout, state.replace(key=key, sampler=sampler_state)


@partial(jax.vmap, in_axes=(0, None, None), out_axes=State(0, 0, None, None, None))
def init(sampler_key: KeyType, value_key: KeyType, arg: Args) -> State:
    """ Initialize train state for n parallel rollout samplers.

    Args:
        sampler_key (KeyType): sampler rng
        value_key (KeyType): value initialization rng
        arg (Args): train arguments

    Returns:
        State: initialized train state
    """
    key, init_sampler_key = jrd.split(sampler_key, 2)
    mdp = jaxdp.mdp.grid_world(**arg.mdp_init._asdict())
    return State(
        key,
        sampler.init_sampler_state(init_sampler_key, mdp, arg.sampler_init.queue_size),
        jrd.uniform(value_key, (mdp.action_size, mdp.state_size,),
                    dtype="float", **arg.value_init._asdict()),
        mdp,
        reporter.asynchronous.init_report(
            arg.train_loop.n_steps,
            arg.train_loop.eval_period)
    )


def update(rollout: sampler.RolloutData, state: State, arg: Args) -> QType:
    """ Q-learning update with batch of samples

    Args:
        rollout (sampler.RolloutData): batch of rollouts
        state (State): train state
        arg (Args): train arguments

    Returns:
        QType: updated q values
    """

    @partial(jax.vmap, in_axes=(0, None, None))
    @partial(jax.vmap, in_axes=(0, None, None))
    def value_step(sample: sampler.RolloutData, value: QType, gamma: float):
        return q_learning.asynchronous.step(sample, value, gamma)

    batch_next_value = value_step(rollout, state.value, arg.train_loop.gamma)
    target_value = reducer.every_visit(rollout, batch_next_value)
    next_value = q_learning.update(state.value, target_value, alpha=arg.update_fn.alpha)

    return next_value


@partial(jax.jit, static_argnames=["arg"])
def train(state: State, arg: Args) -> State:
    """ Run training

    Args:
        state (State): train state
        arg (Args): train arguments

    Returns:
        State: final train state
    """

    def train_step(i: int, state: State):
        rollout, state = rollout_sample(state, arg)
        next_value = update(rollout, state, arg)

        # Report the training metrics
        report_data = (
            state.sampler,
            state.report,
            state.mdp,
            state.value,
            next_value,
            jnp.full_like(next_value, jnp.nan),
            arg.train_loop.gamma,
            i,
            arg.train_loop.eval_period,
        )

        is_report = (i % arg.train_loop.eval_period) == (arg.train_loop.eval_period - 1)
        report = jax.lax.cond(
            is_report,
            reporter.asynchronous.record,
            lambda _, report, *__: report,
            *report_data)

        # Refresh reward queues after reporting
        sampler_state = jax.lax.cond(
            is_report,
            sampler.refresh_queues,
            lambda sampler_state: sampler_state,
            state.sampler)

        return state.replace(
            sampler=sampler_state,
            value=next_value,
            report=report
        )

    return jax.lax.fori_loop(0, arg.train_loop.n_steps, train_step, state)


if __name__ == "__main__":
    arg = Args(sampler_init=Args.sampler_init(queue_size=100))
    key = jrd.PRNGKey(42)
    sampler_key, value_key = jrd.split(key, 2)

    state = init(jrd.split(sampler_key, arg.n_env), value_key, arg)
    final_state = train(state, arg)

    # TODO: Add report data
    # TODO: Save final_state
    # TODO: Make a rendering in possible
    # TODO: (Low priority) Add CLI for args
