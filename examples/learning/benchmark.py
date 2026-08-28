from typing import Any, Literal

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
import tyro
from algorithms import Transition, q_learning
from flax import struct
from policies import epsilon_greedy
from utils import log_results

from jaxdp.mapping import greedy_map
from jaxdp.mdp import MDP
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.mdp.grid_world import grid_world
from jaxdp.mdp.sampler.mdp import sample_initial, sample_step
from jaxdp.mdp.simple_graph import graph_mdp
from jaxdp.operator import bellman_opt_op

jax.config.update("jax_enable_x64", True)

bellman_optimality = bellman_opt_op


class metrics:
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Metrics Namespace

    Computes and stores training metrics.
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    class State(struct.PyTreeNode):
        """Metrics collected during training"""

        l1: jax.Array  # L1 norm of Q-value change (scalar)
        l2: jax.Array  # L2 norm of Q-value change (scalar)
        linf: jax.Array  # L-infinity norm of Q-value change (scalar)
        bellman_err: jax.Array  # Bellman error (scalar)
        iteration: jax.Array  # Iteration number (scalar)
        ep_return: jax.Array  # Episode returns [n_envs] - NaN if episode not complete
        ep_len: jax.Array  # Episode lengths [n_envs] - NaN if episode not complete
        eval_mean_return: jax.Array  # Mean evaluation return (scalar) - NaN if no eval
        eval_std_return: jax.Array  # Std evaluation return (scalar) - NaN if no eval

    @staticmethod
    def compute(
        prev: "loop.State",
        new: "loop.State",
        args: "loop.Args",
        step: jax.Array,
        dones: jax.Array,
        eval_results: "loop.EvalResult",
    ) -> "metrics.State":
        """Compute metrics for the current iteration

        Args:
            prev: Previous loop state
            new: New loop state
            args: Loop arguments
            step: Current step number
            dones: Boolean array indicating which environments completed episodes [n_envs]
            eval_results: Evaluation results (loop.EvalResult dataclass)

        Returns:
            Metrics state with NaN values where no information is available
        """
        prev_alg = prev.alg_state
        new_alg = new.alg_state

        diff = new_alg.q_vals - prev_alg.q_vals
        l1 = jnp.sum(jnp.abs(diff))
        l2 = jnp.sqrt(jnp.sum(diff**2))
        linf = jnp.max(jnp.abs(diff))

        bellman_target = bellman_optimality.q(args.mdp, prev_alg.q_vals, prev_alg.gamma)
        bellman_err = jnp.max(jnp.abs(prev_alg.q_vals - bellman_target))

        ep_return = jnp.where(dones, new.last_return, jnp.nan)
        ep_len = jnp.where(dones, prev.ep_step, jnp.nan)

        return metrics.State(
            l1=l1,
            l2=l2,
            linf=linf,
            bellman_err=bellman_err,
            iteration=step,
            ep_return=ep_return,
            ep_len=ep_len,
            eval_mean_return=eval_results.mean_return,
            eval_std_return=eval_results.std_return,
        )


class sampler:
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Sampler Namespace

    Provides sampling functions for interacting with MDPs using policies.
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    class StepResult(struct.PyTreeNode):
        """Result from sampling a single step"""

        action: jax.Array  # Action taken (one-hot) [n_actions]
        next_state: jax.Array  # Next state (one-hot) [n_states]
        reward: jax.Array  # Reward received (scalar)
        terminal: jax.Array  # Terminal flag (scalar)
        timeout: jax.Array  # Timeout flag (scalar)
        stepped_state: jax.Array  # Actual next state for continuing (one-hot) [n_states]
        ep_step: jax.Array  # Episode step counter (scalar)

    class EpisodeResult(struct.PyTreeNode):
        """Result from sampling a complete episode"""

        total_return: jax.Array  # Total episode return (scalar)
        episode_length: jax.Array  # Episode length (scalar)

    @staticmethod
    def step(
        mdp: MDP,
        policy: jax.Array,
        mdp_state: jax.Array,
        ep_step: jax.Array,
        max_ep_len: int,
        key: chex.PRNGKey,
    ) -> "sampler.StepResult":
        """
        Sample a single step from the MDP using the given policy.

        Args:
            mdp: The MDP to sample from
            policy: Policy to use for action selection [n_actions, n_states]
            mdp_state: Current MDP state (one-hot) [n_states]
            ep_step: Current episode step counter (scalar)
            max_ep_len: Maximum episode length
            key: Random key for sampling

        Returns:
            StepResult containing action, next state, reward, flags, and updated state
        """
        data, stepped_state, new_ep_step = sample_step(
            key,
            mdp_state,
            ep_step,
            policy,
            mdp,
            max_ep_len,
        )

        return sampler.StepResult(
            action=data.action,
            next_state=data.next_state,
            reward=data.reward,
            terminal=data.terminal,
            timeout=data.timeout,
            stepped_state=stepped_state,
            ep_step=new_ep_step,
        )

    @staticmethod
    def step_batch(
        mdp: MDP,
        policy: jax.Array,
        mdp_states: jax.Array,
        ep_steps: jax.Array,
        max_ep_len: int,
        keys: chex.PRNGKey,
    ) -> "sampler.StepResult":
        """
        Sample a single step for a batch of environments.

        Args:
            mdp: The MDP to sample from
            policy: Policy to use for action selection [n_actions, n_states]
            mdp_states: Current MDP states [n_envs, n_states]
            ep_steps: Current episode step counters [n_envs]
            max_ep_len: Maximum episode length
            keys: Random keys for sampling [n_envs]

        Returns:
            StepResult with batch dimension for all parallel environments
        """
        vmap_sample = jax.vmap(
            lambda s, ep, k: sampler.step(mdp, policy, s, ep, max_ep_len, k), in_axes=(0, 0, 0)
        )
        results = vmap_sample(mdp_states, ep_steps, keys)

        return results

    @staticmethod
    def episode(
        mdp: MDP, policy: jax.Array, max_ep_len: int, key: chex.PRNGKey
    ) -> "sampler.EpisodeResult":
        """
        Sample a complete episode from the MDP using the given policy.

        Args:
            mdp: The MDP to sample from
            policy: Policy to use for action selection [n_actions, n_states]
            max_ep_len: Maximum episode length
            key: Random key for sampling

        Returns:
            EpisodeResult containing total return and episode length
        """
        mdp_state = sample_initial(key, mdp)
        ep_step = jnp.array(0.0)
        ep_return = jnp.array(0.0)
        done = jnp.array(False)

        def step_fn(carry, _):
            mdp_state, ep_step, ep_return, done = carry
            result = sampler.step(mdp, policy, mdp_state, ep_step, max_ep_len, key)
            new_return = jnp.where(done, ep_return, ep_return + result.reward)
            new_done = done | (result.terminal + result.timeout > 0)
            return (result.stepped_state, result.ep_step, new_return, new_done), None

        (final_mdp_state, final_ep_step, final_return, final_done), _ = jax.lax.scan(
            step_fn, (mdp_state, ep_step, ep_return, done), None, length=max_ep_len
        )

        return sampler.EpisodeResult(total_return=final_return, episode_length=final_ep_step)


class loop:
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Training Loop for Value-based RL Algorithms

    Provides init(), train(), and evaluate() functions for managing the
    training loop state and running value-based RL algorithms with various
    exploration policies.
    ◈─────────────────────────────────────────────────────────────────────────◈
    """

    class State(struct.PyTreeNode):
        """Training loop state - manages MDP interaction and episode tracking"""

        alg_state: Any  # q_learning.State
        policy_state: Any  # epsilon_greedy.State or soft_policy.State, etc.
        mdp_state: jax.Array  # MDP states [n_envs, n_states]
        ep_step: jax.Array  # Episode step counters [n_envs]
        ep_return: jax.Array  # Current episode returns [n_envs]
        last_return: jax.Array  # Last completed episode returns [n_envs]

    class Args(struct.PyTreeNode):
        """Training loop arguments - static configuration"""

        value_fn: Any  # Algorithm namespace (e.g., q_learning)
        policy_ns: Any  # Policy namespace (e.g., epsilon_greedy)
        mdp: MDP
        seed: int
        n_steps: int
        max_ep_len: int
        n_envs: int
        eval_period: int = 0  # Evaluate every N steps (0 = no evaluation)
        n_eval_episodes: int = 10  # Number of episodes for evaluation
        eval_seed: int = 42  # Seed for evaluation

    class EvalResult(struct.PyTreeNode):
        """Evaluation results"""

        mean_return: jax.Array  # Mean return across eval episodes (scalar)
        std_return: jax.Array  # Std return across eval episodes (scalar)
        mean_length: jax.Array  # Mean episode length (scalar)
        std_length: jax.Array  # Std episode length (scalar)

    @staticmethod
    def init(alg_state: Any, policy_state: Any, args: "loop.Args") -> "loop.State":
        """Initialize loop state with algorithm and policy states

        Args:
            alg_state: Initialized algorithm state (e.g., q_learning.State)
            policy_state: Initialized policy state (e.g., epsilon_greedy.State)
            args: Loop arguments (includes mdp, seed, n_steps, n_envs, etc.)

        Returns:
            Initialized loop state with environment states and episode tracking
        """
        key = jrd.PRNGKey(args.seed)

        env_keys = jrd.split(key, args.n_envs)
        mdp_state = jax.vmap(sample_initial, in_axes=(0, None))(env_keys, args.mdp)
        ep_step = jnp.zeros(args.n_envs)
        ep_return = jnp.zeros(args.n_envs)
        last_return = jnp.zeros(args.n_envs)

        return loop.State(
            alg_state=alg_state,
            policy_state=policy_state,
            mdp_state=mdp_state,
            ep_step=ep_step,
            ep_return=ep_return,
            last_return=last_return,
        )

    @staticmethod
    def train(state: "loop.State", args: "loop.Args") -> tuple["loop.State", Any]:
        """Run training loop for n_steps with specified value function and policy

        Args:
            state: Initial loop state
            args: Loop arguments (includes value_fn, policy_ns, mdp, n_steps, etc.)

        Returns:
            Final loop state and all metrics collected during training
        """

        def step_fn(state: loop.State, step_and_keys):
            step, keys = step_and_keys
            prev = state

            policy = args.policy_ns.get_policy(state.alg_state.q_vals, state.policy_state)

            sample_results = sampler.step_batch(
                args.mdp, policy, state.mdp_state, state.ep_step, args.max_ep_len, keys
            )

            transitions = Transition(
                state=state.mdp_state,
                action=sample_results.action,
                reward=sample_results.reward,
                next_state=sample_results.next_state,
                terminal=sample_results.terminal,
            )

            new_alg_state = args.value_fn.batch_update(state.alg_state, transitions)

            dones = sample_results.terminal + sample_results.timeout > 0
            any_done = jnp.any(dones)
            new_policy_state = args.policy_ns.update(state.policy_state, any_done)

            new_returns = state.ep_return + sample_results.reward
            last_returns = jnp.where(dones, new_returns, state.last_return)
            ep_returns = jnp.where(dones, 0.0, new_returns)

            new_state = state.replace(
                alg_state=new_alg_state,
                policy_state=new_policy_state,
                mdp_state=sample_results.stepped_state,
                ep_step=sample_results.ep_step,
                ep_return=ep_returns,
                last_return=last_returns,
            )

            should_eval = (args.eval_period > 0) & ((step + 1) % args.eval_period == 0)
            eval_results = jax.lax.cond(
                should_eval,
                lambda s: loop.evaluate(s, args),
                lambda s: loop.EvalResult(
                    mean_return=jnp.asarray(jnp.nan),
                    std_return=jnp.asarray(jnp.nan),
                    mean_length=jnp.asarray(jnp.nan),
                    std_length=jnp.asarray(jnp.nan),
                ),
                new_state,
            )

            metrics_state = metrics.compute(prev, new_state, args, step, dones, eval_results)
            return new_state, metrics_state

        keys = jrd.split(jrd.PRNGKey(args.seed), args.n_steps * args.n_envs)
        keys = keys.reshape(args.n_steps, args.n_envs, -1)
        steps_and_keys = (jnp.arange(args.n_steps), keys)

        run_scan = chex.chexify(
            lambda loop_state, inputs: jax.lax.scan(step_fn, loop_state, inputs),
            async_check=False,
        )
        return run_scan(state, steps_and_keys)

    @staticmethod
    def evaluate(state: "loop.State", args: "loop.Args") -> "loop.EvalResult":
        """Evaluate the learned policy (greedy, no exploration)

        Args:
            state: Current loop state with learned Q-values
            args: Loop arguments (includes mdp, max_ep_len, n_eval_episodes, eval_seed)

        Returns:
            loop.EvalResult dataclass with evaluation statistics
        """
        policy = greedy_map.q(state.alg_state.q_vals)

        def run_episode(key):
            result = sampler.episode(args.mdp, policy, args.max_ep_len, key)
            return result.total_return, result.episode_length

        keys = jrd.split(jrd.PRNGKey(args.eval_seed), args.n_eval_episodes)
        returns, lengths = jax.vmap(run_episode)(keys)

        return loop.EvalResult(
            mean_return=jnp.mean(returns),
            std_return=jnp.std(returns),
            mean_length=jnp.mean(lengths),
            std_length=jnp.std(lengths),
        )


def grid_mdp_factory() -> MDP:
    """
    Create a GridWorld MDP with fixed 5x5 layout.

    Returns:
        GridWorld MDP with player at (1,3), goal at (1,3), and wall at (2,2)
    """
    board = ["#####", "#  @#", "# #X#", "#P  #", "#####"]
    return grid_world(board=board, p_slip=0.0)


def garnet_mdp_factory(
    key: chex.PRNGKey, state_size: int, action_size: int, branch_size: int
) -> MDP:
    """
    Create a random Garnet MDP.

    Args:
        key: Random key for MDP generation
        state_size: Number of states in the MDP
        action_size: Number of actions available
        branch_size: Branching factor for transitions

    Returns:
        Randomly generated Garnet MDP
    """
    return garnet_mdp(
        state_size=state_size, action_size=action_size, branch_size=branch_size, key=key
    )


def graph_mdp_factory() -> MDP:
    """
    Create a simple graph-based MDP.

    Returns:
        Graph MDP with predefined structure
    """
    return graph_mdp()


def q_learning_parallel_envs():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning with Parallel Environments
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = grid_mdp_factory()
    alg_name = "Q-Learning (Parallel Envs)"

    args = loop.Args(
        value_fn=q_learning,
        policy_ns=epsilon_greedy,
        mdp=mdp,
        seed=0,
        n_steps=5000,
        max_ep_len=50,
        n_envs=16,
        eval_period=1000,
        n_eval_episodes=10,
        eval_seed=42,
    )

    key = jrd.PRNGKey(args.seed)
    alg_state = q_learning.init(mdp, key, gamma=0.99, alpha=0.5)
    policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.997, eps_min=0.1)

    state = loop.init(alg_state, policy_state, args)

    final_state, all_metrics = loop.train(state, args)

    results = {"GridWorld": (all_metrics, final_state.alg_state.q_vals)}
    log_results(results, alg_name)
    return final_state.alg_state.q_vals


def q_learning_multi_seed():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning Multi-Seed (Parallel Independent Runs)
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = grid_mdp_factory()
    alg_name = "Q-Learning (Multi-Seed)"
    n_seeds = 5

    def run_one_seed(seed):
        args = loop.Args(
            value_fn=q_learning,
            policy_ns=epsilon_greedy,
            mdp=mdp,
            seed=seed,
            n_steps=5000,
            max_ep_len=50,
            n_envs=1,
        )

        key = jrd.PRNGKey(seed)
        alg_state = q_learning.init(mdp, key, gamma=0.99, alpha=0.5)
        policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.997, eps_min=0.1)

        state = loop.init(alg_state, policy_state, args)

        final_state, all_metrics = loop.train(state, args)
        return final_state, all_metrics

    seeds = jnp.arange(n_seeds)
    vmap_run = jax.vmap(run_one_seed)
    final_states, all_metrics = vmap_run(seeds)

    avg_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), all_metrics)
    avg_q = jnp.mean(final_states.alg_state.q_vals, axis=0)

    results = {"GridWorld": (avg_metrics, avg_q)}
    log_results(results, alg_name)
    return results


def q_learning_grid_world():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning GridWorld
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = grid_mdp_factory()
    alg_name = "Q-Learning"

    args = loop.Args(
        value_fn=q_learning,
        policy_ns=epsilon_greedy,
        mdp=mdp,
        seed=0,
        n_steps=5000,
        max_ep_len=50,
        n_envs=1,
        eval_period=1000,
        n_eval_episodes=10,
        eval_seed=42,
    )

    key = jrd.PRNGKey(args.seed)
    alg_state = q_learning.init(mdp, key, gamma=0.99, alpha=0.5)
    policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.997, eps_min=0.1)

    state = loop.init(alg_state, policy_state, args)

    final_state, all_metrics = loop.train(state, args)

    results = {"GridWorld": (all_metrics, final_state.alg_state.q_vals)}
    log_results(results, alg_name)
    return final_state.alg_state.q_vals


def q_learning_garnet():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning Garnet MDP
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = garnet_mdp_factory(jrd.PRNGKey(42), state_size=10, action_size=4, branch_size=2)
    alg_name = "Q-Learning"

    args = loop.Args(
        value_fn=q_learning,
        policy_ns=epsilon_greedy,
        mdp=mdp,
        seed=0,
        n_steps=30000,
        max_ep_len=50,
        n_envs=1,
    )

    key = jrd.PRNGKey(args.seed)
    alg_state = q_learning.init(mdp, key, gamma=0.99, alpha=0.3)
    policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.9995, eps_min=0.05)

    state = loop.init(alg_state, policy_state, args)

    final_state, all_metrics = loop.train(state, args)

    results = {"GarnetMDP": (all_metrics, final_state.alg_state.q_vals)}
    log_results(results, alg_name)
    return final_state.alg_state.q_vals


def q_learning_graph():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning Graph MDP
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    mdp = graph_mdp_factory()
    alg_name = "Q-Learning"

    args = loop.Args(
        value_fn=q_learning,
        policy_ns=epsilon_greedy,
        mdp=mdp,
        seed=0,
        n_steps=40000,
        max_ep_len=50,
        n_envs=1,
    )

    key = jrd.PRNGKey(args.seed)
    alg_state = q_learning.init(mdp, key, gamma=0.99, alpha=0.5)
    policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.9996, eps_min=0.05)

    state = loop.init(alg_state, policy_state, args)

    final_state, all_metrics = loop.train(state, args)

    results = {"GraphMDP": (all_metrics, final_state.alg_state.q_vals)}
    log_results(results, alg_name)
    return final_state.alg_state.q_vals


def q_learning_benchmark():
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Q-Learning Comprehensive Benchmark
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    alg_name = "Q-Learning"

    @struct.dataclass
    class MDPConfig:
        name: str
        mdp: MDP
        n_steps: int
        alpha: float
        epsilon: float
        eps_decay: float
        eps_min: float

    configs = [
        MDPConfig(
            name="GridWorld",
            mdp=grid_mdp_factory(),
            n_steps=5000,
            alpha=0.5,
            epsilon=1.0,
            eps_decay=0.997,
            eps_min=0.1,
        ),
        MDPConfig(
            name="GarnetMDP",
            mdp=garnet_mdp_factory(jrd.PRNGKey(42), state_size=10, action_size=4, branch_size=2),
            n_steps=30000,
            alpha=0.3,
            epsilon=1.0,
            eps_decay=0.9995,
            eps_min=0.05,
        ),
        MDPConfig(
            name="GraphMDP",
            mdp=graph_mdp_factory(),
            n_steps=40000,
            alpha=0.5,
            epsilon=1.0,
            eps_decay=0.9996,
            eps_min=0.05,
        ),
    ]

    results = {}
    for config in configs:
        args = loop.Args(
            value_fn=q_learning,
            policy_ns=epsilon_greedy,
            mdp=config.mdp,
            seed=0,
            n_steps=config.n_steps,
            max_ep_len=50,
            n_envs=1,
        )

        key = jrd.PRNGKey(args.seed)
        alg_state = q_learning.init(config.mdp, key, gamma=0.99, alpha=config.alpha)
        policy_state = epsilon_greedy.init(
            epsilon=config.epsilon, eps_decay=config.eps_decay, eps_min=config.eps_min
        )

        state = loop.init(alg_state, policy_state, args)

        final_state, all_metrics = loop.train(state, args)

        results[config.name] = (all_metrics, final_state.alg_state.q_vals)

    log_results(results, alg_name)
    return results


Benchmark = Literal[
    "q_learning",
    "parallel_envs",
    "multi_seed",
    "q_learning_garnet",
    "q_learning_graph",
    "benchmark",
]


def main(benchmark_type: Benchmark, /) -> None:
    """Run a learning benchmark."""
    if benchmark_type == "q_learning":
        q_learning_grid_world()
    elif benchmark_type == "parallel_envs":
        q_learning_parallel_envs()
    elif benchmark_type == "multi_seed":
        q_learning_multi_seed()
    elif benchmark_type == "q_learning_garnet":
        q_learning_garnet()
    elif benchmark_type == "q_learning_graph":
        q_learning_graph()
    elif benchmark_type == "benchmark":
        q_learning_benchmark()


if __name__ == "__main__":
    tyro.cli(main)
