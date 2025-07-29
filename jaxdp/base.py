import chex
import distrax
import jax
import jax.numpy as jnp
import jax.random as jrd

from jaxdp.mdp.mdp import MDP
from jaxdp.typehints import F, PiType, QType, StaticMeta, VType


class greedy_policy(metaclass=StaticMeta):

    def q(value: QType) -> PiType:
        """
        Greedy policy distribution from Q values.

        Args:
            value (QType): Q Value array

        Returns:
            PiType: Policy distribution as one hot vectors

        """
        return jax.nn.one_hot(jnp.argmax(value, axis=0, keepdims=False),
                              num_classes=value.shape[0],
                              axis=0)

    def v(mdp: MDP, value: VType, gamma: float) -> PiType:
        """
        Greedy policy distribution from state values.

        Args:
            mdp (MDP): Markov Decision Process
            value (VType): State value array
            gamma (float): Discount factor

        Returns:
            PiType: Policy distribution as one hot vectors

        """
        # TODO: Add test
        return greedy_policy.q(to_state_action_value(mdp, value, gamma))


class soft_policy(metaclass=StaticMeta):

    def q(value: QType, temperature: float) -> PiType:
        r"""
        Softmax policy distribution.

        Args:
            value (QType): Q Value array
            temperature (float): Temperature parameter of softmax. Higher values will result in
                uniform policy distribution while lower values will result a distribution closer
                to greedy policy.

        Returns:
            PiType: Policy distribution

        """
        return jax.nn.softmax(value / temperature, axis=0)

    def v(value: VType, temperature: float) -> PiType:
        raise NotImplementedError


class e_greedy_policy(metaclass=StaticMeta):

    def q(value: QType, epsilon: float) -> PiType:
        r"""
        Epsilon greedy policy distribution.

        Args:
            value (QType): Q Value array
            epsilon (float): Epsilon parameter. The policy takes the greedy action with
                (1 - epsilon + epsilon/|A|) probability while take a non-greedy action with
                (epsilon / |A|) probability where |A| is the dimension of the action space.

        Returns:
            PiType: Policy distribution

        """
        greedy_policy.p = greedy_policy.q(value)
        return greedy_policy.p * (1 - epsilon) + jnp.ones_like(value) * (epsilon / value.shape[0])

    def v(value: VType, epsilon: float, ) -> PiType:
        raise NotImplementedError


class expected_value(metaclass=StaticMeta):

    def q(mdp: MDP, value: QType) -> chex.Scalar:
        r"""
        Expected value of the state-action values (Q) over initial distribution.

        .. math::
            \underset{\substack{s_0 \sim \rho \\ a \sim \pi^Q}}{\mathbb{E}} [Q(S, A)]

        Args:
            mdp (MDP): Markov Decision Process
            value (QType): Q Value array

        Returns:
            Scalar: Expected value

        """
        return expected_value.v(mdp, jnp.max(value, axis=0))

    def v(mdp: MDP, value: VType) -> chex.Scalar:
        r"""
        Expected value of the state-values over initial distribution.

        .. math::
            \underset{s_0 \sim \rho}{\mathbb{E}} [V(S)]

        Args:
            mdp (MDP): Markov Decision Process
            value (VType): Value array

        Returns:
            Scalar: Expected value

        """
        return (value * mdp.initial).sum()


class policy_evaluation(metaclass=StaticMeta):

    def q(mdp: MDP, policy: PiType, gamma: float) -> QType:
        r"""
        Evaluate the policy for each state-action pair using the true MDP

        .. math::
            \\eta(\\pi)(s_i, a_j) = \\big[r^{a_j} + \\gamma P^\\pi
            (\\mathrm{I}_n - \\gamma P^\\pi)r^\\pi\\big]_i

        Args:
            mdp (MDP): Markov Decision Process
            policy (PiType): Policy distribution
            gamma (float): Discount factor

        Returns:
            QType: Cumulative discounted reward of each state-action pair

        """
        mc_state_values = policy_evaluation.v(mdp, policy, gamma)
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        return (reward + gamma * jnp.einsum("axs,x->as", mdp.transition, mc_state_values))

    def v(mdp: MDP, policy: PiType, gamma: float) -> VType:
        r"""
        Evaluate the policy for each state using the true MDP

        .. math::
            \eta(\pi)(s_i) = \big[(\mathrm{I}_n - \gamma P^\pi)r^\pi\big]_i

        Args:
            mdp (MDP): Markov Decision Process
            policy (PiType): Policy distribution
            gamma (float): Discount factor

        Returns:
            VType: Cumulative discounted reward of each state

        """
        transition_pi, reward_pi = _markov_chain_pi(mdp, policy)

        return (jnp.linalg.inv(jnp.eye(mdp.state_size) - gamma * transition_pi.T) @
                jnp.einsum("sx,sx->s", transition_pi.T, reward_pi))


class bellman_operator(metaclass=StaticMeta):

    def q(mdp: MDP, policy: PiType, value: QType, gamma: float) -> QType:
        r"""
        Evaluate the Bellman policy operator for each state-action pair

        .. math::
            \\mathcal{B}(Q)(s_i, a_j) = \\big[r^{a_j} + \\gamma
                \\underset{s^+ \\sim P^{a_j}}{\\mathbb{E}}[Q(s^+, a_j)]\\big]_i - Q(s_i, a_j))

        Args:
            mdp (MDP): Markov Decision Process
            policy (PiType): Policy distribution
            value (QType): Q Value array
            gamma (float): Discount factor

        Returns:
            QType: Target value

        """
        target_values = jnp.einsum("axs,ux,ux,x->as",
                                   mdp.transition, value, policy, (1 - mdp.terminal))
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        return reward + gamma * target_values

    def v(mdp: MDP, policy: PiType, value: VType, gamma: float) -> VType:
        r"""
        Bellman policy operator for state values.

        .. math::
            \mathcal{T}^\pi(V)(s) = \sum_a \pi(a|s) \left[ r(s, a) + \gamma \sum_{s'} P(s'|s, a) V(s') \right]

        Args:
            mdp (MDP): Markov Decision Process
            policy (PiType): Policy distribution
            value (VType): State value array
            gamma (float): Discount factor

        Returns:
            VType: Updated state values after applying Bellman policy operator

        """
        # TODO: Add test
        target_values = jnp.einsum("axs,x,x->as",
                                   mdp.transition, value, (1 - mdp.terminal))
        reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        return jnp.einsum("as,as->s", reward + gamma * target_values, policy)


class bellman_optimality_operator(metaclass=StaticMeta):

    def q(mdp: MDP, value: QType, gamma: float) -> QType:
        r"""
        Bellman optimality operator for Q-values.

        .. math::
            \mathcal{T}^*(Q)(s, a) = r(s, a) + \gamma \sum_{s'} P(s'|s, a) \max_{a'} Q(s', a')

        Args:
            mdp (MDP): Markov Decision Process
            value (QType): Q Value array
            gamma (float): Discount factor

        Returns:
            QType: Updated Q-values after applying Bellman optimality operator

        """
        # TODO: Add test
        target_values = jnp.einsum("axs,x->as", mdp.transition,
                                   jnp.max(value, axis=0, keepdims=False))
        rewards = jnp.einsum("axs,asx->as", mdp.transition, mdp.reward)
        return rewards + gamma * target_values


class stationary_distribution(metaclass=StaticMeta):

    def q(mdp: MDP, policy: PiType, iterations: int = 10) -> F["AS"]:
        """
        Compute the stationary distribution of the Markov chain induced by the policy.

        Args:
            mdp (MDP): Markov Decision Process
            policy (PiType): Policy distribution
            iterations (int): Number of iterations for power method

        Returns:
            F["AS"]: Stationary distribution over states
        """
        distribution = jnp.einsum(
            "s,as->as",
            mdp.initial,
            policy)
        return jax.lax.fori_loop(
            0,
            iterations,
            lambda i, d: jnp.einsum(
                "axs,ux,as->ux",
                mdp.transition,
                policy,
                d
            ),
            distribution)

    def v(mdp: MDP, policy: PiType, iterations: int = 10) -> F["S"]:
        raise NotImplementedError


def markov_chain_eigen_values(mdp: MDP, policy: PiType) -> F["S"]:
    r"""
    Eigen values of the Markov Chain of the MDP by fixing the policy.

    Args:
        mdp (MDP): Markov Decision Process
        policy (PiType): Policy distribution

    Returns:
        Array: Eigen values

    """
    # TODO: Add test
    # TODO: Add test
    transition_pi, _ = _markov_chain_pi(mdp, policy)
    return jnp.linalg.eigvals(transition_pi.T)


def to_greedy_state_value(value: QType) -> VType:
    """Convert Q-values to greedy state values by taking the maximum over actions."""
    # TODO: Add test
    return jnp.max(value, axis=0)


def to_state_action_value(mdp: MDP, value: VType, gamma: float) -> QType:
    """Convert state values to Q-values using the MDP dynamics."""
    # TODO: Add test
    return (jnp.einsum("asx,axs->as", mdp.reward, mdp.transition) +
            gamma * jnp.einsum("axs,x->as", mdp.transition, value))


def _markov_chain_pi(mdp: MDP, policy: PiType) -> tuple[F["SS"], F["SS"]]:
    r"""
    Make Markov Chain of an MDP by fixing the policy.

    .. math::
        P^\pi = \underset{a \sim \pi}{\mathbb{E}}[P^a]
        r^\pi = \underset{a \sim \pi}{\mathbb{E}}[r^a]

    Args:
        mdp (MDP): Markov Decision Process
        policy (PiType): Policy distribution

    Returns:
        tuple[chex.Array, chex.Array]: Transition matrix, Reward matrix

    """
    transition_pi = jnp.einsum("as,axs->xs", policy, mdp.transition)
    reward_pi = jnp.einsum("as,asx->sx", policy, mdp.reward)
    return transition_pi, reward_pi


def sample_from(policy: PiType, key: chex.PRNGKey) -> F["AS"]:
    r"""
    Sample from a policy. The samples will be one-hot vectors.

    Args:
        policy (PiType): Policy distribution
        key (chex.PRNGKey): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        Array: Sampled actions in the one-hot vector form for each state.

    """
    return distrax.OneHotCategorical(probs=policy.T, dtype="float").sample(seed=key).T


def sample_based_policy_evaluation(mdp: MDP,
                                   policy: PiType,
                                   key: chex.PRNGKey,
                                   gamma: float,
                                   max_episode_length: int
                                   ) -> chex.Scalar:
    """
    Evaluate policy using sample-based Monte Carlo estimation.

    Args:
        mdp (MDP): Markov Decision Process
        policy (PiType): Policy distribution
        key (chex.PRNGKey): State of the JAX pseudorandom number generators (PRNGs)
        gamma (float): Discount factor
        max_episode_length (int): Maximum length of episode for sampling

    Returns:
        chex.Scalar: Estimated value of the policy

    """
    # TODO: Add test
    episode_step = jnp.zeros((1,))
    state = mdp.initial
    episode_rewards = jnp.full((max_episode_length,), jnp.nan)
    is_terminated = jnp.array(False).astype("bool")

    def _step(index, _data):
        _episode_step, _key, _episode_rewards, _state, _is_terminated = _data
        _key, step_key = jrd.split(_key)
        (act, next_state, reward, terminal, timeout, _state, _episode_step
         ) = async_sample_step_pi(
            mdp, policy, _state, _episode_step, max_episode_length, step_key)
        reward = (1 - _is_terminated) * reward * (gamma ** index)
        _is_terminated = jnp.logical_or(terminal, _is_terminated)
        _episode_rewards = _episode_rewards.at[index].set(reward)

        return _episode_step, _key, _episode_rewards, _state, _is_terminated

    _, _, episode_rewards, _, _ = jax.lax.fori_loop(
        0, max_episode_length, _step,
        (episode_step, key, episode_rewards, state, is_terminated))
    return episode_rewards.sum()


def sync_sample(mdp: MDP, key: chex.PRNGKey) -> tuple[F["AS"], F["ASS"], F["AS"]]:
    r"""
    Synchronously sample starting from each state action pair in the given MDP

    Args:
        mdp (MDP): Markov Decision Process
        key (chex.PRNGKey): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        tuple[chex.Array, chex.Array, chex.Array]: Rewards, Next states, Termination condition

    """
    next_state = distrax.OneHotCategorical(
        probs=jnp.einsum("axs->asx", mdp.transition), dtype="float").sample(seed=key)
    terminal = jnp.einsum("asx,x->as", next_state, mdp.terminal)
    reward = jnp.einsum("asx,asx->as", mdp.reward, next_state)

    return reward, next_state, terminal


def async_sample_step(mdp: MDP,
                      action: F["A"],
                      state: F["S"],
                      episode_step: chex.Scalar,
                      episode_length: int,
                      key: chex.PRNGKey
                      ) -> tuple[F["S"], chex.Scalar, chex.Scalar,
                                 chex.Scalar, F["S"], chex.Scalar]:
    r"""
    Asynchronously sample from the given MDP by following the given action. The starting state
    is given by the <state> argument. Similar to stateless version of the env.step() function
    from "gym", this function iterates the MDP only one step and returns the transition artifacts
    and the stepped MDP states (state & episode_length). The <episode_length> argument limits
    the maximum episode length (artificially). If an episode is terminated by reaching the maximum
    episodic length, this function sets <timeout> value to "True" while <terminal> may leave as
    "False".
        Note that: The terminated MDP state is automatically set to initial state.

    Args:
        mdp (MDP): Markov Decision Process
        action (Array): One hot action
        state (Array): Current state of the MDP
        episode_step (Scalar): Step count of the MDP
        episode_length (int): Maximum allowed episode length
        key (chex.PRNGKey): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        chex.Array: Next states of the transition (not necessarily equal to stepped State)
        chex.Scalar: Rewards of the transition
        chex.Scalar: termination condition (either 0 or 1) of the transition
        chex.Scalar: timeout condition (either 0 or 1) of the transition
        chex.Array: Stepped state
        chex.Scalar: Stepped step count

    """
    state_key, init_key = jrd.split(key, num=2)

    next_state_p = jnp.einsum(
        "a,axs,s->x", action, mdp.transition, state)
    next_state = distrax.OneHotCategorical(
        probs=next_state_p, dtype="float").sample(seed=state_key)
    reward = jnp.einsum("asx,a,s,x->", mdp.reward, action, state, next_state)
    terminal = jnp.einsum("s,s->", mdp.terminal, next_state)

    episode_step = episode_step + 1
    timeout = episode_step >= episode_length
    terminal = jnp.einsum("s,s->", mdp.terminal, next_state)
    done = jnp.logical_or(terminal, timeout)

    init_state = distrax.OneHotCategorical(
        probs=mdp.initial, dtype="float").sample(seed=init_key)
    state = next_state * (1 - done) + init_state * done
    episode_step = episode_step * (1 - done)

    return next_state, reward, terminal, timeout, state, episode_step


def async_sample_step_pi(mdp: MDP,
                         policy: PiType,
                         state: F["S"],
                         episode_step: chex.Scalar,
                         episode_length: int,
                         key: chex.PRNGKey
                         ) -> tuple[F["A"], F["S"], chex.Scalar,
                                    chex.Scalar, chex.Scalar, F["S"], chex.Scalar]:
    r"""
    Asynchronously sample from the given MDP by following the given policy.

    Args:
        mdp (MDP): Markov Decision Process
        policy (PiType): Policy distribution
        state (Array): Current state of the MDP
        episode_step (Scalar): Step count of the MDP
        episode_length (int): Maximum allowed episode length
        key (chex.PRNGKey): State of the JAX pseudorandom number generators (PRNGs)

    Returns:
        chex.Array: Action of the transition
        chex.Array: Next states of the transition (not necessarily equal to stepped State)
        chex.Scalar: Rewards of the transition
        chex.Scalar: termination condition (either 0 or 1) of the transition
        chex.Scalar: timeout condition (either 0 or 1) of the transition
        chex.Array: Stepped state
        chex.Scalar: Stepped step count

    """
    act_key, step_key = jrd.split(key, num=2)
    policy_p = jnp.einsum("as,s->a", policy, state)
    action = sample_from(policy_p, key=act_key)

    return action, *async_sample_step(mdp=mdp,
                                      action=action,
                                      state=state,
                                      episode_step=episode_step,
                                      episode_length=episode_length,
                                      key=step_key)


def sg(array: chex.Array) -> chex.Array:
    """Stop Gradient function"""
    return jax.lax.stop_gradient(array)
