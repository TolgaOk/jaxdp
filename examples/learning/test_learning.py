import unittest

import jax.numpy as jnp
import jax.random as jrd

from jaxdp.mdp.grid_world import grid_world

from algorithms import Transition, q_learning
from policies import epsilon_greedy, soft_policy
from benchmark import loop, metrics


class TestQLearningAlgorithm(unittest.TestCase):
    """Test q_learning namespace based on function signatures."""

    def setUp(self):
        board = ["#####", "#  @#", "# #X#", "#P  #", "#####"]
        self.mdp = grid_world(board=board, p_slip=0.0)
        self.key = jrd.PRNGKey(0)

    def test_init_creates_state(self):
        """Test that init() creates a valid q_learning.State."""
        gamma = 0.99
        alpha = 0.1
        state = q_learning.init(self.mdp, self.key, gamma, alpha)

        self.assertTrue(hasattr(state, "q_vals"))
        self.assertTrue(hasattr(state, "gamma"))
        self.assertTrue(hasattr(state, "alpha"))
        self.assertEqual(state.q_vals.shape, (self.mdp.action_size, self.mdp.state_size))
        self.assertEqual(float(state.gamma), gamma)
        self.assertEqual(float(state.alpha), alpha)

    def test_init_with_custom_q_values(self):
        """Test that init_q parameter sets initial Q-values."""
        init_q = 5.0
        state = q_learning.init(self.mdp, self.key, gamma=0.99, alpha=0.1, init_q=init_q)

        self.assertTrue(jnp.allclose(state.q_vals, init_q))

    def test_compute_delta_returns_correct_shape(self):
        """Test that _compute_delta returns delta with correct shape."""
        state = q_learning.init(self.mdp, self.key, gamma=0.99, alpha=0.1)

        s = jnp.zeros(self.mdp.state_size).at[0].set(1.0)
        a = jnp.zeros(self.mdp.action_size).at[0].set(1.0)
        s_next = jnp.zeros(self.mdp.state_size).at[1].set(1.0)

        transition = Transition(
            state=s,
            action=a,
            reward=jnp.array(1.0),
            next_state=s_next,
            terminal=jnp.array(0.0),
        )

        delta = q_learning._compute_delta(state, transition)

        self.assertEqual(delta.shape, (self.mdp.action_size, self.mdp.state_size))

    def test_compute_delta_without_alpha(self):
        """Test that _compute_delta returns unscaled delta (no alpha multiplication)."""
        state = q_learning.init(self.mdp, self.key, gamma=0.99, alpha=0.5)

        s = jnp.zeros(self.mdp.state_size).at[0].set(1.0)
        a = jnp.zeros(self.mdp.action_size).at[0].set(1.0)
        s_next = jnp.zeros(self.mdp.state_size).at[1].set(1.0)

        transition = Transition(
            state=s,
            action=a,
            reward=jnp.array(1.0),
            next_state=s_next,
            terminal=jnp.array(1.0),
        )

        delta = q_learning._compute_delta(state, transition)

        expected_delta = jnp.zeros((self.mdp.action_size, self.mdp.state_size))
        expected_delta = expected_delta.at[0, 0].set(1.0)

        self.assertTrue(jnp.allclose(delta, expected_delta))

    def test_update_applies_alpha(self):
        """Test that update() applies alpha to delta."""
        alpha = 0.5
        state = q_learning.init(self.mdp, self.key, gamma=0.99, alpha=alpha)
        initial_q = state.q_vals.copy()

        s = jnp.zeros(self.mdp.state_size).at[0].set(1.0)
        a = jnp.zeros(self.mdp.action_size).at[0].set(1.0)
        s_next = jnp.zeros(self.mdp.state_size).at[1].set(1.0)

        transition = Transition(
            state=s,
            action=a,
            reward=jnp.array(1.0),
            next_state=s_next,
            terminal=jnp.array(1.0),
        )

        updated_state = q_learning.update(state, transition)

        q_change = updated_state.q_vals[0, 0] - initial_q[0, 0]
        self.assertTrue(jnp.isclose(q_change, alpha * 1.0))

    def test_batch_update_normalizes_by_occurrence(self):
        """Test that batch_update() divides by per-(s,a) occurrence count."""
        state = q_learning.init(self.mdp, self.key, gamma=0.99, alpha=0.1)

        s = jnp.zeros(self.mdp.state_size).at[0].set(1.0)
        a = jnp.zeros(self.mdp.action_size).at[0].set(1.0)
        s_next = jnp.zeros(self.mdp.state_size).at[1].set(1.0)

        transitions = Transition(
            state=jnp.stack([s, s]),
            action=jnp.stack([a, a]),
            reward=jnp.array([1.0, 1.0]),
            next_state=jnp.stack([s_next, s_next]),
            terminal=jnp.array([1.0, 1.0]),
        )

        updated_state = q_learning.batch_update(state, transitions)

        single_updated = q_learning.update(
            state,
            Transition(
                state=s,
                action=a,
                reward=jnp.array(1.0),
                next_state=s_next,
                terminal=jnp.array(1.0),
            ),
        )

        self.assertTrue(jnp.allclose(updated_state.q_vals, single_updated.q_vals))


class TestEpsilonGreedyPolicy(unittest.TestCase):
    """Test epsilon_greedy namespace based on function signatures."""

    def test_init_creates_state(self):
        """Test that init() creates a valid epsilon_greedy.State."""
        epsilon = 0.9
        eps_decay = 0.995
        eps_min = 0.01
        state = epsilon_greedy.init(epsilon, eps_decay, eps_min)

        self.assertTrue(hasattr(state, "epsilon"))
        self.assertTrue(hasattr(state, "eps_decay"))
        self.assertTrue(hasattr(state, "eps_min"))
        self.assertEqual(float(state.epsilon), epsilon)
        self.assertEqual(float(state.eps_decay), eps_decay)
        self.assertEqual(float(state.eps_min), eps_min)

    def test_update_decays_epsilon_when_done(self):
        """Test that update() decays epsilon when episode is done."""
        state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.9, eps_min=0.1)
        done = jnp.array(1.0)

        updated_state = epsilon_greedy.update(state, done)

        self.assertEqual(float(updated_state.epsilon), 0.9)

    def test_update_does_not_decay_when_not_done(self):
        """Test that update() does not decay epsilon when episode continues."""
        state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.9, eps_min=0.1)
        done = jnp.array(0.0)

        updated_state = epsilon_greedy.update(state, done)

        self.assertEqual(float(updated_state.epsilon), 1.0)

    def test_update_respects_minimum_epsilon(self):
        """Test that update() does not decay below eps_min."""
        state = epsilon_greedy.init(epsilon=0.11, eps_decay=0.9, eps_min=0.1)
        done = jnp.array(1.0)

        updated_state = epsilon_greedy.update(state, done)

        self.assertEqual(float(updated_state.epsilon), 0.1)

    def test_get_policy_returns_policy_matrix(self):
        """Test that get_policy() returns a policy of correct shape."""
        q_vals = jnp.array([[1.0, 2.0], [3.0, 0.5]])
        state = epsilon_greedy.init(epsilon=0.1, eps_decay=0.995, eps_min=0.01)

        policy = epsilon_greedy.get_policy(q_vals, state)

        self.assertEqual(policy.shape, q_vals.shape)
        self.assertTrue(jnp.allclose(jnp.sum(policy, axis=0), 1.0))


class TestSoftPolicy(unittest.TestCase):
    """Test soft_policy namespace based on function signatures."""

    def test_init_creates_state(self):
        """Test that init() creates a valid soft_policy.State."""
        temperature = 2.0
        temp_decay = 0.99
        temp_min = 0.1
        state = soft_policy.init(temperature, temp_decay, temp_min)

        self.assertTrue(hasattr(state, "temperature"))
        self.assertTrue(hasattr(state, "temp_decay"))
        self.assertTrue(hasattr(state, "temp_min"))
        self.assertEqual(float(state.temperature), temperature)
        self.assertEqual(float(state.temp_decay), temp_decay)
        self.assertEqual(float(state.temp_min), temp_min)

    def test_update_decays_temperature_when_done(self):
        """Test that update() decays temperature when episode is done."""
        state = soft_policy.init(temperature=1.0, temp_decay=0.9, temp_min=0.1)
        done = jnp.array(1.0)

        updated_state = soft_policy.update(state, done)

        self.assertEqual(float(updated_state.temperature), 0.9)

    def test_update_does_not_decay_when_not_done(self):
        """Test that update() does not decay temperature when episode continues."""
        state = soft_policy.init(temperature=1.0, temp_decay=0.9, temp_min=0.1)
        done = jnp.array(0.0)

        updated_state = soft_policy.update(state, done)

        self.assertEqual(float(updated_state.temperature), 1.0)

    def test_get_policy_returns_softmax(self):
        """Test that get_policy() returns softmax policy."""
        q_vals = jnp.array([[1.0, 2.0], [3.0, 0.5]])
        state = soft_policy.init(temperature=1.0, temp_decay=0.99, temp_min=0.1)

        policy = soft_policy.get_policy(q_vals, state)

        self.assertEqual(policy.shape, q_vals.shape)
        self.assertTrue(jnp.allclose(jnp.sum(policy, axis=0), 1.0))
        self.assertTrue(jnp.all(policy > 0))


class TestLoopNamespace(unittest.TestCase):
    """Test loop namespace based on function signatures."""

    def setUp(self):
        board = ["#####", "#  @#", "# #X#", "#P  #", "#####"]
        self.mdp = grid_world(board=board, p_slip=0.0)
        self.key = jrd.PRNGKey(0)
        self.alg_state = q_learning.init(self.mdp, self.key, gamma=0.99, alpha=0.1)
        self.policy_state = epsilon_greedy.init(epsilon=1.0, eps_decay=0.997, eps_min=0.1)

    def test_init_creates_loop_state(self):
        """
        Test that loop.init() creates a valid loop.State from function signature.

        Note: loop.init() requires an Args object, not individual arguments.
        This reveals an unintuitive part of the API where you must construct
        loop.Args before calling init().
        """
        n_envs = 4
        args = loop.Args(
            value_fn=q_learning,
            policy_ns=epsilon_greedy,
            mdp=self.mdp,
            seed=0,
            n_steps=100,
            max_ep_len=100,
            n_envs=n_envs,
        )

        state = loop.init(self.alg_state, self.policy_state, args)

        self.assertTrue(hasattr(state, "alg_state"))
        self.assertTrue(hasattr(state, "policy_state"))
        self.assertTrue(hasattr(state, "mdp_state"))
        self.assertTrue(hasattr(state, "ep_step"))
        self.assertTrue(hasattr(state, "ep_return"))
        self.assertTrue(hasattr(state, "last_return"))


class TestTransitionDataclass(unittest.TestCase):
    """Test Transition protocol dataclass."""

    def test_transition_creation(self):
        """Test that Transition can be created with all fields."""
        transition = Transition(
            state=jnp.array([1.0, 0.0, 0.0, 0.0]),
            action=jnp.array([1.0, 0.0, 0.0, 0.0]),
            reward=jnp.array(1.0),
            next_state=jnp.array([0.0, 1.0, 0.0, 0.0]),
            terminal=jnp.array(0.0),
        )

        self.assertEqual(transition.state.shape, (4,))
        self.assertEqual(transition.action.shape, (4,))
        self.assertEqual(transition.reward.shape, ())
        self.assertEqual(transition.next_state.shape, (4,))
        self.assertEqual(transition.terminal.shape, ())

    def test_transition_batch(self):
        """Test that Transition supports batch dimensions."""
        batch_size = 8
        state_size = 4
        action_size = 4

        transitions = Transition(
            state=jnp.zeros((batch_size, state_size)),
            action=jnp.zeros((batch_size, action_size)),
            reward=jnp.zeros(batch_size),
            next_state=jnp.zeros((batch_size, state_size)),
            terminal=jnp.zeros(batch_size),
        )

        self.assertEqual(transitions.state.shape, (batch_size, state_size))
        self.assertEqual(transitions.action.shape, (batch_size, action_size))
        self.assertEqual(transitions.reward.shape, (batch_size,))


if __name__ == "__main__":
    unittest.main()
