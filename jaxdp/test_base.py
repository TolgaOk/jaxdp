import unittest
import jax.numpy as jnp
import jax.random as jrd

from base import BaseDP, ValueIteration


class TestBaseDP(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_async_sample_with_cyclic_mdp(self) -> None:
        transition = jnp.array([
            [[0, 0, 0, 1],  # 1 -> 2 -> 3 -> 4 -> 1
             [1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0]],
            [[0, 0, 0, 1],  # 1 -> 3 -> 2 -> 4 -> 1
             [0, 0, 1, 0],
             [1, 0, 0, 0],
             [0, 1, 0, 0]]
        ])
        initial = jnp.array([1, 0, 0, 0])
        terminal = jnp.array([0, 0, 0, 0])
        reward = jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

        agent = BaseDP(
            transition=transition,
            initial=initial,
            terminal=terminal,
            reward=reward,
            episode_length=8
        )
        trajectory = agent.async_sample(
            value=jnp.array([
                [1, 1, 1, 1],
                [-1, -1, -1, -1],
            ]),  # Always choose action 0
            length=agent.episode_length,
            n_sample=2,
            key=jrd.PRNGKey(42)
        )
        self.assertTrue(
            jnp.all(jnp.argmax(trajectory["state"][0], axis=-1) == jnp.arange(8) % 4))
        self.assertTrue(
            jnp.all(jnp.argmax(trajectory["next_state"][0], axis=-1) == jnp.arange(1, 9) % 4))

        trajectory = agent.async_sample(
            value=jnp.array([
                [-1, -1, -1, -1],
                [1, 1, 1, 1],
            ]),  # Always choose action 1
            length=agent.episode_length,
            n_sample=2,
            key=jrd.PRNGKey(42)
        )

        self.assertTrue(
            jnp.all(jnp.argmax(trajectory["state"][0], axis=-1) ==
                    jnp.array([0, 2, 1, 3, 0, 2, 1, 3])))
        self.assertTrue(
            jnp.all(jnp.argmax(trajectory["next_state"][0], axis=-1) ==
                    jnp.array([2, 1, 3, 0, 2, 1, 3, 0])))

    def test_async_sample_reward_terminal_and_timeout(self) -> None:
        transition = jnp.array([
            [[0, 0, 0],
             [1, 1, 0],
             [0, 0, 1]],
            [[0, 0, 0],
             [0, 1, 0],
             [1, 0, 1]]
        ])
        initial = jnp.array([1, 0, 0])
        terminal = jnp.array([0, 0, 1])
        reward = jnp.array([
            [0, -1, -10],
            [0, 1, 10],
        ])

        agent = BaseDP(
            transition=transition,
            initial=initial,
            terminal=terminal,
            reward=reward,
            episode_length=5
        )
        trajectory = agent.async_sample(
            value=jnp.array([
                [1, 1, 1],
                [-1, -1, -1],
            ]),  # Always choose action 0
            length=agent.episode_length,
            n_sample=1,
            key=jrd.PRNGKey(42)
        )

        self.assertTrue(
            jnp.all(jnp.argmax(trajectory["state"][0], axis=-1) == jnp.array([0, 1, 1, 1, 1])))
        self.assertTrue(
            jnp.all(jnp.argmax(trajectory["next_state"][0], axis=-1) ==
                    jnp.array([1, 1, 1, 1, 1])))
        self.assertTrue(
            jnp.all(trajectory["reward"][0] == jnp.array([0, -1, -1, -1, -1])))
        self.assertTrue(
            jnp.all(trajectory["timeout"][0] == jnp.array([0, 0, 0, 0, 1])))
        self.assertTrue(
            jnp.all(trajectory["terminal"][0] == jnp.array([0, 0, 0, 0, 0])))

        trajectory = agent.async_sample(
            value=jnp.array([
                [-1, -1, -1],
                [1, 1, 1],
            ]),  # Always choose action 1
            length=agent.episode_length,
            n_sample=1,
            key=jrd.PRNGKey(42)
        )

        self.assertTrue(
            jnp.all(jnp.argmax(trajectory["state"][0], axis=-1) == jnp.array([0, 0, 0, 0, 0])))
        self.assertTrue(
            jnp.all(jnp.argmax(trajectory["next_state"][0], axis=-1) == jnp.array([2, 2, 2, 2, 2])))
        self.assertTrue(
            jnp.all(trajectory["reward"][0] == jnp.array([0, 0, 0, 0, 0])))
        self.assertTrue(
            jnp.all(trajectory["timeout"][0] == jnp.array([0, 0, 0, 0, 0])))
        self.assertTrue(
            jnp.all(trajectory["terminal"][0] == jnp.array([1, 1, 1, 1, 1])))


class TestValueIteration(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    def test_sync_learn(self) -> None:
        transition = jnp.array([
            [[0, 0, 0],
             [0.25, 0, 0],
             [0.75, 1, 1]],
            [[0, 0, 0],
             [0.75, 1, 0],
             [0.25, 0, 1]]
        ])
        initial = jnp.array([1, 0, 0])
        terminal = jnp.array([0, 0, 1])
        reward = jnp.array([
            [0, 4, -100],
            [0, 1, 100],
        ])

        agent = ValueIteration(
            transition=transition,
            initial=initial,
            terminal=terminal,
            reward=reward,
            episode_length=5
        )

        value, *_ = agent.sync_learn(
            steps=2,
            gamma=0.5
        )
        self.assertTrue(jnp.allclose(
            value,
            jnp.array([
                [0.5, 4.0, 0.0],
                [1.5, 3.0, 0.0]
            ])
            , atol=1e-4)
        )
