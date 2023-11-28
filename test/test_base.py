import unittest
import jax
import jax.numpy as jnp
import jax.random as jrd
import distrax

import jaxdp.base as DP
from jaxdp.mdp.mdp import MDP


class TestBaseDP(unittest.TestCase):

    def setUp(self) -> None:
        """Load MDPs from mdp data"""
        self.seed = 42
        for mdp_name in ("cyclic", "sequential", "tree", "exploitable", "random_exploitable"):
            setattr(self, f"{mdp_name}_mdp", MDP.load_mdp_from_json(
                f"./mdp_data/{mdp_name}.json"))
        return super().setUp()

    def test_greedy_policy(self):
        self.assertTrue(
            jnp.allclose(
                DP.greedy_policy(value=jnp.array([
                    [2, 0, 2, 0],
                    [0, 2, 0, 2],
                ])),
                jnp.array([
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                ])
            )
        )

    def test_e_greedy_policy(self):
        eps = 0.3
        value = jnp.array([
            [1, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
        ])
        self.assertTrue(
            jnp.allclose(
                DP.e_greedy_policy(value, epsilon=eps),
                jnp.array([
                    [1 - 2 * eps / 3, eps / 3, 1 - 2 * eps / 3, eps / 3],
                    [eps / 3, eps / 3, eps / 3, 1 - 2 * eps / 3],
                    [eps / 3, 1 - 2 * eps / 3, eps / 3, eps / 3],
                ])
            ))

    def test_softmax_policy(self):
        value = jnp.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ])
        self.assertTrue(
            jnp.allclose(
                DP.soft_policy(value, temperature=0),
                jnp.ones_like(value) / value.shape[0]
            ))

        self.assertTrue(
            jnp.allclose(
                DP.soft_policy(value, temperature=1e10),
                value,
                atol=1e-5
            ))

    def test_sample_from(self):
        policy = jnp.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ])

        self.assertTrue(
            jnp.allclose(
                DP.sample_from(policy, key=jrd.PRNGKey(self.seed)),
                policy,
                atol=1e-5
            ))

        n_sample = 250000
        policy = jnp.array([
            [0.1, 0.5, 0.25, 0],
            [0.9, 0.5, 0.75, 1],
        ])
        batch_sample_from = jax.vmap(
            DP.sample_from,
            in_axes=(None, 0),
            out_axes=0)
        self.assertTrue(
            jnp.allclose(
                jnp.mean(
                    batch_sample_from(
                        policy,
                        jrd.split(jrd.PRNGKey(self.seed), n_sample)
                    ), axis=0),
                policy,
                atol=2e-3)
        )

    def test_policy_evaluations(self):
        gamma = 0.5
        policy = jnp.array([
            [0.25, 0.75, 0.5, 1],
            [0.75, 0.25, 0.5, 0],
        ])
        true_values = jnp.array([
            [3 * gamma / 4 + gamma ** 2 / 2, 1 + gamma / 2, 0.0, 0],
            [1 + 3 * gamma / 4 + gamma ** 2 / 2, gamma / 2, 1.0, 0]
        ])
        self.assertTrue(
            jnp.allclose(
                DP.q_policy_evaluation(
                    self.sequential_mdp,
                    policy,
                    gamma=gamma),
                true_values
            )
        )

        self.assertTrue(
            jnp.allclose(
                DP.policy_evaluation(
                    self.sequential_mdp,
                    policy,
                    gamma=gamma),
                jnp.einsum("as,as->s", policy, true_values)
            )
        )

    def test_bellman_operator(self):
        gamma = 0.5
        policy = jnp.array([
            [0.25, 0.75, 0.5, 1],
            [0.75, 0.25, 0.5, 0],
        ])
        value = jnp.array([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
        ])
        true_diff = jnp.array([
            [5/4 * gamma - 1, 3/2 * gamma, -1, 0],
            [5/4 * gamma - 1, 3/2 * gamma - 2, -1, 0]
        ])
        self.assertTrue(
            jnp.allclose(
                DP.bellman_operator(
                    self.sequential_mdp,
                    policy,
                    value,
                    gamma=gamma) - value,
                true_diff
            )
        )

        self.assertTrue(
            jnp.allclose(
                DP.bellman_operator(
                    self.sequential_mdp,
                    policy,
                    DP.q_policy_evaluation(
                        self.sequential_mdp,
                        policy,
                        gamma=gamma
                    ) - value,
                    gamma=gamma),
                jnp.zeros((2, 4))
            )
        )

    def test_sync_sample(self) -> None:
        mdp = MDP.load_mdp_from_json("./mdp_data/tree.json")

        policy = jnp.array([
            [0.75, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.5, 0.5, 0.5, 0.5],
            [0.25, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5, 0.5, 0.5],
        ])
        batch_size = 100000
        key = jrd.PRNGKey(self.seed)
        keys = jrd.split(key, batch_size)

        sync_sample_step = jax.jit(
            jax.vmap(
                DP.sync_sample,
                in_axes=(None, None, 0),
                out_axes=0)
        )

        state, action, reward, next_state, terminal = sync_sample_step(
            mdp, policy, keys)

        target_next_state = jnp.array([
            [0, 0.375, 0.375, 0, 0, 0, 0, 0.125, 0.125, 0, 0, 0, 0],
            [0, 0, 0, 0.125, 0.125, 0.375, 0.375, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.375, 0.375, 0.125, 0.125],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        target_terminal = jnp.array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        self.assertTrue(jnp.allclose(
            target_next_state, next_state.mean(axis=0), atol=5e-2
        ))

        self.assertTrue(jnp.allclose(
            policy, action.mean(axis=0).T, atol=1e-2
        ))
        self.assertTrue(jnp.allclose(
            target_terminal, terminal.mean(axis=0), atol=1e-2
        ))
        self.assertTrue(jnp.allclose(
            (policy[0] - policy[1]) * (1 - mdp.terminal), reward.mean(axis=0), atol=1e-2
        ))

    def test_async_sample_step_pi(self) -> None:
        mdp = MDP.load_mdp_from_json("./mdp_data/tree.json")

        policies = jnp.array([[
            [0.75, 0.25, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.75, 0.5, 0.5, 0.5, 0.5],
            [0.25, 0.75, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.25, 0.5, 0.5, 0.5, 0.5]],
        ])
        batch_size = 100000
        policy_size = policies.shape[0]

        key = jrd.PRNGKey(self.seed)
        key, initial_state_key = jrd.split(key, 2)
        batch_key_split = jax.vmap(jrd.split,
                                   in_axes=(0, None),
                                   out_axes=1)

        async_step = jax.jit(
            jax.vmap(
                jax.vmap(
                    DP.async_sample_step_pi,
                    in_axes=(None, None, 0, 0, None, 0),
                    out_axes=0),
                in_axes=(None, 0, 0, 0, None, 0),
                out_axes=0)
        )

        episode_length = 2
        states = distrax.OneHotCategorical(probs=mdp.initial.T).sample(
            seed=initial_state_key, sample_shape=(policy_size, batch_size))
        episode_step = jnp.zeros((policy_size, batch_size))

        data = {
            "states": states,
            "action": None,
            "reward": None,
            "next_state": None,
            "terminal": None,
            "timeout": None,
            "episode_step": episode_step,
            "key": key,
        }

        def _step(data):

            data["key"], step_key = jrd.split(data["key"], 2)
            keys = batch_key_split(
                jrd.split(step_key, batch_size),
                policy_size
            )
            (
             data["action"],
             data["next_state"],
             data["reward"],
             data["terminal"],
             data["timeout"],
             data["states"],
             data["episode_step"]
             ) = async_step(
                mdp,
                policies,
                data["states"],
                data["episode_step"],
                episode_length,
                keys
            )
            return data

        self.assertTrue(
            jnp.allclose(
                jnp.argmax(states, axis=-1),
                0
            )
        )
        _step(data)

        self.assertTrue(
            jnp.allclose(
                data["next_state"][0].mean(axis=0),
                jnp.array([0, 0.375, 0.375, 0, 0, 0, 0,
                          0.125, 0.125, 0, 0, 0, 0]),
                atol=1e-2,
            )
        )
        self.assertTrue(
            jnp.allclose(
                data["states"][0].mean(axis=0),
                jnp.array([0.5, 0.375, 0, 0, 0, 0, 0, 0, 0.125, 0, 0, 0, 0]),
                atol=1e-2,
            )
        )
        self.assertTrue(jnp.allclose(
            data["terminal"][0].mean(axis=0), 0.5, atol=1e-2))
        self.assertTrue(jnp.allclose(
            data["reward"][0].mean(axis=0), 0.5, atol=1e-2))
        _step(data)

        self.assertTrue(
            jnp.allclose(
                data["next_state"][0].mean(axis=0),
                jnp.array([0, 0.375/2, 0.375/2, 0.375/8,
                           0.375/8, 0.375/8*3, 0.375/8*3,
                           0.125/2, 0.125/2, 0.125/8*3,
                           0.125/8*3, 0.125/8, 0.125/8]),
                atol=5e-2,
            )
        )
        self.assertTrue(jnp.allclose(
            data["timeout"][0].mean(axis=0), 0.5, atol=1e-2))
