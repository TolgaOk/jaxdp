from itertools import product

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdp import MDP
from jaxdp.mdp.sysadmin import sysadmin_mdp


@pytest.mark.parametrize(("machine_size", "topology"), [(3, "circular"), (4, "three-leg")])
def test_sysadmin_matches_independent_factored_transition_tables(
    machine_size: int,
    topology: str,
) -> None:
    mdp = sysadmin_mdp(machine_size, topology)
    configurations = list(product((0, 1), repeat=machine_size))
    state_size = len(configurations)
    expected = np.zeros((machine_size + 1, state_size, state_size))
    circular = ((0.0238, 0.475), (0.0475, 0.95))

    for action in range(machine_size + 1):
        for state, current in enumerate(configurations):
            probabilities = []
            for machine in range(machine_size):
                if topology == "circular":
                    parent = (machine - 1) % machine_size
                    probability = (
                        1.0 if action == machine else circular[current[parent]][current[machine]]
                    )
                elif action == machine:
                    probability = 0.95
                elif current[machine] == 0:
                    probability = 0.01
                elif current[max(0, machine - 3)] == 1:
                    probability = 0.90
                else:
                    probability = 0.67
                probabilities.append(probability)
            for successor, following in enumerate(configurations):
                expected[action, successor, state] = np.prod(
                    [
                        probability if working else 1 - probability
                        for probability, working in zip(probabilities, following, strict=True)
                    ]
                )

    assert isinstance(mdp, MDP)
    np.testing.assert_allclose(mdp.transition, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(mdp.transition.sum(axis=-2), 1, rtol=1e-6, atol=1e-7)
    expected_reward = np.asarray([sum(state) / machine_size for state in configurations])
    np.testing.assert_allclose(
        mdp.reward,
        np.broadcast_to(expected_reward[None, :, None], mdp.reward.shape),
        rtol=1e-7,
        atol=1e-8,
    )
    np.testing.assert_array_equal(mdp.initial, np.eye(1, state_size, 0)[0])
    assert not jnp.any(mdp.terminal)
    assert jnp.all(jnp.mean(mdp.transition, axis=0) > 0)
    mdp.validate()


def test_circular_reboot_selects_only_its_machine_and_idle_is_the_last_action() -> None:
    mdp = sysadmin_mdp(3)
    configurations = np.asarray(list(product((0, 1), repeat=3)))
    marginal = np.einsum("xi,axs->ais", configurations, np.asarray(mdp.transition))

    np.testing.assert_allclose(marginal[0, :, 0], [1.0, 0.0238, 0.0238], atol=1e-7)
    np.testing.assert_allclose(marginal[1, :, 0], [0.0238, 1.0, 0.0238], atol=1e-7)
    np.testing.assert_allclose(marginal[3, :, 0], [0.0238, 0.0238, 0.0238], atol=1e-7)
    # In 001, the last machine is machine zero's predecessor, not its own status.
    np.testing.assert_allclose(marginal[3, 0, 1], 0.0475, atol=1e-7)
    np.testing.assert_allclose(marginal[3, 0, 4], 0.475, atol=1e-7)


def test_three_leg_dependencies_use_current_states_and_preserve_the_published_reboot_rate() -> None:
    mdp = sysadmin_mdp(7, "three-leg")
    configurations = np.asarray(list(product((0, 1), repeat=7)))
    marginal = np.einsum("xi,axs->ais", configurations, np.asarray(mdp.transition))
    # State 0100000 has machine one working and the root failed.
    np.testing.assert_allclose(marginal[0, [0, 1], 32], [0.95, 0.67], atol=1e-7)
    # On the next level, machine four depends on machine one rather than the root.
    np.testing.assert_allclose(marginal[7, 4, 36], 0.90, atol=1e-7)
    np.testing.assert_allclose(marginal[7, 4, 4], 0.67, atol=1e-7)
    np.testing.assert_allclose(marginal[7, 0, 0], 0.01, atol=1e-7)
    np.testing.assert_allclose(marginal[7, 0, 64], 0.90, atol=1e-7)


def test_circular_sysadmin_supports_one_machine() -> None:
    mdp = sysadmin_mdp(1)

    np.testing.assert_array_equal(mdp.transition[0], [[0, 0], [1, 1]])
    np.testing.assert_allclose(mdp.transition[1], [[0.9762, 0.05], [0.0238, 0.95]], atol=1e-7)


@pytest.mark.parametrize(("machine_size", "topology"), [(3, "circular"), (4, "three-leg")])
def test_sysadmin_composes_with_jit_and_vmap(machine_size: int, topology: str) -> None:
    def create_one(key: jax.Array) -> MDP:
        return sysadmin_mdp(machine_size, topology)

    create = chex.chexify(jax.jit(create_one), async_check=False)
    create_batch = chex.chexify(jax.jit(jax.vmap(create_one)), async_check=False)
    keys = jax.random.split(jax.random.key(7), 2)
    expected = sysadmin_mdp(machine_size, topology)
    expected_batch = jax.tree.map(lambda array: jnp.stack((array, array)), expected)

    chex.assert_trees_all_close(create(keys[0]), expected, rtol=1e-6, atol=1e-7)
    batch = create_batch(keys)
    chex.assert_trees_all_close(batch, expected_batch, rtol=1e-6, atol=1e-7)
    batch.validate()


@pytest.mark.parametrize("machine_size", [0, -1, 1.5])
def test_sysadmin_rejects_invalid_machine_counts(machine_size: int) -> None:
    with pytest.raises(AssertionError, match="machine_size"):
        sysadmin_mdp(machine_size)


@pytest.mark.parametrize("machine_size", [1, 2, 3, 5, 6, 8])
def test_three_leg_sysadmin_requires_three_equal_nonempty_legs(machine_size: int) -> None:
    with pytest.raises(AssertionError, match="three-leg"):
        sysadmin_mdp(machine_size, "three-leg")


def test_sysadmin_rejects_unknown_topology() -> None:
    with pytest.raises(ValueError, match="topology"):
        sysadmin_mdp(4, "unknown")
