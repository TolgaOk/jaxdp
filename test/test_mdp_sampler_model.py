import jax
import jax.numpy as jnp

from jaxdp.mdp import Mdp
from jaxdp.mdp.sampler.mdp import State, _queue_push, rollout, step


def _terminal_mdp() -> Mdp:
    transition = jnp.array([[[0.0, 0.0], [1.0, 1.0]]])
    reward = jnp.zeros((1, 2, 2)).at[0, 0, 1].set(3.0)
    initial = jnp.array([1.0, 0.0])
    terminal = jnp.array([0.0, 1.0])
    return Mdp(transition, reward, initial, terminal)


def _sampler_state(mdp: Mdp) -> State:
    return State(
        last_state=mdp.initial,
        episode_step=jnp.array(0),
        rewards=jnp.array(0.0),
        lengths=jnp.array(0),
        episode_reward_queue=jnp.full(2, jnp.nan),
        episode_length_queue=jnp.full(2, jnp.nan),
    )


def test_step_keeps_transition_successor_separate_from_reset_state() -> None:
    mdp = _terminal_mdp()
    data, state = jax.jit(step)(
        jax.random.key(0),
        _sampler_state(mdp),
        jnp.ones((1, 2)),
        mdp,
        10,
    )

    assert jnp.array_equal(data.state, jnp.array([1.0, 0.0]))
    assert jnp.array_equal(data.next_state, jnp.array([0.0, 1.0]))
    assert data.reward == 3.0
    assert data.terminal
    assert jnp.array_equal(state.last_state, mdp.initial)
    assert state.episode_step == 0
    assert state.episode_reward_queue[0] == 3.0
    assert state.episode_length_queue[0] == 1


def test_queue_push_preserves_nan_slots_when_idle() -> None:
    queue = jnp.array([2.0, jnp.nan])

    assert jnp.allclose(_queue_push(queue, jnp.array(3.0), jnp.array(False)), queue, equal_nan=True)


def test_rollout_records_successors_across_resets() -> None:
    mdp = _terminal_mdp()
    data, state = rollout(
        jax.random.key(0),
        _sampler_state(mdp),
        jnp.ones((1, 2)),
        mdp,
        rollout_len=2,
        max_episode_len=10,
    )

    assert jnp.array_equal(data.state, jnp.array([[1.0, 0.0], [1.0, 0.0]]))
    assert jnp.array_equal(data.next_state, jnp.array([[0.0, 1.0], [0.0, 1.0]]))
    assert jnp.array_equal(state.last_state, mdp.initial)
    assert jnp.array_equal(state.episode_reward_queue, jnp.array([3.0, 3.0]))
