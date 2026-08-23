import jax
import jax.numpy as jnp

from jaxdp.mdp import Mdp
from jaxdp.mdp.sampler.mdp import State, _queue_push, rollout, sample_step, step


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


def test_sample_step_returns_the_successor_and_continuation_separately() -> None:
    mdp = _terminal_mdp()
    data, continuation, episode_step = jax.jit(sample_step)(
        jax.random.key(0),
        mdp.initial,
        jnp.array(0),
        jnp.ones((1, 2)),
        mdp,
        10,
    )

    assert jnp.array_equal(data.next_state, jnp.array([0.0, 1.0]))
    assert jnp.array_equal(continuation, mdp.initial)
    assert episode_step == 0


def test_sample_step_matches_action_and_transition_probabilities() -> None:
    mdp = Mdp(
        transition=jnp.array(
            [
                [[0.25, 0.0], [0.75, 1.0]],
                [[0.80, 0.0], [0.20, 1.0]],
            ]
        ),
        reward=jnp.zeros((2, 2, 2)),
        initial=jnp.array([1.0, 0.0]),
        terminal=jnp.zeros(2),
    )
    policy = jnp.array([[0.25, 0.5], [0.75, 0.5]])
    keys = jax.random.split(jax.random.key(0), 20_000)

    data = jax.vmap(
        lambda key: sample_step(key, mdp.initial, jnp.array(0), policy, mdp, 10)[0]
    )(keys)

    assert jnp.allclose(jnp.mean(data.action, axis=0), jnp.array([0.25, 0.75]), atol=0.02)
    assert jnp.allclose(
        jnp.mean(data.next_state, axis=0),
        jnp.array([0.6625, 0.3375]),
        atol=0.02,
    )


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
