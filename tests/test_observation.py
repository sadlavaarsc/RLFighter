import numpy as np

from rlfighter.core.action import ActionType, Phase
from rlfighter.core.observation import build_observation
from rlfighter.core.world import World


def test_observation_shape():
    world = World([1, 1], seed=0)
    obs = build_observation(world.agents[0], world)
    # self: 20, others: 4 * 19 = 76, total = 96
    assert obs.shape == (96,)
    assert obs.dtype == np.float32


def test_self_hp_normalized():
    world = World([1, 1], seed=0)
    agent = world.agents[0]
    agent.hp = 50.0
    obs = build_observation(agent, world)
    assert obs[0] == 0.5


def test_mask_for_missing_others():
    world = World([1, 1], seed=0)
    obs = build_observation(world.agents[0], world)
    # First other starts at index 20, mask is at index 20
    assert obs[20] == 1.0
    # Second other (missing) mask at index 20 + 19 = 39
    assert obs[39] == 0.0
    assert obs[58] == 0.0
    assert obs[77] == 0.0


def test_action_type_onehot():
    world = World([1, 1], seed=0)
    agent = world.agents[0]
    world.step({agent.agent_id: (ActionType.THRUST, 2)})
    obs = build_observation(agent, world)
    # action_type one-hot starts at index 3, size 6
    # THRUST = 3, so index 3 + 3 = 6 should be 1
    assert obs[6] == 1.0
