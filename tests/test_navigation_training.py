"""Navigation through the framework's training tools: the curriculum over its domain
randomization and the robustness sweep over sensor noise (``rl_core.training.curriculum``,
``rl_core.training.robustness``, ``rl_navigation.environments``).

Characterization tests: they pin behaviour that already worked before navigation PR 5, so the
TESTING rows ``TR-36`` and ``ROB-2`` rest on it."""

import contextlib

import pytest

from rl_core.training.curriculum import curriculum_from_config
from rl_core.training.robustness import physics_defaults, training_range
from rl_navigation.benchmark import NavigationChallenge
from rl_navigation.environments import make_env_from_config

SECTION = {
    "name": "Navigation-v0",
    "seed": 4,
    "domain_randomization": {"sensor_noise_std": [0.0, 0.2], "num_obstacles": [2.0, 8.0]},
    "curriculum": {
        "mode": "linear",
        "episodes": 2,
        "start": {
            "domain_randomization": {"sensor_noise_std": [0.01, 0.01], "num_obstacles": [2, 2]}
        },
    },
}


def test_a_curriculum_widens_the_sensor_noise_and_obstacle_draws_step_by_step():
    with contextlib.closing(make_env_from_config(dict(SECTION))) as env:
        curriculum = curriculum_from_config(SECTION, env)
        curriculum.record(0.0)  # one of two episodes played: difficulty 0.5
        settings = curriculum.settings()["domain_randomization"]
        assert settings["sensor_noise_std"] == pytest.approx((0.005, 0.105))
        assert settings["num_obstacles"] == pytest.approx((2.0, 5.0))
        env.apply_settings(curriculum.settings())
        noise, counts = [], []
        for _ in range(40):
            env.reset()
            raw = env.env.unwrapped
            noise.append(raw.sensor_noise_std)
            counts.append(raw.num_obstacles)
    assert all(0.005 <= n <= 0.105 for n in noise), "each draw lies in the halfway range"
    assert set(counts) <= {2, 3, 4, 5} and len(set(counts)) > 1, (
        "obstacle counts are drawn from 2 to 5, not pinned at the start's 2"
    )


def test_the_robustness_sweep_reads_the_sonar_defaults_as_its_trained_range():
    section = {"name": "Navigation-v0", "seed": 1}
    defaults = physics_defaults(NavigationChallenge(), section)
    assert defaults["sensor_noise_std"] == pytest.approx(0.01), "the HC-SR04's own noise"
    assert defaults["dropout_prob"] == pytest.approx(0.02), "the HC-SR04's own dropout"
    with contextlib.closing(make_env_from_config(dict(section))) as env:
        target = env.scheduled_settings()
    trained = training_range("sensor_noise_std", target, None, 1.0, defaults)
    assert trained == pytest.approx((0.01, 0.01)), (
        "without randomization the agent trained at the sensor's own noise only, so a sweep "
        "value of 0.01 is in range"
    )
