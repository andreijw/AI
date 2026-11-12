# Training an Agent on Another Project's Environment

Another project, such as a video game, can train this framework's agents on its own environment without any edit
here. The project depends on `rl-core`, wraps its game as a Gymnasium environment, and registers a challenge through
the `rl_core.challenges` entry-point group. `train.py` and `benchmark.py` then find it by the config's environment id,
as they find CartPole. This page builds a toy game, `my-game`, a corridor to walk to its exit. The commands and the
code below were run as written on 2026-11-11: 20 PPO episodes trained, `benchmark.py --help` offered `--challenge
{corridor}`, and the benchmark command wrote its report.

## 1. The project

```text
my-game/
  pyproject.toml
  configs/corridor_ppo.yaml
  src/my_game/__init__.py
  src/my_game/env.py          # the game as a Gymnasium environment
  src/my_game/benchmark.py    # the challenge: wrapper, agents, configs
```

```toml
[build-system]
requires = ["setuptools>=82.0.1"]
build-backend = "setuptools.build_meta"

[project]
name = "my-game"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["rl-core~=0.1.0", "gymnasium>=1.0.0", "numpy>=1.26"]

# The one line that makes the challenge visible to rl-core: the entry's name is the import package,
# its value the module that registers the challenge.
[project.entry-points."rl_core.challenges"]
my_game = "my_game.benchmark"

[tool.setuptools.packages.find]
where = ["src"]
```

`rl-core~=0.1.0` accepts 0.1.x and stops before 0.2, which may change the API ([releasing.md](releasing.md)).

## 2. The game as a Gymnasium environment

Use the Gymnasium 1.x API: `reset(seed=...)` returns `(obs, info)`, and `step` returns `(obs, reward, terminated,
truncated, info)`, where `terminated` means the game ended (won or lost) and `truncated` that time ran out. Register
an id so that `gym.make` can build it.

```python
"""src/my_game/env.py: the game as a Gymnasium environment."""

import gymnasium as gym
import numpy as np


class CorridorEnv(gym.Env):
    """Start at cell 0 of a 10-cell corridor; reach the last cell. Actions: 0 left, 1 right."""

    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, length=10):
        self.length = length
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.position = 0

    def _obs(self):
        return np.array([self.position / (self.length - 1)], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        return self._obs(), {}

    def step(self, action):
        step = 1 if action == 1 else -1
        self.position = int(np.clip(self.position + step, 0, self.length - 1))
        terminated = self.position == self.length - 1
        return self._obs(), (1.0 if terminated else -0.01), terminated, False, {}


gym.register(id="Corridor-v0", entry_point="my_game.env:CorridorEnv")
```

A real game exposes its state as the observation (positions, health, a downscaled frame) and its controls as the
action space; it must be able to run without a window, faster than real time, and from a seed.

## 3. The challenge

The challenge tells the framework how to build the environment and which agents may run on it. `BaseEnvWrapper` adds
the framework's seeding, observation noise and episode statistics around the Gymnasium environment; `env_from_config`
reads a config's `environment` section against the settings the challenge lists.

```python
"""src/my_game/benchmark.py: the game as a challenge, found by its entry point."""

from collections.abc import Mapping
from typing import Any, ClassVar

import my_game.env  # noqa: F401  (registers Corridor-v0)
from rl_core.agents import BaseAgent, DQNAgent, PPOAgent, RandomAgent
from rl_core.challenges import BenchmarkChallenge, register_challenge
from rl_core.environments import OBSERVATION_DEFAULTS, BaseEnvWrapper, env_from_config


class CorridorWrapper(BaseEnvWrapper):
    """The framework's wrapper: seeding, observation noise and episode statistics."""

    def __init__(
        self,
        env_name="Corridor-v0",
        render_mode=None,
        max_episode_steps=50,
        seed=None,
        obs_noise_std=0.0,
        **observation,
    ):
        super().__init__(
            env_name, render_mode, max_episode_steps, seed, obs_noise_std, **observation
        )
        self._make_env()


# Every environment setting a config may set, with its default.
ENV_DEFAULTS = {
    "name": "Corridor-v0",
    "max_episode_steps": 50,
    "obs_noise_std": 0.0,
    **OBSERVATION_DEFAULTS,
}


def make_env(env_name="Corridor-v0", **settings):
    return CorridorWrapper(env_name=env_name, **settings)


@register_challenge
class CorridorChallenge(BenchmarkChallenge):
    AGENT_CLASSES: ClassVar[dict[str, type[BaseAgent]]] = {
        "random": RandomAgent,
        "ppo": PPOAgent,
        "dqn": DQNAgent,
    }
    CONFIG_MAP: ClassVar[dict[str, str]] = {"ppo": "corridor_ppo.yaml"}

    @property
    def name(self) -> str:
        return "Corridor-v0"  # the Gymnasium id; a config's environment.name selects the challenge

    @property
    def key(self) -> str:
        return "corridor"  # benchmark.py --challenge corridor

    @property
    def solve_threshold(self) -> float | None:
        return 0.9

    @property
    def default_episodes(self) -> int:
        return 200

    def make_env(self, env_config: Mapping[str, Any], render_mode: str | None):
        config = {**env_config, "render_mode": render_mode}
        return env_from_config(make_env, config, ENV_DEFAULTS)
```

Where `rl-challenges` is installed too, a key that one of its challenges already has (`cartpole`) is refused: the
repository's distributions load first, and the game's challenge is then reported as failing to import, with the
reason, rather than replacing CartPole.

Continuous actions (`gym.spaces.Box`) take the continuous agents (`ContinuousPPOAgent`, `SACAgent`, `TD3Agent`,
...). They read the action bounds from their config (`action_low`, `action_high`) and default to [-1, 1], so a
challenge with other bounds passes the environment's on, as `rl_pendulum/benchmark.py` in this repository does:

```python
from rl_core.challenges import with_box_action_bounds

    # inside the challenge class
    def agent_config(self, agent_config, env):
        return with_box_action_bounds(agent_config, env, self.name)
```

Without it, a game whose actions span `Box(-5, 5)` trains silently in [-1, 1].

`CorridorEnv` has no rendering, so its configs keep `render_mode: null`. For `train.py --render`, `record.py` or
training videos, list `"rgb_array"` in `metadata["render_modes"]`, keep the `render_mode` argument and implement
`render()` to return the frame.

## 4. A config

The same format as `configs/*.yaml` in this repository. Point `checkpoint_dir` and `log_dir` into the game's own tree.

```yaml
environment:
  name: "Corridor-v0"
  render_mode: null
  max_episode_steps: 50
  seed: 1

agent:
  type: "ppo"
  config:
    learning_rate: 0.0003
    rollout_steps: 256
    minibatch_size: 64
    ppo_epochs: 4
    gamma: 0.99
    gae_lambda: 0.95
    hidden_dim: 32
    seed: 1

training:
  num_episodes: 20
  max_steps_per_episode: 50
  eval_frequency: 10
  save_frequency: 10
  checkpoint_dir: "./out/checkpoints"
  log_dir: "./out/logs"

logging:
  level: "INFO"
```

## 5. Training

The game trains in **its own virtual environment**, never in this repository's `.venv`: the tests here see only this
repository's challenges, and a game installed beside them would be imported by every run. Install `rl-core` (from
this checkout while developing, or from its release URL) and the game, then run this checkout's scripts with that
environment's interpreter:

```bash
cd my-game
python3.12 -m venv .venv
.venv/bin/pip install -e /path/to/AI/packages/rl-core -e .
# or, from a release: .venv/bin/pip install \
#   https://github.com/andreijw/AI/releases/download/rl-core-v0.1.0/rl_core-0.1.0-py3-none-any.whl -e .

.venv/bin/python /path/to/AI/train.py --config configs/corridor_ppo.yaml --num-episodes 20
.venv/bin/python /path/to/AI/benchmark.py --challenge corridor --agents ppo --episodes 50 --seeds 1 \
    --output-dir out/benchmark --report out/benchmark.md
```

`train.py` picks the challenge from the config's `environment.name`. The benchmark reads each agent's config, named in
`CONFIG_MAP`, from its `--config-dir` (default `configs` under the current directory). Without `--agents` it runs
every agent in `AGENT_CLASSES`, and one with no `CONFIG_MAP` entry stops it with a `KeyError`: here, pass
`--agents ppo`, or add configs for `random` and `dqn`. The entry point is installed metadata: after changing
`[project.entry-points]`, re-run the game's `pip install -e .`. On Windows the interpreter is `.venv\Scripts\python`.
