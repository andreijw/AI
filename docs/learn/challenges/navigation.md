# Navigation-v0 and NavigationContinuous-v0

Drive a disc robot to a goal past obstacles, using its range sensors and the goal's distance and bearing. These are
the only challenges here built on this repository's own simulator rather than Gymnasium's, and they are the bridge to
robotics: after this page, the [navigation track](../navigation/sim-and-sensors.md) builds the classical alternative
(estimation, mapping, SLAM, planning) and ends with [how a policy reaches a robot](../navigation/export-to-robot.md).

## The task

`Navigation-v0` and `NavigationContinuous-v0` are this project's own Gymnasium environments (Phase 8 of
[docs/roadmap.md](../../roadmap.md)): a disc robot drives to a goal through randomly placed obstacles, seeing its range
sensors and told the goal's distance and bearing, as an oracle: a real robot would need localisation to supply them
([estimation](../navigation/estimation.md)). Each reset draws a new layout; `reset(options={"layout": layout})` plays a
given `rl_navigation.sim.Layout` instead (CHANGELOG.md, 2026-11-05). The robot, its sensors and the layout generator are
described in [the simulator and its sensors](../navigation/sim-and-sensors.md).

The environment does not need the training framework. `rl_navigation.gym_env` imports only Gymnasium and the NumPy
simulator `rl_navigation.sim`, never `rl_core` (a subprocess test holds it), so another project can use it directly
(CHANGELOG.md, 2026-11-05); the snippet is in
[docs/navigation.md](../../navigation.md#using-the-environment-on-its-own). The framework's noise and domain
randomization live in `rl_navigation.environments`, which does import `rl_core`.

## Observation, action and reward

- **Actions**: `Navigation-v0` has five discrete actions (straight ahead, forward arcs left and right at half speed,
  turns on the spot left and right). `NavigationContinuous-v0` takes two actions in [-1, 1]: forward speed from 0 to 0.5
  m/s (no reverse, as the sonars face forward) and turn rate within ±2 rad/s (CHANGELOG.md, 2026-11-05).
- **Observation**: goal distance, cos and sin of the goal's bearing, the last executed forward speed and turn rate, then
  the range readings: five HC-SR04 sonars by default (`sensor: "hc_sr04"`, 10 values in all), or a full-circle lidar
  (`sensor: "lidar"`, `lidar_beams`, default 64) (CHANGELOG.md, 2026-11-05).
- **Reward**: 10 × the progress toward the goal over the starting distance, -0.01 per step and -10 on a collision, so a
  crash never pays (the maintainer's decision) (CHANGELOG.md, 2026-11-05).
- **Framework settings** (`rl_navigation.environments.NavigationEnv`): `action_noise_prob` (discrete) or
  `action_noise_std` (continuous), `obs_noise_std`, `frame_stack`, `frame_deltas`, `num_obstacles`, and
  `domain_randomization` ranges of `wheel_slip`, `sensor_noise_std`, `dropout_prob` and `num_obstacles`. The ranges are
  checked against what the arena holds before any episode, also in a curriculum or robustness sweep.
  `mask_velocity: true` is refused (CHANGELOG.md, 2026-11-05).
- **Rendering**: `render_mode: "rgb_array"` draws the arena in NumPy (`rl_navigation/sim/render.py`), so
  `--record-video` and `record.py` work as for any challenge; `"human"` opens a window through Gymnasium's
  `HumanRendering`, which needs pygame-ce (installed by the `box2d` extra) (CHANGELOG.md, 2026-11-05; the
`rl_navigation.environments.NavigationEnv` docstring).

## How an episode ends and the solve threshold

Reaching the goal or colliding ends the episode; otherwise it is **truncated** after 400 steps ([terminated and
truncated](../foundations.md#terminated-and-truncated); CHANGELOG.md, 2026-11-05). Both ids are solved when the training
average over 100 episodes reaches 7.0 ([solved and held](../foundations.md#solved-and-held)), which is about 90 % of
training episodes at the goal (the maintainer's choice, 2026-11-07); the reward has no goal bonus, so the threshold also
asks for short trips (CHANGELOG.md, 2026-11-07).

Success is counted too: an episode's success is `info["is_success"]` at its last step (the Gymnasium and SB3 Monitor
convention), and the training log, the evaluation, the Markdown report's "Eval Success" column and the JSON carry the
rate. It is reported beside the reward and does not gate the solve (CHANGELOG.md, 2026-11-07). `info` holds
`is_success`, `collision` and `goal_distance`.

## Who solves it

The benchmark runs `random`, `ppo` and `dqn` on `Navigation-v0` (`--challenge navigation`) and `random`, `ppo` and `td3`
on `NavigationContinuous-v0` (`--challenge navigation_continuous`), each from its committed config
(`configs/navigation_*.yaml`). The table is in [results.md](../../results.md#which-agents-solve-which-challenge), and
the published runs, on seeds 1 to 5 with greedy evaluation over 100 held-out layouts, in [the Navigation
report](../../benchmarks/navigation_results.md) and [the NavigationContinuous
report](../../benchmarks/navigation_continuous_results.md). Read the "Eval Success" column there: it is the share of
evaluation episodes that end at the goal. In short, TD3 and DQN solve and hold; PPO reaches 7.0 but does not hold
it, with either action space; the random agent solves none (CHANGELOG.md, 2026-11-07).

**Robustness.** A sweep over sensor noise and dropout (TESTING.md `ROB-2`, three TD3 seeds, 200 episodes a row) kept
success between 87 and 98 % on every row, from no noise to 0.4 m and from no dropout to 40 % (one DQN seed fell to 82 %
at dropout 0.4). A domain-randomization curriculum (noise to [0, 0.2], dropout to [0, 0.3]) did not make TD3 more
robust: 85.3 % against 91.7 % at dropout 0.4 over 3 seeds, so no curriculum config is committed. Two predictions written
before the runs failed: dropout 0.2 did not cost TD3 10 points (it gained 2.3) (CHANGELOG.md, 2026-11-07).

## What was hard

**DQN learned nothing at first.** With the squared-error loss and a fixed learning rate, DQN met its pre-registered
prediction on none of seeds 1 to 5: it reached the goal in 11 % of evaluation episodes (0 to 50 % by seed), mostly
running out the 400-step limit, while 53 to 66 % of its last 100 training episodes ended in a collision (CHANGELOG.md,
2026-11-06). The training-doctor found two stacked causes. Rare collisions (-10, about 200 times the step reward's
standard deviation) let the squared error dominate the gradient and overestimate Q (12 to 18 where no return exceeds
about 10), so the robot parked and dithered; and a fixed learning rate on action gaps of about 0.07 kept flipping the
greedy policy. The fix is two changes, `td_loss: huber` and a learning rate decaying to 0 over the budget, and each
alone falls short: Huber with a fixed rate reached the goal in 0 to 18 % of evaluation episodes (mean 4 %), squared
error with the decaying rate in 14 to 86 % (mean 64 %). With both, seeds 1 to 5 reach the goal in 89 to 96 % of
episodes, with mean Q about 3.3 (CHANGELOG.md, 2026-11-07). Double DQN, dueling and an epsilon floor of 0.02 did not
help (seed 1) (CHANGELOG.md, 2026-11-07). The loss is described with the other DQN fixes in [DQN and its
variants](../algorithms/dqn.md).

**PPO reaches the threshold and cannot keep it.** See the held column of the reports above. TD3, with half the step
budget, holds it, at a higher cost per run in wall time (both in [the continuous
report](../../benchmarks/navigation_continuous_results.md)).

**Sonars write false walls.** The agents see five sonar readings, a cone each. The classical
planner of [Planning and control](../navigation/planning.md#planner-against-trained-policies), with a map built from the
same readings, is stalled by false walls under range noise where TD3 is not.

## See it

![Navigation-v0, random agent: robot (blue), goal (green), sonar rays](../../images/navigation_random_still.png)

![The same moment with sensor: lidar, the robot's full circle of beams](../../images/navigation_lidar_still.png)

![The random agent's whole episode, sped up](../../images/navigation_random.gif)

The random agent of `configs/navigation_default.yaml`, and the same moment with the config's `sensor` set to `lidar`.
The clip is sped up.

![Navigation-v0, trained DQN: the robot steers around the obstacles to the goal](../../images/navigation_dqn.gif)

![NavigationContinuous-v0, trained TD3: a smooth path to the goal](../../images/navigation_continuous_td3.gif)

The trained agents, each seed 1 of its committed config, retrained on 2026-11-19 with the same training result as the
reports' seed 1: DQN of `configs/navigation_dqn.yaml` ([Navigation report](../../benchmarks/navigation_results.md))
and TD3 of `configs/navigation_continuous_td3.yaml` ([NavigationContinuous
report](../../benchmarks/navigation_continuous_results.md)). Each final checkpoint plays one greedy episode from the
reset seed in the middle of 20 such episodes (reset seeds 0 to 19): DQN from seed 1 reaches the goal in 85 steps
(return 8.59), TD3 from seed 8 in 83 (return 8.65). DQN reaches the goal in all 20, TD3 in 19, failing from seed 1.
Both clips run at twice the environment's speed (CHANGELOG.md, 2026-11-19).

The classical pipeline's pictures are on its own pages: a map built from true poses against one built from dead
reckoning is on [Mapping](../navigation/mapping.md), and a filter's estimate after a kidnapping on
[Estimation](../navigation/estimation.md#watching-one-trial).

## Try it

```bash
python train.py --config configs/navigation_dqn.yaml    # also navigation_ppo, navigation_continuous_{ppo,td3}; random: _default
python record.py --config configs/navigation_dqn.yaml --checkpoint checkpoints/navigation/dqn/agent_episode_<n>.pt
python benchmark.py --challenge navigation --agents random dqn --seeds 1 --output-dir <scratch> --report <scratch>/report.md
```

Every command is in [usage.md](../../usage.md). Then continue with the [navigation
track](../navigation/sim-and-sensors.md).
