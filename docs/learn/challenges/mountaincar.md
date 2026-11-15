# MountainCar-v0 and MountainCarContinuous-v0

Drive an underpowered car out of a valley by rocking it back and forth. The classic hard-exploration problem: the engine
is too weak to climb the hill directly, and until the car first reaches the flag nothing in the reward points the way.

## The task

A car sits in a valley between two hills. Its engine cannot push it up the right-hand hill in one go; it must first
drive up the left-hand slope, then use gravity and momentum to reach the flag on the right. Both versions share the
physics and differ in the action and the reward. This repository runs Gymnasium's
[`MountainCar-v0`](https://gymnasium.farama.org/environments/classic_control/mountain_car/) and
[`MountainCarContinuous-v0`](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) through
one wrapper, `rl_mountaincar.environments.mountaincar_env.MountainCarEnv`, built by
`rl_mountaincar.environments.make_env.make_env_from_config`; it adds optional observation noise only.

## Observation, action and reward

- **Observation** (both): two numbers, `Box(2,)`: the car's position and its velocity.
- **Action**: `MountainCar-v0` has `Discrete(3)`: accelerate left, do nothing, accelerate right.
  `MountainCarContinuous-v0` takes one force in `[-1, 1]`.
- **Reward**: `MountainCar-v0` pays `-1` every step until the flag, so the return is minus the episode's length.
  `MountainCarContinuous-v0` charges `0.1 · action²` every step and pays `+100` on reaching the flag, so standing still
  scores about zero and is a tempting local optimum.

## How an episode ends and the solve threshold

Both are **terminated** when the car reaches the flag, and **truncated** by the time limit: `200` steps for
`MountainCar-v0`, `999` for the continuous one ([terminated and truncated](../foundations.md#terminated-and-truncated)).

The thresholds are Gymnasium's `reward_threshold`s: -110 for `MountainCar-v0`
(`rl_mountaincar.benchmark.MountainCarChallenge.solve_threshold`) (CHANGELOG.md, 2026-10-23), which asks for the flag
within about 110 steps on average, exploration included; and 90 for the continuous version
(`rl_mountaincar.benchmark.MountainCarContinuousChallenge.solve_threshold`) (CHANGELOG.md, 2026-10-23), the +100 bonus
less the fuel spent.

## Who solves it

DQN learns `MountainCar-v0` but does not reliably hold -110 (CHANGELOG.md, 2026-11-12); DDPG and TD3 solve the
continuous version. The table is in [results.md](../../results.md#which-agents-solve-which-challenge).

- **MountainCar-v0, DQN.** No benchmark report; measured on seeds 1–5 and 11–15 (CHANGELOG.md, 2026-11-04), "greedy
  evaluation -110 ± 7" across the ten; as the 2026-11-12 entry sums that run up, "all of seeds 1 to 5 and 11 to 15 reach
  -110 but 3 of 10 hold it (final averages -106 to -119)" (CHANGELOG.md, 2026-11-12). The random agent scores -200
  (CHANGELOG.md, 2026-10-23).
- **MountainCarContinuous-v0, DDPG.** No benchmark report; measured on seeds 1–5 (CHANGELOG.md, 2026-10-23): "5 of 5
  reach 90 at 25,000 to 69,000 steps (episodes 112 to 252) and hold it, final 100-episode training averages 91.0 to
  93.2, greedy evaluation 93.4 ± 0.2 across seed means against rl-zoo's 93.5".
- **MountainCarContinuous-v0, TD3.** No benchmark report; measured on seeds 1–5 (CHANGELOG.md, 2026-10-27): "5 of 5
  reach 90 at 36,800 to 70,500 steps (episodes 125 to 163) and hold it, final 100-episode training averages 92.3 to
  93.9, greedy evaluation 93.8 ± 0.2 across seed means". It trains rl-zoo's 400×300 networks on the PyTorch backend, so
  it needs `.venv-gpu`; a seed took about 20 minutes alone on the Jetson's GPU. Seed 42, which `train.py` uses, reaches
  90 at about episode 150 and holds it with a last-100 average of 92.3 (CHANGELOG.md, 2026-10-27).

## What was hard

- **The target interval's unit.** rl-zoo counts its target-network interval in environment steps, so its 600 is 300
  gradient steps here; left at 600 gradient steps, seeds 1 to 5 ended at -157 to -165, at 300 at -141 to -148. The
  exploration floor is 0.02 instead of 0.07 (CHANGELOG.md, 2026-10-23).
- **Plausible add-ons that lost.** Scaled inputs, a learning rate of 0.001, updates every 16 steps mid-episode, a Huber
  loss, two hidden layers of 256 and 300,000 steps were ruled out on seeds 1 to 3; PPO with rl-zoo's settings and no
  reward normalisation never reached the flag in 1,000,000 steps on any of seeds 1 to 5, so it has no config
  (CHANGELOG.md, 2026-10-23). n-step returns and Double DQN did no better than the decaying learning rate (CHANGELOG.md,
  2026-10-23).
- **A buffer that forgets.** On seed 4 the greedy policy scored -100 to -107 between episodes 550 and 700, then about
  -122 to -131: a 10,000-transition buffer holds only the last 85 or so episodes, and once the car stops failing it
  stops seeing failure. At 100,000 the final training average was better on all 10 seeds (mean -120.5 against -131.2),
  the greedy evaluation -120 ± 13, and 2 of 10 seeds ended at or above -110, both crossing it only in their last 5,000
  to 14,000 steps (CHANGELOG.md, 2026-10-25).
- **Normalised inputs.** Normalising each replayed batch made every seed reach -110 (CHANGELOG.md, 2026-11-04).
- **A documented limit.** Polyak averaging, a higher learning rate, a batch of 256, prioritised replay, Double and
  Dueling DQN each held no more seeds. The -110 rule judges a training average with exploration in it; rl-zoo's -100.8
  is a greedy evaluation, which the agent already reaches (CHANGELOG.md, 2026-11-12).
- **Noise that does not push.** On the continuous version DDPG with Ornstein-Uhlenbeck noise of sigma 0.5, rl-zoo's own
  setting, solved 5 of 5 seeds; with Gaussian noise of the same 0.5 it solved 0 of 5, four seeds learning to stand still
  (final averages about -23) (CHANGELOG.md, 2026-10-23). Independent noise averages out; correlated noise holds a
  direction long enough to climb. See [DDPG and TD3](../algorithms/ddpg-td3.md).

## See it

Frames and clips of a trained agent and of the random agent are added with the docs media pull request.

## Try it

```bash
python train.py --config configs/mountaincar_dqn.yaml                     # learns, does not reliably hold -110
python train.py --config configs/mountaincar_continuous_ddpg.yaml         # OU noise, NumPy
.venv-gpu/bin/python train.py --config configs/mountaincar_continuous_td3.yaml   # PyTorch backend
python record.py --config configs/mountaincar_dqn.yaml --checkpoint checkpoints/mountaincar/dqn/agent_episode_<n>.pt
```

Every command is in [usage.md](../../usage.md); [DQN](../algorithms/dqn.md) and [DDPG and
TD3](../algorithms/ddpg-td3.md) explain the agents.
