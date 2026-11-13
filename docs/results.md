# Results: Which Agents Solve Which Challenge

The measured outcome of every committed agent configuration, with the caveats that go with each number. The front
page ([README](../README.md)) has the short table; the dated entries of [CHANGELOG.md](../CHANGELOG.md) hold the full
numbers, and the per-challenge reports are in [`benchmarks/`](benchmarks/).

## Which agents solve which challenge

A run solves the challenge when its average training reward over 100 episodes reaches the threshold: 475 for
CartPole-v1, -200 for Pendulum-v1, -100 for Acrobot-v1, -110 for MountainCar-v0, 90 for MountainCarContinuous-v0, 200
for LunarLander-v3 and LunarLanderContinuous-v3, 300 for BipedalWalker-v3, 7 for Navigation-v0 and
NavigationContinuous-v0. It **holds** when the average over its last
100 episodes is still at or above the threshold. Each agent trains for a budget of environment steps set against its
rl-baselines3-zoo reference (see the config comments). Measured with the committed configs on seeds 1 to 5
(MountainCar-v0 DQN also on seeds 11 to 15) on 2026-10-18
(CartPole PPO on 2026-10-19, after time-limit cuts were bootstrapped), with evaluation on its own environment and named
random streams ([`docs/benchmarks/cartpole_results.md`](benchmarks/cartpole_results.md),
[`docs/benchmarks/pendulum_results.md`](benchmarks/pendulum_results.md); LunarLanderContinuous TD3 on 2026-11-01 in
[`docs/benchmarks/lunarlander_continuous_results.md`](benchmarks/lunarlander_continuous_results.md); BipedalWalker
SAC and TD3 on 2026-11-05 in [`docs/benchmarks/bipedalwalker_results.md`](benchmarks/bipedalwalker_results.md);
navigation on 2026-11-07 in [`docs/benchmarks/navigation_results.md`](benchmarks/navigation_results.md) and
[`docs/benchmarks/navigation_continuous_results.md`](benchmarks/navigation_continuous_results.md)).
`train.py` uses seed 42, which the published benchmarks do not include (BipedalWalker's one seed-42 run is below;
`benchmark.py` itself defaults to seeds 42 and 123 when `--seeds` is not given), so a single `train.py` run may
land on the unlucky side of these numbers:

| Challenge | Agent | Config | Budget (steps) | Seeds solved | Steps to solve | Held |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| CartPole-v1 | PPO | `cartpole_ppo.yaml` (standard PPO, annealed) | 200,000 | 5 of 5 | 72k to 90k | 5 of 5 |
| CartPole-v1 | Actor-Critic | `cartpole_actor_critic.yaml` (Adam, time-limit bootstrap); CHANGELOG.md, 2026-11-12 | 500,000 | 5 of 5 | 156k to 282k | 5 of 5 |
| CartPole-v1 | REINFORCE | `cartpole_reinforce.yaml` (Adam) | 500,000 | 4 of 5 | 255k to 376k | 3 of 5 |
| CartPole-v1 | DQN | `cartpole_dqn.yaml` (Adam, per-step exploration) | 300,000 | 5 of 5 | 144k to 183k | 5 of 5 |
| Pendulum-v1 | DDPG | `pendulum_ddpg.yaml` (rl-zoo settings, 64×64 networks) | 40,000 | 5 of 5 | 31k | 5 of 5 |
| Pendulum-v1 | TD3 | `pendulum_td3.yaml` (rl-zoo settings, 64×64 networks); CHANGELOG.md, 2026-10-25 | 40,000 | 5 of 5 | 32.6k to 33.4k | 5 of 5 |
| Pendulum-v1 | SAC | `pendulum_sac.yaml` (rl-zoo settings, 64×64 networks); no published report, CHANGELOG.md, 2026-11-04 | 40,000 | 5 of 5 | 21.8k to 23.2k | 5 of 5 |
| Pendulum-v1 | PPO | `pendulum_ppo.yaml` (2048-step rollouts, annealed) | 200,000 | 5 of 5 | 72k to 157k | 4 of 5 |
| Pendulum-v1 | Actor-Critic | `pendulum_actor_critic.yaml` (bounded mean, 8 environments x 8 steps per update, observation normalisation); CHANGELOG.md, 2026-11-12 | 1,000,000 | 2 of 5 | 523k to 745k | 0 of 5 |
| Acrobot-v1 | PPO | `acrobot_ppo.yaml` (rl-zoo settings, no normalisation) | 500,000 | 5 of 5 | 72k to 278k | 5 of 5 |
| Acrobot-v1 | DQN | `acrobot_dqn.yaml` (rl-zoo settings, one 256 layer, epsilon floor 0.02) | 100,000 | 5 of 5 | 25k to 32k | 5 of 5 |
| MountainCar-v0 | DQN | `mountaincar_dqn.yaml` (rl-zoo settings, target sync translated, epsilon floor 0.02, decaying rate, 100k buffer, observations normalised when sampled); seeds 1 to 5 and 11 to 15, 2026-11-04 | 200,000 | 10 of 10 | 141k to 197k | 3 of 10 |
| MountainCarContinuous-v0 | DDPG | `mountaincar_continuous_ddpg.yaml` (rl-zoo settings, OU noise, 64×64 networks) | 300,000 | 5 of 5 | 25k to 69k | 5 of 5 |
| MountainCarContinuous-v0 | TD3 | `mountaincar_continuous_td3.yaml` (rl-zoo settings, OU noise, 400×300 networks, PyTorch backend); CHANGELOG.md, 2026-10-27 | 300,000 | 5 of 5 | 37k to 70k | 5 of 5 |
| LunarLander-v3 | DQN | `lunarlander_dqn.yaml` (rl-zoo settings, one hidden layer of 256, epsilon floor 0.1); seeds 1 to 5, which chose the floor; held-out seeds 11 to 15: 5 of 5 reach (166k to 297k), 4 of 5 hold; CHANGELOG.md, 2026-10-27 | 300,000 | 5 of 5 | 152k to 228k | 5 of 5 |
| LunarLanderContinuous-v3 | TD3 | `lunarlander_continuous_td3.yaml` (rl-zoo settings, 400×300 networks, updates after every step, PyTorch backend); CHANGELOG.md, 2026-11-01; held-out seeds 11 to 15: 5 of 5 reach and hold, CHANGELOG.md, 2026-10-30 | 400,000 | 5 of 5 | 174k to 294k | 5 of 5 |
| LunarLanderContinuous-v3 | SAC | `lunarlander_continuous_sac.yaml` (rl-zoo settings, 400×300 networks, PyTorch backend); no published report, CHANGELOG.md, 2026-11-04; greedy evaluation 282 ± 5 against TD3's 182 ± 69 | 500,000 | 5 of 5 | 79.6k to 244.5k | 5 of 5 |
| BipedalWalker-v3 | SAC | `bipedalwalker_sac.yaml` (rl-zoo settings without gSDE, 400×300 networks, PyTorch backend); CHANGELOG.md, 2026-11-05 | 500,000 | 4 of 5 | 210k to 387k | 1 of 5 |
| BipedalWalker-v3 | TD3 | `bipedalwalker_td3.yaml` (rl-zoo settings, 400×300 networks, updates after every step, PyTorch backend, half rl-zoo's budget); CHANGELOG.md, 2026-11-05 | 500,000 | 2 of 5 | 399k to 425k | 0 of 5 |
| Navigation-v0 | PPO | `navigation_ppo.yaml` (`docs/benchmarks/navigation_results.md`, 2026-11-07) | 200,000 | 5 of 5 | 116k (mean) | 1 of 5 |
| Navigation-v0 | DQN | `navigation_dqn.yaml` (`td_loss: huber`, decaying rate; same report) | 300,000 | 5 of 5 | 227k (mean) | 5 of 5 |
| NavigationContinuous-v0 | PPO | `navigation_continuous_ppo.yaml` (`docs/benchmarks/navigation_continuous_results.md`, 2026-11-07) | 200,000 | 5 of 5 | 82k (mean) | 0 of 5 |
| NavigationContinuous-v0 | TD3 | `navigation_continuous_td3.yaml` (same report) | 100,000 | 5 of 5 | 38k (mean) | 5 of 5 |

CartPole DQN trains for 300,000 steps rather than rl-zoo's 50,000: holding a 100-episode average of 475 alone takes
47,500 steps, and rl-zoo's 50,000-step result is a greedy evaluation after training (CHANGELOG.md, 2026-10-18).
Pendulum DDPG uses rl-zoo's settings with 64×64 networks instead of 400×300, which take about 16 times longer in
NumPy; its greedy evaluation averages -158 (± 25 across the 5 seeds' means) against rl-zoo's -152 (± 94 across
episodes of one run) (CHANGELOG.md, 2026-10-19). Pendulum PPO trains on rollouts of whole episodes for 200,000
steps, twice rl-zoo's 100,000; updating on one episode at a time it solved no seed (CHANGELOG.md, 2026-10-18).
Continuous Actor-Critic does not hold -200: with 8 environments (since 2026-11-12) two seeds reach it in a
100-episode window and fall back, and the training average ends at -227.6 to -268.7 (mean -244.1 against -282.7
with one environment, better on every seed); its greedy evaluation of -184.0 ± 42.7 is worse than with one
(-170.0 ± 25.9) but beats the random agent on every seed (CHANGELOG.md, 2026-11-10 and 2026-11-12; `TESTING.md`
`TR-5`; with one environment before 2026-11-12, no seed reached -200: -268 to -298, CHANGELOG.md, 2026-11-04).
Acrobot was measured on 2026-10-22 (CHANGELOG.md, 2026-10-22); it has no published report under `docs/benchmarks/`
yet. Acrobot DQN keeps exploring with 2% random actions rather than rl-zoo's 10%: with 10% its training average ended
at -95 to -103 and held on 1 of 5 seeds; with 2% it ends at -90 to -96, and its greedy evaluation is -85 ± 6.
The floor was chosen on those same seeds 1 to 5 with no held-out seeds, so the 5 of 5 is optimistic.
MountainCar DQN learns to reach the flag but does not reliably hold -110. Its greedy policy reaches about -100 mid-run
and does not stay there; a 100,000-transition replay buffer (rl-zoo: 10,000) helps the training average. On seeds 1 to
10 the final training averages are -107 to -132 (mean -120.5, against -131.2 with 10,000) against the random agent's
-200, its greedy evaluation is -120 ± 13, and 2 of 10 seeds end at or above -110, both crossing it in their last 5,000
to 14,000 steps (CHANGELOG.md, 2026-10-25). Since 2026-11-04 its observations are normalised when a batch is
sampled: on seeds 1 to 5 and 11 to 15 every seed reaches -110 and 3 of 10 hold it, final averages -106 to -119,
greedy evaluation -110 ± 7 (CHANGELOG.md, 2026-11-04). Two hidden layers, a Huber loss or scaled inputs did
not help (CHANGELOG.md, 2026-10-23). PPO has no
MountainCar config: without reward normalisation it never reached the flag in 1,000,000 steps on any of 5 seeds.
MountainCarContinuous DDPG explores with Ornstein-Uhlenbeck noise (sigma 0.5), as rl-zoo's does; its greedy evaluation
averages 93.4 (± 0.2 across seed means) against rl-zoo's 93.5. With Gaussian noise of the same std it solved no seed:
four learned to stand still (CHANGELOG.md, 2026-10-23). It was measured on 2026-10-23 and has no published report
under `docs/benchmarks/`. MountainCarContinuous TD3 trains rl-zoo's 400×300 networks on the PyTorch backend, so it
needs `.venv-gpu`; its greedy evaluation averages 93.8 (± 0.2 across seed means), and a seed takes about 20 minutes
on the Jetson's GPU (CHANGELOG.md, 2026-10-27). Seed 42, which `train.py` uses, was measured too: it reaches 90 at
about episode 150 and holds it (last 100 episodes 92.3). It has no published report under `docs/benchmarks/` either.
LunarLander DQN uses rl-zoo's settings with one hidden layer of 256 and a squared-error loss (SB3 uses Huber), for
300,000 steps against rl-zoo's 100,000. rl-zoo's exploration floor of 0.1 held 200 on 5 of 5 seeds against 3 of 5 for
0.02. On held-out seeds 11 to 15 it reached 200 on 5 of 5 (one at 297,000 steps, near the end of the budget) and held it
on 4 of 5, greedy evaluation 238 ± 25; seed 42 reaches 200 at about episode 450 and ends at 237 (CHANGELOG.md,
2026-10-27). The random agent's final averages are -178 to -194.
LunarLanderContinuous TD3 trains rl-zoo's 400×300 networks on the PyTorch backend, so it needs `.venv-gpu`, for
400,000 steps against rl-zoo's 300,000, and updates after every environment step as rl-zoo does
(`training.update_every: 1`). Every seed in both sets reaches 200 and holds it: final 100-episode averages 202 to 248
on seeds 1 to 5 (CHANGELOG.md, 2026-11-01), and a mean of 242.8 on held-out seeds 11 to 15 (CHANGELOG.md, 2026-10-30).
Greedy evaluation still varies by seed: 58 to 251 on seeds 1 to 5 (mean 182 ± 69), with only 2 of 5 at 200 or more. A
seed takes about 54 minutes alone on the Jetson's GPU (CHANGELOG.md, 2026-11-01). Before that change the config
updated after each episode, and held 200 on 3 and 4 of 5 seeds (CHANGELOG.md, 2026-10-28 and 2026-10-30). The random
agent's final averages are -214 to -228.
BipedalWalker SAC and TD3 both train rl-zoo's 400×300 networks on the PyTorch backend for 500,000 steps, so they need
`.venv-gpu`; SAC explores without rl-zoo's gSDE, and TD3 gets half rl-zoo's 1,000,000 steps so the two share a budget
(the configs' headers list the departures). SAC's greedy evaluation is 324.4 ± 1.9 (321.3 to 326.9), at 300 or more
on 5 of 5 seeds, though its training average, exploration included, holds 300 only on seed 1 (final averages 251 to
307). TD3's greedy evaluation is 300.1 ± 12.3 (281.6 to 317.7), at 300 or more on 2 of 5, and its final training
averages are 242 to 295. SAC's evaluation beats TD3's on all 5 paired seeds. With three seeds sharing the Jetson's GPU
a seed took about 1.7 hours for SAC and 2.1 hours for TD3; one alone takes about 0.9 and 1.3 hours, estimated from a
13,000-step probe rather than a full run (CHANGELOG.md, 2026-11-05). Seed 42, which `train.py` uses, was trained once
with SAC: its final 100-episode average is 302.4 after 636 episodes (500,412 steps, 4,752 seconds). The random agent's
final averages are -99 to -101.
