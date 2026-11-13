# Results: Which Agents Solve Which Challenge

The measured outcome of every committed agent configuration; the caveats behind each number are on each challenge's page
under [`learn/challenges/`](learn/challenges/) (listed after the table). The front page ([README](../README.md)) has the
short table; the dated entries of [CHANGELOG.md](../CHANGELOG.md) hold the full numbers, and the per-challenge reports
are in [`benchmarks/`](benchmarks/).

## Which agents solve which challenge

A run solves the challenge when its average training reward over 100 episodes reaches the threshold: 475 for
CartPole-v1, -200 for Pendulum-v1, -100 for Acrobot-v1, -110 for MountainCar-v0, 90 for MountainCarContinuous-v0, 200
for LunarLander-v3 and LunarLanderContinuous-v3, 300 for BipedalWalker-v3, 7 for Navigation-v0 and
NavigationContinuous-v0. It **holds** when the average over its last 100 episodes is still at or above the threshold.
Each agent trains for a budget of environment steps set against its rl-baselines3-zoo reference (see the config
comments). Measured with the committed configs on seeds 1 to 5 (MountainCar-v0 DQN also on seeds 11 to 15) on 2026-10-18
(CartPole on 2026-11-12, after time-limit cuts were bootstrapped in PPO and Actor-Critic; the Pendulum report on
2026-11-12, with the Actor-Critic on 8 environments), with evaluation on its own environment and named random streams
([`docs/benchmarks/cartpole_results.md`](benchmarks/cartpole_results.md),
[`docs/benchmarks/pendulum_results.md`](benchmarks/pendulum_results.md); LunarLanderContinuous TD3 on 2026-11-01 in
[`docs/benchmarks/lunarlander_continuous_results.md`](benchmarks/lunarlander_continuous_results.md); BipedalWalker SAC
and TD3 on 2026-11-05 in [`docs/benchmarks/bipedalwalker_results.md`](benchmarks/bipedalwalker_results.md); navigation
on 2026-11-07 in [`docs/benchmarks/navigation_results.md`](benchmarks/navigation_results.md) and
[`docs/benchmarks/navigation_continuous_results.md`](benchmarks/navigation_continuous_results.md)). `train.py` uses seed
42, which the published benchmarks do not include (BipedalWalker's one seed-42 run is on [its
page](learn/challenges/bipedalwalker.md); `benchmark.py` itself defaults to seeds 42 and 123 when `--seeds` is not
given), so a single `train.py` run may land on the unlucky side of these numbers:

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

What lies behind each row, from the reasons a config departs from rl-zoo to the runs that did not solve and the
measurements that have no published report, is on each challenge's page:
[CartPole](learn/challenges/cartpole.md), [Acrobot](learn/challenges/acrobot.md),
[MountainCar](learn/challenges/mountaincar.md), [Pendulum](learn/challenges/pendulum.md),
[LunarLander](learn/challenges/lunarlander.md), [BipedalWalker](learn/challenges/bipedalwalker.md) and
[Navigation](navigation.md).
