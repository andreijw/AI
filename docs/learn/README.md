# Learning reinforcement learning and robot navigation with this repository

A learning track through the code, the measurements and the mistakes of this repository: from the loop between an agent
and an environment, through ten challenges and the algorithms that solve them, to a robot that estimates where it is,
maps what it sees, plans a path and runs a trained policy on a Jetson. Every page points at the code that does the
thing it teaches, and every number on it comes from a benchmark report or a dated entry of
[CHANGELOG.md](../../CHANGELOG.md).

## Start here

**Who it is for.** Someone who can read Python and NumPy and has met the idea of training a model, but not necessarily
reinforcement learning; and someone who knows reinforcement learning and wants to see what it takes to make each agent
learn, and then to put it on a robot. It is not a replacement for a textbook: it teaches each idea where this code uses
it, with the fixes that were needed to make it work.

**What to install.** Reading needs nothing. Running a command from a page needs the virtual environment, and the
challenges that use Box2D, the PyTorch backend or the video extras need their extras: follow
[usage.md, Installation](../usage.md#installation), which is the one place the install commands live. Each challenge
page ends with one train and one record command that work as written once the environment is set up.

**What to read, in order.**

1. [Foundations](foundations.md): the episode loop, the two ways an episode ends, returns and bootstrapping, and what
   "solved" and "held" mean. Every other page uses these words as that page defines them.
2. [CartPole](challenges/cartpole.md), with its algorithms, one after another: [Policy
   gradients](algorithms/policy-gradient.md) (REINFORCE, then the Actor-Critic), [PPO](algorithms/ppo.md) and
   [DQN](algorithms/dqn.md). Four ideas on one small problem.
3. [Acrobot](challenges/acrobot.md) and [MountainCar](challenges/mountaincar.md): exploration, when the reward says
   "not yet" for a long time.
4. [Pendulum](challenges/pendulum.md), the first continuous action space, with [DDPG and
   TD3](algorithms/ddpg-td3.md) and [SAC](algorithms/sac.md).
5. [LunarLander](challenges/lunarlander.md), then [BipedalWalker](challenges/bipedalwalker.md): the larger problems.
6. [Navigation](challenges/navigation.md), then the navigation track from [the simulator and its
   sensors](navigation/sim-and-sensors.md) to [export to the robot](navigation/export-to-robot.md).

A reader short of time can take one path: foundations, CartPole with [PPO](algorithms/ppo.md), and the navigation
challenge. When a result looks wrong, [the fixes that apply to every
agent](foundations.md#fixes-that-apply-to-every-agent) are the first things to check.

## Learning path

- [Foundations](foundations.md)
- Challenges: [CartPole-v1](challenges/cartpole.md), [Acrobot-v1](challenges/acrobot.md),
  [MountainCar-v0](challenges/mountaincar.md), [Pendulum-v1](challenges/pendulum.md),
  [LunarLander-v3](challenges/lunarlander.md), [BipedalWalker-v3](challenges/bipedalwalker.md),
  [Navigation-v0](challenges/navigation.md)
- Algorithms: [Policy gradients](algorithms/policy-gradient.md), [PPO](algorithms/ppo.md),
  [DQN and its variants](algorithms/dqn.md), [DDPG and TD3](algorithms/ddpg-td3.md), [SAC](algorithms/sac.md)
- Navigation track, after the Navigation challenge: [the simulator and its sensors](navigation/sim-and-sensors.md),
  [estimation](navigation/estimation.md), [mapping](navigation/mapping.md), [SLAM](navigation/slam.md),
  [planning and control](navigation/planning.md) and [export to the robot](navigation/export-to-robot.md), where a
  trained policy reaches the Jetson

## Challenges and algorithms at a glance

One row per challenge: its action space, what makes it hard, the algorithms with a committed config in
[`configs/`](../../configs/), and who solves and holds it. The last column only points: the measured numbers are in
the row of [results.md](../results.md#which-agents-solve-which-challenge) and in the report linked in the row, and
"solves" and "holds" mean what the [glossary](#glossary) says. Challenges that share a page share its link.

| Challenge | Actions | What makes it hard | Algorithms with a committed config | Solves and holds, as results.md states it |
| :--- | :--- | :--- | :--- | :--- |
| [CartPole-v1](challenges/cartpole.md) | discrete | a time limit that is not a failure | [REINFORCE and Actor-Critic](algorithms/policy-gradient.md), [PPO](algorithms/ppo.md), [DQN](algorithms/dqn.md) | PPO, Actor-Critic and DQN solve and hold; REINFORCE solves on most seeds and holds on fewer; [report](../benchmarks/cartpole_results.md) |
| [Acrobot-v1](challenges/acrobot.md) | discrete | no sign of progress until the first success | [PPO](algorithms/ppo.md), [DQN](algorithms/dqn.md) | both solve and hold; no report, see [results.md](../results.md#which-agents-solve-which-challenge) |
| [MountainCar-v0](challenges/mountaincar.md) | discrete | sparse reward, exploration | [DQN](algorithms/dqn.md) | solves, holds on only some seeds; no report |
| [MountainCarContinuous-v0](challenges/mountaincar.md) | continuous | exploration, a cost for acting, a tempting standstill | [DDPG and TD3](algorithms/ddpg-td3.md) | both solve and hold; no report |
| [Pendulum-v1](challenges/pendulum.md) | continuous | no true end, only a time limit | [Actor-Critic](algorithms/policy-gradient.md), [PPO](algorithms/ppo.md), [DDPG and TD3](algorithms/ddpg-td3.md), [SAC](algorithms/sac.md) | DDPG, TD3 and SAC solve and hold; PPO solves, holds on most seeds; the Actor-Critic solves on some seeds and holds on none; [report](../benchmarks/pendulum_results.md) (no TD3 or SAC row) |
| [LunarLander-v3](challenges/lunarlander.md) | discrete | many-part shaped reward, long episodes | [DQN](algorithms/dqn.md) | solves and holds; no report |
| [LunarLanderContinuous-v3](challenges/lunarlander.md) | continuous | the same, with two engine throttles | [TD3](algorithms/ddpg-td3.md), [SAC](algorithms/sac.md) | both solve and hold; [report](../benchmarks/lunarlander_continuous_results.md) (TD3 only) |
| [BipedalWalker-v3](challenges/bipedalwalker.md) | continuous | four joints, long episodes, falls | [TD3](algorithms/ddpg-td3.md), [SAC](algorithms/sac.md) | SAC solves on most seeds and holds on one; TD3 solves on some and holds on none; [report](../benchmarks/bipedalwalker_results.md) |
| [Navigation-v0](challenges/navigation.md) | discrete | this repository's own simulator; sensors, obstacles | [PPO](algorithms/ppo.md), [DQN](algorithms/dqn.md) | DQN solves and holds; PPO solves, holds on few; [report](../benchmarks/navigation_results.md) |
| [NavigationContinuous-v0](challenges/navigation.md) | continuous | the same, with speed and turn rate | [PPO](algorithms/ppo.md), [TD3](algorithms/ddpg-td3.md) | TD3 solves and holds; PPO solves, does not hold; [report](../benchmarks/navigation_continuous_results.md) |

## Glossary

The one place these words are defined. Where a page uses one differently, this page is wrong or that page is: say so
in a pull request.

**Episode.** One run of the task from `reset` to the step where `terminated` or `truncated` is true.
[Foundations](foundations.md#the-episode-loop).

**Step.** One call of `env.step(action)`: the agent acts, the environment answers with the next observation and a
reward. [Foundations](foundations.md#the-episode-loop).

**Step budget.** The length of a training run in environment steps (`training.total_steps` in a config), counted over
all episodes, the one that finishes the budget included; `training.num_episodes` is used when no step budget is set.
Budgets are set against rl-zoo's, and each challenge's report names them. [Foundations](foundations.md#the-episode-loop)
and [the fixes](foundations.md#fixes-that-apply-to-every-agent) (CHANGELOG.md, 2026-10-16).

**Terminated.** The task reached a state it defines as an end (the pole fell, the lander crashed, the goal was
reached). Nothing follows, so the value after it is zero.
[Foundations](foundations.md#terminated-and-truncated).

**Truncated.** Something outside the task stopped the episode, usually a time limit. The agent was not in trouble and
the state after it still has a value, so an agent with a critic bootstraps from it.
[Foundations](foundations.md#terminated-and-truncated).

**Return.** The sum of the rewards from a step on, each discounted by `gamma` for every step it lies ahead:
`G_t = r_t + gamma * G_{t+1}`. [Foundations](foundations.md#returns-and-bootstrapping).

**Bootstrapping.** Building an estimate from an estimate: a target made of the reward and `gamma` times the critic's
own value of the next state, instead of the whole return. Lower variance, and a bias, since the estimate is part of
its own target. [Foundations](foundations.md#returns-and-bootstrapping).

**Critic.** A network that learns to predict the return from a state (`V(s)`), or from a state and an action (`Q`).
The Actor-Critic and PPO use one as a baseline and to bootstrap; DQN's network is one; DDPG's actor climbs one, and TD3
and SAC use two. [Policy gradients](algorithms/policy-gradient.md#the-idea), [DQN](algorithms/dqn.md#the-idea) and
[DDPG and TD3](algorithms/ddpg-td3.md#the-idea).

**Actor.** The network that chooses the action, the policy. The counterpart of the critic.
[Policy gradients](algorithms/policy-gradient.md#the-idea).

**On-policy.** Learns from data the current policy just collected, which is discarded once the policy has moved:
REINFORCE and the Actor-Critic use each episode once, PPO a rollout for a few passes, corrected by a clipped ratio.
[PPO](algorithms/ppo.md#the-idea).

**Off-policy.** Learns from stored transitions whatever policy produced them, so each is reused: DQN, DDPG, TD3 and
SAC. [DQN](algorithms/dqn.md#the-idea).

**Replay buffer.** The store of past transitions an off-policy agent samples mini-batches from, so that consecutive,
correlated steps are not trained on together. [DQN](algorithms/dqn.md#the-idea).

**Target network.** A slowly moving copy of a critic that builds the targets, so the estimate does not chase itself.
[DQN](algorithms/dqn.md#the-idea) and [DDPG and TD3](algorithms/ddpg-td3.md#the-idea).

**Seed.** The number that fixes a run's random numbers. A run draws the environment's, the weights', the exploration's
and the sampling's from separate streams of it. The published reports use a fixed list of seeds and `train.py` uses
another, so a single `train.py` run can land on either side of a report's solve rate.
[Foundations](foundations.md#fixes-that-apply-to-every-agent) (CHANGELOG.md, 2026-10-19).

**Solved.** A run solves a challenge when its training reward, averaged over a rolling window of consecutive episodes,
first reaches the challenge's threshold. Only full windows count, and the window is 100 episodes (CHANGELOG.md,
2026-10-16). The rewards are those the agent earned while training, exploration included. `classify_run` in
[`benchmark.py`](../../benchmark.py) decides it. [Foundations](foundations.md#solved-and-held).

**Held.** A run holds the challenge when its last window of episodes still averages at or above the threshold: it can
solve and then fall back, which is solving without holding. [Foundations](foundations.md#solved-and-held).

**Greedy evaluation.** Playing the policy's most likely action, with no exploration, on its own environment and from
fixed start states, after training or every `eval_frequency` episodes. It is reported beside the solve rate; reaching
the threshold in it is recorded but is not a solve. [Foundations](foundations.md#solved-and-held).

**Threshold.** The challenge's `solve_threshold`: the score Gymnasium or this repository calls the task done. Each
challenge page states its own. [Foundations](foundations.md#solved-and-held).
