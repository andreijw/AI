---
name: algorithm-crafter
description: >
  Turns a reinforcement learning algorithm into a pure NumPy agent on this framework: pins down the exact
  variant, writes the update equations and array shapes, derives the backward pass by hand, fits it to the
  agent base classes, and registers it everywhere an agent has to be known. Use this whenever the user asks
  to implement or add an algorithm or agent (DDPG, TD3, SAC, double or dueling DQN, GAE, n-step returns,
  a new exploration scheme), to add a new network layer, loss or optimizer, to change an existing update
  rule, or asks how an algorithm's gradients work in this codebase. It is the planning and implementation
  detail inside the dev loop for algorithm work.
---

# Algorithm crafter

There is no autograd here. Every gradient is derived by a person and typed in, and a wrong one rarely
crashes: the agent just learns worse, and the blame lands on hyperparameters. So algorithm work is done on
paper first, in a form a test can check, and only then in code.

This skill supplies the plan and the implementation discipline for algorithm work. The surrounding process
(roadmap item, plan review, gates, hand-off) is the `dev-loop` skill's.

## 1. Pin down the variant

"DDPG" or "PPO" names a family. In the plan, write:

- the source being followed (paper, section, or a named reference implementation)
- each update equation that will be implemented, in the plan's own notation
- every choice the source leaves open, and the choice made: advantage estimator, normalisation, loss
  reduction (mean or sum), target update (hard every N, or Polyak with τ), exploration noise, clipping
- what is deliberately left out, matching the roadmap item's **Out of scope**

Check anything from Gymnasium or NumPy in Context7 rather than from memory.

## 2. Write the shapes

A table with one row per array in the forward pass, the loss and the backward pass: name, shape, dtype,
and what it means. Use symbols for sizes (`B` batch, `O` observation, `H` hidden, `A` action). Decide here
whether a per-sample quantity is `(B,)` or `(B, 1)` and keep it that way; mixing the two is the classic
silent bug.

## 3. Derive the backward pass

Work from the scalar loss back to each parameter, one line per step, and mark what is **constant**: TD
targets, advantages, returns, stored log-probabilities and anything from a target network are data. Useful
results, each of which a gradient check should confirm in this code:

| Piece | Gradient |
| :--- | :--- |
| Softmax policy, loss `-log π(a)` | with respect to logits: `π - onehot(a)` |
| Gaussian policy, `log p(a) = -½((a-μ)/σ)² - log σ - ½ log 2π` | with respect to `μ`: `(a-μ)/σ²`; with respect to `log σ`: `((a-μ)/σ)² - 1` |
| Gaussian entropy, `Σ (log σ + ½ log 2πe)` | with respect to `log σ`: `1` |
| Squared error `½(v - y)²` | with respect to `v`: `v - y` |
| ReLU | upstream gradient where the pre-activation was positive, else `0` |
| tanh | upstream gradient times `1 - tanh²` |
| Linear layer `z = xW + b` | `dW = xᵀ dz`, `db = Σ dz` over the batch, `dx = dz Wᵀ` |
| Clipped term `min(r·A, clip(r)·A)` | flows through whichever branch is the minimum; zero through a clipped one |

State the sign convention once: the code minimises a loss, so an objective to maximise enters negated.
State the reduction once: averaging over the batch divides every gradient by `B`.

## 4. Fit the framework

- **Family.** A discrete actor-critic extends `ActorCriticBase`; a continuous one extends
  `ContinuousActorCriticBase`; anything else extends `BaseAgent`. Read the base class before writing: the
  network, forward pass, optimizer step and save/load may already exist.
- **Contract.** `select_action(observation, training)` returns an `int` for a discrete space and a
  float array for a continuous one, and is greedy (or the clipped mean) when `training` is false. A
  continuous training action of a Gaussian policy is the unclipped sample, so the batch holds the
  action its log-probability belongs to; the environment wrapper clips the torque it applies. A
  deterministic policy (DDPG) has no log-probability: it stores the clipped action it executed, so its
  critic learns the value of what the environment received. `update(batch)`
  takes the trainer's batch and returns `dict[str, float]` with every quantity worth watching.
  `save` and `load` round-trip every array and counter the agent needs to resume.
- **Separate computing from applying.** One method computes the loss and gradients for a batch and changes
  nothing; another applies the step. This is what makes a gradient check possible, and it keeps the
  optimizer swappable. Everything held constant when differentiating is an argument of that method.
  `ReinforceAgent._loss_and_grads`, `ActorCriticAgent._loss_and_grads` and
  `PPOAgent._epoch_loss_and_grads` are the pattern to follow.
- **Hyperparameters** come from `config.get(name, default)` in `__init__`, validated there.
- **Randomness** comes from the agent's own generators, one per role, each `derive_rng(config.get("seed"),
  <stream>)` from `rl_core.seeding` (weights, exploration, replay or minibatch order); add the agent to
  `tests/test_random_streams.py`.
- **Episode ends.** The batch has one `dones` flag covering both `terminated` and `truncated`. If the
  algorithm bootstraps, say in the plan what it does at a time limit.
- **Stability**: subtract the max before a softmax, bound a log standard deviation, add an epsilon inside
  a log and under a division, clip a gradient or a target where the source does.
- Match the dtype, initialisation style and naming (`_W1`, `_b1`, …) of the family being extended.

## 5. Test in this order

Each step is cheap and rules out a class of error before the next (details in the `test-crafter` skill):

1. Contract: add the agent to `ALL_AGENTS` in `tests/test_agent_harness.py`.
2. Gradient check with `assert_gradients_match` from `tests/helpers.py`, on a tiny network.
3. Values worked out by hand for returns, targets and advantages.
4. Direction and overfitting: a favoured action becomes more likely; the loss on a fixed batch falls.
5. Edges and stability: one-step episode, empty batch, extreme logits, zero spread.
6. Determinism and save/load.

## 6. Register it everywhere

A new agent is known in more than one place. Miss one and it works in a test and not from the command
line:

- `src/rl_core/agents/__init__.py` (import and export)
- `AGENT_CLASSES` and `CONFIG_MAP` in each challenge's `benchmark.py` that supports it. `train.py`,
  `record.py` and the benchmark build agents through the challenge's `make_agent`, so they need no edit
- `configs/<challenge>_<agent>.yaml`, every key commented with its meaning and unit
- `ALL_AGENTS` in `tests/test_agent_harness.py`, and `tests/test_<agent>_agent.py`
- docs, through the `doc-sync` skill: the class diagram in ARCHITECTURE.md, the README agent list, a
  `TR-*` row in TESTING.md, GEMINI.md §2

## 7. Choose starting hyperparameters

Start from the source's published values for the same or a similar task, then walk the numbers for this
trainer: it updates once per episode with that episode's transitions, which is not what most references
assume. Record where each value came from as a comment in the YAML. The committed values are the
maintainer's to approve.

## 8. Show that it learns

Use the `experiment-runner` skill: several seeds, against the `random` floor and the nearest existing agent
on the same challenge, with the roadmap item's target as the claim. If it does not learn, use
`training-doctor` before touching a hyperparameter.
