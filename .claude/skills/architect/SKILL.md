---
name: architect
description: >
  Reviews a plan or a piece of structure on the RL framework before it is built on: which package each piece
  belongs in, Python type and class design, inheritance against composition for agents, NumPy and Gymnasium
  hazards, and whether the next agent or challenge will need new code in old files. Use this on every
  implementation plan before the maintainer sees it, whenever a change adds a module, class, agent, wrapper
  or config key, moves logic between packages, or refactors across agents, and when asked "where should this
  live", "is this the right design", or about best practices. It runs brooks-lint's architecture audit for
  the general rubric and adds what only this codebase knows.
---

# Architect

The cheapest time to fix a design is before the code exists. This skill is a second opinion on shape: it
reads a plan (or existing structure) and says what will hurt later, and what is more machinery than the
problem needs. It does not edit code.

## How to run it

Run it in a fresh agent when reviewing your own plan, because the author of a plan reads it charitably. Give
the agent the plan text, the roadmap item and this file's path; it cannot see the conversation.

1. Load `brooks-lint:brooks-audit` and apply it to the modules the plan touches. It brings the general
   rubric (module depth, dependency direction, change propagation, accidental complexity) and reads
   `.brooks-lint.yaml`, which holds this project's own risks C1 to C6. If the plugin is not installed, say
   so and carry on with the lens below.
2. Apply the project lens below. This is the part the plugin cannot know.
3. Report in the format at the end.

Judge both directions. A missing abstraction and a speculative one are both findings; the second is more
common in plans written by an eager author.

## The shape of this codebase

| Where | Holds | May import |
| :--- | :--- | :--- |
| `src/rl_core/agents/` | Algorithms: networks, update rules, `ReplayBuffer` | NumPy and the standard library. No Gymnasium, no challenge package |
| `src/rl_core/training/` | `Trainer`: the episode loop, evaluation, checkpoint cadence | Gymnasium as an interface only; it never names an environment or an algorithm |
| `src/rl_core/utils/` | Config, logging, video, plots | The I/O libraries (PyYAML, Matplotlib, moviepy) stay here |
| `src/rl_core/challenges/` | `BenchmarkChallenge`, the challenge registry and `discover_challenges` | `rl_core`; it finds challenge packages by scanning, never by name |
| `src/rl_<challenge>/` | One environment: wrapper, noise, domain randomization, `make_env` factories, the `BenchmarkChallenge` adapter in `benchmark.py` | `rl_core`; never another challenge, never `src/benchmark.py` |
| `src/benchmark.py` | The benchmark runner and its command line | `rl_core`; challenges only through `discover_challenges` |
| `train.py`, `record.py` | Command line: parse, build, call | Everything; they hold no algorithm logic |

Dependencies point one way: scripts and challenges import `rl_core`, never the reverse. No file in
`rl_core/agents/` imports Gymnasium today, which is what lets an agent be tested with a hand-built batch in
milliseconds. A plan that breaks that needs a very good reason. Layout and extension steps are in
ARCHITECTURE.md; read its **Extension Guidelines** before judging a plan that adds an agent or a challenge.

## Design rules, and why

**The maths belongs to the agent, the loop to the trainer.** `Trainer._run_episode` collects one episode
and hands every agent the same batch (`observations`, `actions`, `rewards`, `next_observations`, `dones`).
It does not know which algorithm it is driving. An agent that steps an environment, or a trainer that
branches on the agent's type, has crossed the line. If a new algorithm needs something the batch lacks,
the plan widens the batch for everyone and says so.

**An agent is a registry entry.** Agents are looked up by name in each challenge's `AGENT_CLASSES` and
`CONFIG_MAP`; `train.py`, `record.py` and the benchmark build them through the challenge's `make_agent`. A
new agent today touches its own file, `rl_core/agents/__init__.py`, those maps, a YAML config, the `ALL_AGENTS`
list in `tests/test_agent_harness.py` and its own test file. That is already more places than it should be.
Ask of any plan: when the next agent arrives, how many files change? A plan that adds another place to keep
in step (an `if agent_type == "ddpg"`, an `isinstance` check, a second map) is a finding; a plan that
removes one is welcome if it is in scope.

**Hyperparameters are config.** An agent reads each one in `__init__` with `config.get(name, default)`,
validates it there, and fails with a `ValueError` that names the key. No tuning literal inside `update`.
A new key needs a default, an entry with a comment in each `configs/*.yaml` that uses the agent, and a
test for a bad value. A key that changes meaning needs a note on what happens to configs people already
have.

**Share through a base class once there are two real users.** `ActorCriticBase` and
`ContinuousActorCriticBase` own the network, the forward pass, save and load; a subclass supplies `update`.
That is the pattern for a family of agents. Stateless maths shared across families (returns, a softmax, an
optimizer step) is a module-level function, not a mixin. Two occurrences are a note; the third is the
signal to extract. An abstract base with one subclass and no second one on the roadmap is ceremony.

**Randomness is owned and seeded.** Every object that draws random numbers holds its own generator from
`rl_core.seeding.derive_rng(seed, stream)`, one named stream per role: weights, exploration (action sampling,
noise, epsilon), replay and minibatch order, evaluation. No `np.random.seed`, no bare `np.random.*` calls, and no
`default_rng(seed)` in an agent: Gymnasium seeds start states exactly like `default_rng(seed)`, and configs give
agent and environment the same seed. Environments are seeded through the wrapper's constructor seed, applied on
the first reset and continued after that. A plan that adds a source of randomness says where its seed comes from
and which stream it uses; `tests/test_random_streams.py` fails if two generators of an agent repeat each other.

**`terminated` and `truncated` are different facts.** The trainer's batch carries both `dones` (terminated
or truncated) and `terminated`. For Monte Carlo returns the difference is harmless. For anything that
bootstraps from the next state's value (DQN, DDPG, TD targets, GAE) a time-limit cut is not a terminal state,
and treating it as one teaches the critic that the future is worth nothing at the limit. Pendulum never
terminates: every episode ends by truncation at 200 steps. DQN and DDPG store `terminated` in their replay
buffers; both PPO agents add γ·V(next state) to the last reward of each cut episode in a rollout, through
`rl_core.agents.returns.episode_end` (discrete PPO treated every episode end as terminal until 2026-10-19). A
plan for a new bootstrapping agent must say how it handles this.

**A checkpoint is a file format.** Weights are saved as NumPy `.npz` archives (the trainer names them
`agent_episode_<n>.pt`; `BaseAgent._resolve_save_path` and `_resolve_load_path` own the extension rules).
A plan that adds, renames or reshapes a saved array says how a checkpoint written before the change loads.
Never `pickle`, and never `allow_pickle=True`: a checkpoint is a file someone may have downloaded.

**Numerical safety is designed in.** State the shape and dtype of every array that crosses a function
boundary. Subtract the max before a softmax, add an epsilon inside a log or under a division, clip what can
explode, and decide what happens when a standard deviation is zero (one-step episodes, equal returns).

## Python type and class design

- Type hints on every public function and method, with built-in generics (`dict[str, float]`,
  `list[int]`) and `X | None`. mypy runs in a lenient mode here (`disallow_untyped_defs = false`), so a
  missing hint will not fail CI; the review has to catch it.
- `ABC` and `@abstractmethod` for a family with shared state; `typing.Protocol` for a seam that only needs
  a shape (as `TrainingProgress` in `rl_core/utils/video.py`). An interface earns its place with a second
  implementer or a test seam.
- Keep the public surface small: a leading underscore for helpers, and export through `__init__.py` only
  what callers need. Test-only access does not widen a production class.
- No mutable default arguments, no module-level mutable state beyond the registries, and no function that
  mutates the config dict it was handed: the benchmark builds several agents from related configs.
- Validate at construction, not at step ten thousand. Raise `ValueError` with the key, the value and what
  was expected.
- Style, import order and naming belong to Ruff (CONTRIBUTING.md §2); do not spend findings on them.

## Cheap future-proofing

These are on the roadmap, cheap to respect now and expensive to retrofit:

- **Off-policy continuous control (DDPG).** `ReplayBuffer` and the batch should not assume an action is a
  single integer.
- **Frame stacking and partial observability.** Dimensions come from the environment's spaces, never a
  literal `4` or `3`.
- **Curriculum scheduling.** Wrapper parameters (noise, randomization ranges) that can be changed between
  episodes, not only at construction.
- **C header and ONNX export.** Weights reachable by name with a forward pass plain enough to transcribe,
  and a known dtype.

Do not design for anything else that is not on the roadmap.

## Reviewing a plan

Check that the plan:

- names the existing base classes and helpers it reuses, and shows it searched for them
- says which package each new file goes in, and why any algorithm logic ends up outside `rl_core/agents/`
- names the tests that will lock the behaviour in, including the gradient check for any hand-derived
  backward pass (`assert_gradients_match` in `tests/helpers.py`)
- has walked any hyperparameters through a real run and, for a new algorithm or update rule, contains what
  the `algorithm-crafter` skill asks for: the variant, the shapes, the derivation, what is held constant
- is small enough for one PR, or is split into several that each leave `develop` working
- calls out config schema, checkpoint format, dependency and `.github/` changes
- says how it will be shown that the agent still learns, and who runs that

## Report

Lead with the verdict: **Ready**, or **Revise** with the changes that would make it ready.

Then findings, most severe first, each as: **Symptom** (what the plan or code does) → **Source** (the rule
or the C-number it breaks) → **Consequence** (what goes wrong, concretely, and when) → **Remedy** (the
smallest change that fixes it). Mark each Critical, Warning or Suggestion. Name files and existing helpers
by path. End with anything you could not judge.
