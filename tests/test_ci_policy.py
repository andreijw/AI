"""Contract for .github: the CI rules every workflow, action and Dependabot entry must keep.

The rules come from the CI hardening brief (2026-10-21): cancel superseded PR runs, time-limit every
job, SHA-pin every action, read-only tokens, no secrets but Codecov's, no paid LLM calls in CI, required
checks that always report, short artifact retention, weekly grouped Dependabot updates.

Each rule is a pure function that returns a list of violations. It is tested on small inline
fixtures and then applied to every real file, so a new workflow is checked with no edit here.
The only edit points are the three constants below.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
GITHUB = REPO_ROOT / ".github"
WORKFLOWS = sorted((GITHUB / "workflows").glob("*.yml")) + sorted(
    (GITHUB / "workflows").glob("*.yaml")
)
ACTIONS = sorted((GITHUB / "actions").glob("*/action.yml"))

# Mirror of the required status checks in ruleset BasicProtection (id 12628586) once the maintainer has
# replaced "Ruff Linting", "MyPy Type Checking", "Bandit Security Scan" and "Markdown Linting" with
# "Checks" (2026-10-21). A required check that never reports blocks every merge.
REQUIRED_CHECKS = {"Checks", "Unit Tests"}
# The only secret any workflow may read (fork and Dependabot PRs receive none). Every workflow is held
# to this, not only pull_request ones: ci.yml is also callable through workflow_call.
ALLOWED_SECRETS = {"CODECOV_TOKEN"}
# (workflow file, job id, scope): the only write any job may hold. CodeQL must upload each PR's SARIF for
# the ruleset's code_scanning rule, and security-events cannot change code.
WRITE_ALLOWLIST = {("codeql.yml", "analyze", "security-events")}

PR_EVENTS = {"pull_request", "pull_request_target"}
_USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+)(.*)$", re.MULTILINE)
_PINNED = re.compile(r"^[\w.-]+/[\w.-]+(?:/[\w./-]+)?@[0-9a-f]{40}$")
_VERSION_COMMENT = re.compile(r"^\s*#\s*v\d+(?:\.\d+)*\s*$")
_SECRET_REF = re.compile(r"secrets\.(\w+)")
_AI_PATTERNS = re.compile(
    r"api\.openai\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com|models\.github\.ai"
    r"|api\.mistral\.ai|api\.cohere\.|openrouter\.ai"
    r"|(?:OPENAI|ANTHROPIC|CLAUDE|GEMINI|GOOGLE_AI|MISTRAL|COHERE|OPENROUTER)_API_KEY"
    r"|anthropics/claude-code-action|openai/codex-action|google-github-actions/run-gemini-cli",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- helpers


def events(workflow):
    """Trigger names, whatever shape `on:` has. PyYAML (YAML 1.1) loads the key `on` as True."""
    on = workflow.get("on", workflow.get(True))
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return set(on)
    if isinstance(on, dict):
        return set(on)
    return set()


def trigger_config(workflow, event):
    on = workflow.get("on", workflow.get(True))
    return (on.get(event) or {}) if isinstance(on, dict) else {}


def is_pr_triggered(workflow):
    return bool(events(workflow) & PR_EVENTS)


def write_scopes(permissions):
    """Scopes granted write by a `permissions:` value (mapping, `write-all`, `read-all` or {})."""
    if isinstance(permissions, str):
        return ["write-all"] if permissions == "write-all" else []
    if isinstance(permissions, dict):
        return sorted(scope for scope, level in permissions.items() if level == "write")
    return []


def jobs(workflow):
    return (workflow.get("jobs") or {}).items()


def steps(workflow):
    """Every step with its job; a composite action's `runs.steps` count as job `runs`."""
    for job_id, job in jobs(workflow):
        for step in job.get("steps") or []:
            yield job_id, job, step
    for step in (workflow.get("runs") or {}).get("steps") or []:
        yield "runs", {}, step


# ---------------------------------------------------------------- rules over one workflow


def rule_top_level_read_only(name, workflow, raw):
    if "permissions" not in workflow:
        return [f"{name}: no top-level `permissions:` (the default token may be read-write)"]
    return [
        f"{name}: top-level permissions grant {scope}"
        for scope in write_scopes(workflow["permissions"])
    ]


def rule_pr_concurrency(name, workflow, raw):
    if not is_pr_triggered(workflow):
        return []
    concurrency = workflow.get("concurrency")
    if not isinstance(concurrency, dict) or "group" not in concurrency:
        return [f"{name}: pull_request workflow has no `concurrency:` group"]
    cancel = concurrency.get("cancel-in-progress")
    if cancel in (None, False, "false"):
        return [f"{name}: pull_request workflow does not cancel superseded runs"]
    return []


def rule_job_timeouts(name, workflow, raw):
    return [
        f"{name}: job `{job_id}` has no timeout-minutes"
        for job_id, job in jobs(workflow)
        if "steps" in job and "timeout-minutes" not in job
    ]


def rule_actions_sha_pinned(name, workflow, raw):
    violations = []
    for ref, rest in _USES.findall(raw):
        ref = ref.strip("'\"")
        if ref.startswith("./"):
            continue
        if not _PINNED.match(ref):
            violations.append(f"{name}: `{ref}` is not pinned to a full commit SHA")
        elif not _VERSION_COMMENT.match(rest):
            violations.append(f"{name}: `{ref}` has no trailing `# v<version>` comment")
    return violations


def rule_jobs_cannot_write(name, workflow, raw):
    return [
        f"{name}: job `{job_id}` grants {scope}"
        for job_id, job in jobs(workflow)
        for scope in write_scopes(job.get("permissions"))
        if (name, job_id, scope) not in WRITE_ALLOWLIST
    ]


def rule_secrets(name, workflow, raw):
    violations = [
        f"{name}: reads secrets.{secret}"
        for secret in sorted(set(_SECRET_REF.findall(raw)) - ALLOWED_SECRETS)
    ]
    for form in ("secrets: inherit", "toJSON(secrets)", "secrets["):
        if form in raw:
            violations.append(f"{name}: passes secrets wholesale (`{form}`)")
    return violations


def rule_local_actions_exist(name, workflow, raw):
    return [
        f"{name}: `{ref}` does not exist in the repository"
        for ref, _ in _USES.findall(raw)
        if ref.startswith("./") and not (REPO_ROOT / ref).exists()
    ]


def rule_no_pull_request_target(name, workflow, raw):
    if "pull_request_target" in events(workflow):
        return [f"{name}: pull_request_target runs fork code with secrets and a write token"]
    return []


def rule_no_ai_calls(name, workflow, raw):
    return [
        f"{name}: references a paid LLM API or AI key (`{m}`)" for m in _AI_PATTERNS.findall(raw)
    ]


def rule_upload_retention(name, workflow, raw):
    return [
        f"{name}: upload in job `{job_id}` has no retention-days"
        for job_id, _, step in steps(workflow)
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
        and "retention-days" not in (step.get("with") or {})
    ]


def rule_checkout_drops_credentials(name, workflow, raw):
    """A job that cannot push has no use for the token checkout leaves in .git/config."""
    return [
        f"{name}: checkout in job `{job_id}` keeps credentials (set persist-credentials: false)"
        for job_id, job, step in steps(workflow)
        if str(step.get("uses", "")).startswith("actions/checkout@")
        and "contents" not in write_scopes(job.get("permissions"))
        and (step.get("with") or {}).get("persist-credentials") is not False
    ]


def rule_no_event_text_in_run(name, workflow, raw):
    """PR titles, bodies, branch and tag names and inputs are attacker-controlled; use `env:`."""
    return [
        f"{name}: job `{job_id}` interpolates {m} into a shell command"
        for job_id, _, step in steps(workflow)
        for m in re.findall(
            r"\$\{\{\s*((?:github\.(?:event\.|head_ref|ref_name)|inputs\.)[\w.]*)",
            step.get("run", ""),
        )
    ]


WORKFLOW_RULES = [
    rule_top_level_read_only,
    rule_pr_concurrency,
    rule_job_timeouts,
    rule_actions_sha_pinned,
    rule_jobs_cannot_write,
    rule_secrets,
    rule_local_actions_exist,
    rule_no_pull_request_target,
    rule_no_ai_calls,
    rule_upload_retention,
    rule_checkout_drops_credentials,
    rule_no_event_text_in_run,
]


# ---------------------------------------------------------------- rules over the whole repository


def rule_required_checks_report(workflows):
    """Every required check is the explicit `name:` of a job that runs on every pull request."""
    reporting = {
        job["name"]
        for workflow in workflows.values()
        if "pull_request" in events(workflow)
        and not {"paths", "paths-ignore"} & set(trigger_config(workflow, "pull_request"))
        for _, job in jobs(workflow)
        if "name" in job
    }
    return [
        f"required check `{check}` is not reported by any unfiltered pull_request job"
        for check in sorted(REQUIRED_CHECKS - reporting)
    ]


def rule_dependabot_weekly_and_grouped(config):
    updates = config.get("updates") or []
    violations = [
        f"Dependabot has no `{ecosystem}` entry"
        for ecosystem in ("pip", "github-actions")
        if ecosystem not in {u.get("package-ecosystem") for u in updates}
    ]
    for update in updates:
        ecosystem = update.get("package-ecosystem")
        if (update.get("schedule") or {}).get("interval") != "weekly":
            violations.append(f"Dependabot `{ecosystem}` is not weekly")
        groups = update.get("groups") or {}
        if not any(group.get("patterns") for group in groups.values()):
            violations.append(
                f"Dependabot `{ecosystem}` has no group, so each update is its own PR"
            )
    return violations


# ---------------------------------------------------------------- the rules on inline fixtures

GOOD = """
name: Good
on:
  pull_request:
  workflow_dispatch:
permissions:
  contents: read
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
jobs:
  build:
    name: Build
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@0123456789abcdef0123456789abcdef01234567 # v7.0.1
        with:
          persist-credentials: false
      - uses: ./.github/actions/docs-only
      - run: echo "$TITLE"
        env:
          TITLE: ${{ github.event.pull_request.title }}
      - uses: actions/upload-artifact@0123456789abcdef0123456789abcdef01234567 # v7
        with:
          retention-days: 3
      - uses: codecov/codecov-action@0123456789abcdef0123456789abcdef01234567 # v7.1.1
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
"""

# (rule, edit applied to GOOD, part of the expected message)
BREAKS = [
    (rule_top_level_read_only, ("permissions:\n  contents: read\n", ""), "no top-level"),
    (
        rule_top_level_read_only,
        ("contents: read\nconcurrency", "contents: write\nconcurrency"),
        "grant",
    ),
    (
        rule_top_level_read_only,
        ("permissions:\n  contents: read\n", "permissions: write-all\n"),
        "write-all",
    ),
    (rule_pr_concurrency, ("concurrency:", "x-concurrency:"), "no `concurrency:`"),
    (
        rule_pr_concurrency,
        (
            "cancel-in-progress: ${{ github.event_name == 'pull_request' }}",
            "cancel-in-progress: false",
        ),
        "does not cancel",
    ),
    (rule_job_timeouts, ("    timeout-minutes: 5\n", ""), "no timeout-minutes"),
    (
        rule_actions_sha_pinned,
        ("actions/checkout@0123456789abcdef0123456789abcdef01234567", "actions/checkout@v7"),
        "not pinned",
    ),
    (
        rule_actions_sha_pinned,
        (
            "@0123456789abcdef0123456789abcdef01234567 # v7.0.1",
            "@0123456789ABCDEF0123456789ABCDEF01234567 # v7.0.1",
        ),
        "not pinned",
    ),
    (rule_actions_sha_pinned, (" # v7.0.1", ""), "no trailing"),
    (rule_actions_sha_pinned, ("./.github/actions/docs-only", "docker://alpine:3"), "not pinned"),
    (
        rule_jobs_cannot_write,
        (
            "    timeout-minutes: 5\n",
            "    timeout-minutes: 5\n    permissions:\n      contents: write\n",
        ),
        "grants contents",
    ),
    (
        rule_jobs_cannot_write,
        ("    timeout-minutes: 5\n", "    timeout-minutes: 5\n    permissions: write-all\n"),
        "grants write-all",
    ),
    (rule_secrets, ("secrets.CODECOV_TOKEN", "secrets.GPG_PRIVATE_KEY"), "secrets.GPG_PRIVATE_KEY"),
    (rule_secrets, ("secrets.CODECOV_TOKEN", "secrets.GITHUB_TOKEN"), "secrets.GITHUB_TOKEN"),
    (rule_secrets, ("secrets.CODECOV_TOKEN", "toJSON(secrets)"), "wholesale"),
    (rule_secrets, ("secrets.CODECOV_TOKEN", "secrets['GPG_PASSPHRASE']"), "wholesale"),
    (
        rule_secrets,
        (
            "      - uses: ./.github/actions/docs-only\n",
            "      - uses: ./.github/actions/docs-only\n        secrets: inherit\n",
        ),
        "wholesale",
    ),
    (
        rule_local_actions_exist,
        ("./.github/actions/docs-only", "./.github/actions/missing"),
        "does not exist",
    ),
    (
        rule_no_pull_request_target,
        ("  pull_request:\n", "  pull_request_target:\n"),
        "pull_request_target",
    ),
    (rule_no_ai_calls, ('echo "$TITLE"', "curl https://api.anthropic.com/v1/messages"), "paid LLM"),
    (rule_no_ai_calls, ("secrets.CODECOV_TOKEN", "secrets.OPENAI_API_KEY"), "paid LLM"),
    (
        rule_upload_retention,
        ("          retention-days: 3\n", "          name: report\n"),
        "no retention-days",
    ),
    (
        rule_checkout_drops_credentials,
        ("          persist-credentials: false\n", "          fetch-depth: 2\n"),
        "keeps credentials",
    ),
    (
        rule_no_event_text_in_run,
        ('echo "$TITLE"', 'echo "${{ github.event.pull_request.title }}"'),
        "interpolates",
    ),
    (
        rule_no_event_text_in_run,
        ('echo "$TITLE"', "git push origin ${{ github.head_ref }}"),
        "interpolates",
    ),
    (rule_no_event_text_in_run, ('echo "$TITLE"', "echo ${{ inputs.ref }}"), "interpolates"),
    (
        rule_no_event_text_in_run,
        ('echo "$TITLE"', "git tag ${{ github.ref_name }}"),
        "interpolates",
    ),
]


def _apply(edit):
    old, new = edit
    assert GOOD.count(old) == 1, f"fixture edit {old!r} must match the good workflow exactly once"
    return GOOD.replace(old, new)


@pytest.mark.parametrize("rule", WORKFLOW_RULES, ids=lambda rule: rule.__name__)
def test_rule_accepts_the_good_fixture(rule):
    assert rule("good.yml", yaml.safe_load(GOOD), GOOD) == [], (
        f"{rule.__name__} rejects a valid workflow"
    )


@pytest.mark.parametrize(
    ("rule", "edit", "message"),
    BREAKS,
    ids=[f"{rule.__name__}-{i}" for i, (rule, _, _) in enumerate(BREAKS)],
)
def test_rule_rejects_a_broken_fixture(rule, edit, message):
    raw = _apply(edit)
    violations = rule("good.yml", yaml.safe_load(raw), raw)
    assert any(message in v for v in violations), (
        f"{rule.__name__} should report `{message}` after replacing {edit[0]!r}; got {violations}"
    )


def test_every_workflow_rule_has_a_breaking_fixture():
    covered = {rule for rule, _, _ in BREAKS}
    missing = [rule.__name__ for rule in WORKFLOW_RULES if rule not in covered]
    assert not missing, f"rules without a fixture proving they can fail: {missing}"


def test_writes_and_secrets_are_refused_in_scheduled_workflows_too():
    raw = (
        _apply(("  pull_request:\n", "  schedule:\n    - cron: '0 2 * * 1'\n"))
        .replace(
            "    timeout-minutes: 5\n",
            "    timeout-minutes: 5\n    permissions:\n      contents: write\n",
        )
        .replace("secrets.CODECOV_TOKEN", "secrets.DEPLOY_KEY")
    )
    workflow = yaml.safe_load(raw)
    assert rule_pr_concurrency("weekly.yml", workflow, raw) == [], (
        "concurrency is required on PR runs only"
    )
    assert rule_jobs_cannot_write("weekly.yml", workflow, raw), (
        "a scheduled job may not write either"
    )
    assert rule_secrets("weekly.yml", workflow, raw), (
        "a scheduled job may not read other secrets either"
    )


def test_codeql_security_events_is_the_only_allowed_write():
    raw = _apply(
        (
            "    timeout-minutes: 5\n",
            "    timeout-minutes: 5\n    permissions:\n      security-events: write\n",
        )
    ).replace("  build:\n", "  analyze:\n")
    workflow = yaml.safe_load(raw)
    assert rule_jobs_cannot_write("codeql.yml", workflow, raw) == [], "CodeQL's upload is allowed"
    assert rule_jobs_cannot_write("other.yml", workflow, raw), "only CodeQL's analyze job may write"


@pytest.mark.parametrize(
    ("on", "expected"),
    [
        ("on: push", {"push"}),
        ("on: [push, pull_request]", {"push", "pull_request"}),
        ("on:\n  pull_request:\n", {"pull_request"}),
    ],
)
def test_events_reads_every_shape_of_on(on, expected):
    assert events(yaml.safe_load(on)) == expected, f"`{on}` should give {expected}"


def test_required_checks_must_be_reported_by_an_unfiltered_pr_job():
    good = {"ci.yml": yaml.safe_load(GOOD.replace("name: Build", "name: Checks"))}
    good["t.yml"] = yaml.safe_load(GOOD.replace("name: Build", "name: Unit Tests"))
    assert rule_required_checks_report(good) == []

    renamed = dict(good, **{"t.yml": yaml.safe_load(GOOD.replace("name: Build", "name: Tests"))})
    assert any("Unit Tests" in v for v in rule_required_checks_report(renamed)), (
        "a renamed check never reports"
    )

    filtered = GOOD.replace("  pull_request:\n", "  pull_request:\n    paths: ['src/**']\n")
    path_filtered = dict(
        good, **{"t.yml": yaml.safe_load(filtered.replace("name: Build", "name: Unit Tests"))}
    )
    assert any("Unit Tests" in v for v in rule_required_checks_report(path_filtered)), (
        "a path-filtered required check does not report on other PRs and blocks their merge"
    )


def test_dependabot_must_be_weekly_and_grouped():
    entry = "  - package-ecosystem: {eco}\n    directory: /\n    schedule:\n      interval: {interval}\n{groups}"
    groups = "    groups:\n      all:\n        patterns: ['*']\n"
    good = "version: 2\nupdates:\n" + "".join(
        entry.format(eco=eco, interval="weekly", groups=groups) for eco in ("pip", "github-actions")
    )
    assert rule_dependabot_weekly_and_grouped(yaml.safe_load(good)) == []
    daily = good.replace("weekly", "daily", 1)
    assert any("not weekly" in v for v in rule_dependabot_weekly_and_grouped(yaml.safe_load(daily)))
    ungrouped = good.replace(groups, "", 1)
    assert any(
        "no group" in v for v in rule_dependabot_weekly_and_grouped(yaml.safe_load(ungrouped))
    )
    pip_only = good.split("  - package-ecosystem: github-actions")[0]
    assert any(
        "github-actions" in v for v in rule_dependabot_weekly_and_grouped(yaml.safe_load(pip_only))
    )


# ---------------------------------------------------------------- the rules on the real files


def _load(path):
    raw = path.read_text(encoding="utf-8")
    return yaml.safe_load(raw), raw


def test_the_real_files_are_found():
    assert "ci.yml" in {p.name for p in WORKFLOWS}, (
        "no workflows found: every real-file test would skip"
    )
    assert ACTIONS, "no composite actions found under .github/actions"


def test_composite_action_steps_are_checked_for_injection():
    action = {
        "runs": {"using": "composite", "steps": [{"shell": "bash", "run": "echo ${{ inputs.x }}"}]}
    }
    assert rule_no_event_text_in_run("action.yml", action, ""), (
        "composite `runs.steps` must be scanned"
    )


@pytest.mark.parametrize("path", ACTIONS, ids=lambda p: str(p.relative_to(GITHUB)))
def test_composite_actions_interpolate_nothing_into_run(path):
    action, raw = _load(path)
    assert rule_no_event_text_in_run(path.name, action, raw) == []


@pytest.mark.parametrize("path", WORKFLOWS + ACTIONS, ids=lambda p: str(p.relative_to(GITHUB)))
def test_file_parses_as_yaml(path):
    parsed, _ = _load(path)
    assert isinstance(parsed, dict), (
        f"{path.name} did not parse to a mapping (an unquoted `!` in `if:`?)"
    )


@pytest.mark.parametrize("rule", WORKFLOW_RULES, ids=lambda rule: rule.__name__)
@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_keeps_the_rule(path, rule):
    workflow, raw = _load(path)
    assert rule(path.name, workflow, raw) == []


@pytest.mark.parametrize("path", ACTIONS, ids=lambda p: str(p.relative_to(GITHUB)))
def test_composite_actions_pin_their_actions(path):
    _, raw = _load(path)
    assert rule_actions_sha_pinned(path.name, {}, raw) == []


def test_required_checks_report_on_every_pull_request():
    assert rule_required_checks_report({p.name: _load(p)[0] for p in WORKFLOWS}) == []


def test_dependabot_is_weekly_and_grouped():
    assert rule_dependabot_weekly_and_grouped(_load(GITHUB / "dependabot.yml")[0]) == []
