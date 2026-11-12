# Releasing a Distribution

The repository holds three distributions under `packages/`: `rl-core`, `rl-navigation` and `rl-challenges`
([ARCHITECTURE.md, Distributions](../ARCHITECTURE.md#distributions)). Each has its own version, SemVer 0.x from 0.1.0
(in 0.x a minor version may break the API), and its own `CHANGELOG.md`. A release is a GitHub Release made from a
pushed tag; nothing goes to PyPI.

## Installing a release

By the release's wheel URL, naming **every** `rl-*` wheel the install needs in one command. These names are not on
PyPI, so there is no index to fall back on: a requirement on `rl-core` that no URL in the command satisfies sends pip
to PyPI, where it fails today and where anyone could later register the name. A direct URL is the only candidate pip
considers for its name.

```bash
R=https://github.com/andreijw/AI/releases/download

# A game project: the framework
pip install $R/rl-core-v0.1.0/rl_core-0.1.0-py3-none-any.whl

# The robot: the navigation libraries, NumPy only (or by git tag, as docs/navigation.md shows)
pip install $R/rl-navigation-v0.1.0/rl_navigation-0.1.0-py3-none-any.whl

# The challenges: they require rl-core, so its wheel goes in the same command
pip install $R/rl-core-v0.1.0/rl_core-0.1.0-py3-none-any.whl \
    $R/rl-challenges-v0.1.0/rl_challenges-0.1.0-py3-none-any.whl

# The navigation challenge: rl-navigation's challenge extra requires rl-core too
pip install $R/rl-core-v0.1.0/rl_core-0.1.0-py3-none-any.whl \
    "rl-navigation[challenge] @ $R/rl-navigation-v0.1.0/rl_navigation-0.1.0-py3-none-any.whl"
```

The release's `SHA256SUMS` lists each file's checksum (`sha256sum -c SHA256SUMS` after downloading them).
`pip install -U` of an `rl-*` distribution is not safe: it asks PyPI, where these names are not registered, so
anyone could publish one. Upgrade by installing the new release's URL instead.

## Making a release

1. **In a pull request**, bump the version where it is defined, once per distribution: `__version__` in
   `rl_core/__init__.py`, `rl_navigation/__init__.py` or (for all six challenge packages) `rl_cartpole/__init__.py`;
   each pyproject reads it with `version = {attr = ...}`. Turn the package CHANGELOG's `## [Unreleased]` notes into a
   `## [<version>] - <date>` section. `tests/test_packages.py` checks that the section exists, and that every
   `rl-core~=0.x.0` requirement of the other distributions still admits rl-core's `__version__`: a breaking bump that
   forgets them fails CI (below).
2. **After it merges**, the maintainer pushes a signed tag on the merge commit:

   ```bash
   git fetch origin && git tag -s rl-navigation-v0.1.0 origin/develop -m "rl-navigation 0.1.0"
   git push origin rl-navigation-v0.1.0
   ```

3. **The Release workflow** (`.github/workflows/release.yml`) runs on the tag: it checks that the tagged commit is on
   `develop`, that the tag equals the distribution's version, that its CHANGELOG has the section, and that every
   distribution it requires has a tag at its in-tree version admitted by the requirement
   (`scripts/release.py check`); runs the tests; builds the sdist and wheel; installs the wheel, with the released
   wheels of what it requires, in a fresh environment and imports every module a plain install can
   (`scripts/release.py smoke`; rl-navigation may load only NumPy); writes `SHA256SUMS`; and opens a **draft**
   GitHub Release with the files and the CHANGELOG section as notes.
4. **The maintainer** reads the draft and publishes it. A failed run leaves nothing public; fix the cause in a pull
   request, then delete and push the tag again (both theirs). A run on a tag whose draft exists brings the draft in
   line with it (notes, title, prerelease flag, files; files it did not build are removed); a published release is
   never changed. The workflow does not verify the tag's signature: the tag ruleset below limits who can push one.

**Order.** The first releases: tag `rl-core-v0.1.0` and publish its draft, then tag `rl-navigation-v0.1.0` and
`rl-challenges-v0.1.0`. `check` refuses a distribution while anything it requires, its extras included, "has no tag"
(so rl-navigation, whose `challenge` extra requires rl-core, waits for rl-core's tag), and a run downloads the
*published* releases of what a plain install requires, which the workflow's read-only token cannot see while they
are drafts: rl-challenges' run stops with "has no published release" until rl-core's draft is published.

A prerelease (`rl-core-v0.2.0rc1` or `.dev1`, PEP 440) is marked as one. The `Packages` CI job runs the same smoke test
on every pull request, with the wheels built from that pull request, and the robot-side tests on NumPy 1.26 (ROS 2
Jazzy's).

## A breaking rl-core change

`rl-challenges` and `rl-navigation[challenge]` require `rl-core~=0.1.0`, which stops before 0.2. One pull request
bumps `rl-core` to 0.2.0, the dependents' requirements to `~=0.2.0` and the dependents' own minor versions
(`tests/test_packages.py` fails while a requirement excludes the in-tree `rl-core`). Tag `rl-core-v0.2.0` and
publish its draft first: the dependents' runs refuse to release against an `rl-core` that is not released.

## Repository settings (the maintainer's)

Two settings protect the releases; both are on (2026-11-12). Each is one command with an owner's token
(`gh auth login`), or a few clicks.

- **Immutable releases**: a published release's tag and files can no longer be changed or replaced.

  ```bash
  gh api --method PUT repos/andreijw/AI/immutable-releases
  gh api repos/andreijw/AI/immutable-releases   # {"enabled":true,...}
  ```

  Or Settings → General → Releases → tick **Enable release immutability**.

- **A tag ruleset**: only repository admins may create, move or delete a release tag (`*-v*`).

  ```bash
  gh api --method POST repos/andreijw/AI/rulesets --input - <<'EOF'
  {
    "name": "Release tags",
    "target": "tag",
    "enforcement": "active",
    "conditions": {"ref_name": {"include": ["refs/tags/*-v*"], "exclude": []}},
    "rules": [{"type": "creation"}, {"type": "update"}, {"type": "deletion"}],
    "bypass_actors": [{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}]
  }
  EOF
  gh api repos/andreijw/AI/rulesets --jq '.[] | {id, name, target}'
  ```

  Or Settings → Rules → Rulesets → New ruleset → **New tag ruleset**: name `Release tags`, enforcement Active,
  bypass list **Repository admin**, target tags by pattern `*-v*`, and tick **Restrict creations**, **Restrict
  updates** and **Restrict deletions**.
