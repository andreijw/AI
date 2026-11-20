---
name: release
description: >
  Takes a distribution of the RL framework (rl-core, rl-navigation, rl-challenges) from "ready to release" to a
  published GitHub Release: the version bump and CHANGELOG section in a pull request, the
  local `scripts/release.py check`, the signed tags in dependency order once the maintainer says yes to each,
  watching the Release workflow, checking the draft, publishing it once the maintainer says yes, and the next
  distribution's tag once a dependency is published. Use this whenever
  the user asks to release, tag, publish or cut a version of a distribution, asks what the next release step is,
  or a Release workflow run has finished or failed.
---

# Release

The procedure and its reasons are in [docs/releasing.md](../../../docs/releasing.md). This skill is the order you
run it in and what you check on the way. A release is a GitHub Release built from a pushed tag; nothing goes to
PyPI.

## What is the maintainer's, always

- **The yes to tag.** A tag is signed with their key (`git tag -s`, the signing key this machine's git is set up
  with) and the `Release tags` ruleset lets only admins create one; you push over HTTPS with their `gh` login. You
  ask for each tag by name, with what `check` said, and create and push only the tags they say yes to (the
  maintainer's decision, 2026-11-20). The signature then shows that their machine made the tag, not that they typed
  it: their yes is the record of that. Without a yes, or if they prefer to run it, print the commands instead.
- **The yes to publish.** Immutable releases are on: a published release's tag and files can never be changed.
  You say the draft is ready and why, and ask for each tag by name; you publish only the tags they say yes to
  (the maintainer's decision, 2026-11-12). Without a yes, nothing is published.
- **Deleting or moving a pushed tag** after a failed run (re-tagging after their delete is step 4 again, with a
  fresh yes), and the version number itself (SemVer 0.x: in 0.x a minor
  version may break the API; a breaking `rl-core` change bumps the dependents too, `docs/releasing.md`).

## Steps

### 1. State of play

```bash
gh release list --limit 10
git ls-remote --tags https://github.com/andreijw/AI.git '*-v*'
grep -n "__version__ = " packages/*/src/*/__init__.py
grep -n "^## " packages/*/CHANGELOG.md
```

For each distribution: its in-tree version, whether a tag and a release (draft or published) exist for it, and
whether its open section (`## [Unreleased]`, or the undated `## [<version>]` of a bumped but unreleased version)
holds anything. Dependency order is rl-core first; rl-navigation and
rl-challenges after rl-core's release is **published**, since their runs download its published wheel and the
workflow's read-only token cannot see a draft.

### 2. Version bump (skip if the in-tree version is not yet released)

Through the dev loop as a small PR on its own branch (`chore/release-<dist>-<version>`): bump `__version__` where it
is defined, turn `## [Unreleased]` into `## [<version>] - <today>` with an empty `## [Unreleased]` above it, and for
a breaking `rl-core` bump also the dependents' `rl-core~=0.x.0` requirements and minor versions. If the version
is already bumped with an undated `## [<version>]` section (a breaking change bumps it in its first PR), only date
the heading and add the empty `## [Unreleased]`: `release.py check` refuses an undated section. Run
`python -m pytest tests/test_packages.py tests/test_versions.py`. The maintainer commits; the tag waits for the
merge.

### 3. Local check

On an up-to-date `develop` (fetch over HTTPS, `git fetch https://github.com/andreijw/AI.git develop`):

```bash
python scripts/release.py check <dist>-v<version> --notes <scratch>/notes.md
```

It needs the tags of what the distribution requires (`python scripts/release.py dependency-tags <dist>`); fetch them
first. A refusal here is a refusal the workflow would give; fix it before asking for a tag.

### 4. Ask for the tag, then tag

Ask for each tag by name (AskUserQuestion), one distribution or several whose dependencies are all published. On a
yes, from the main checkout (SSH is refused from this machine's shell, so fetch and push over HTTPS). `<commit>` is
the `develop` commit step 3's `check` ran on; if `develop` has moved since, rerun step 3 first:

```bash
T=<dist>-v<version>
git fetch https://github.com/andreijw/AI.git develop
[ "$(git rev-parse FETCH_HEAD)" = "<commit>" ] || { echo "develop moved: rerun step 3"; exit 1; }
git tag -s "$T" FETCH_HEAD -m "<dist> <version>" || exit 1
git cat-file -p "$T" | grep -q 'BEGIN SSH SIGNATURE' || { git tag -d "$T"; echo "not signed"; exit 1; }
git -c credential.helper='!gh auth git-credential' push https://github.com/andreijw/AI.git "refs/tags/$T"
gh api "repos/andreijw/AI/git/tags/$(git rev-parse "$T")" -q .verification.verified   # must print true
```

`git tag -v` needs an allowed-signers file this machine does not have, so GitHub checks the signature against the
maintainer's signing key, in the last line. If `git tag -s` cannot sign (a passphrase, no key), nothing is pushed:
print the commands for the maintainer, as `docs/releasing.md` gives them. If the last line prints `false`, stop:
the tag is public, so tell the maintainer before its Release run's draft goes further; deleting it is theirs.

### 5. Watch the run

```bash
gh run list --workflow release.yml --limit 3
gh run watch <run id> --exit-status
```

A red run leaves nothing public. Read the failing job's log (`gh run view <id> --log-failed`), fix the cause in a
PR, and once it merges ask the maintainer to delete the tag (`git push origin :refs/tags/<tag>`, theirs), then step 4
with a fresh yes. A run on a tag whose draft exists refreshes the draft.

### 6. Check the draft

```bash
gh release view <tag> --json isDraft,isPrerelease,name,body,assets
d=$(mktemp -d <scratch>/release.XXXX) && gh release download <tag> -D "$d" && (cd "$d" && sha256sum -c SHA256SUMS)
```

It is ready when: it is a draft; the prerelease flag matches the version (`rc`, `.dev` are prereleases); the assets
are the wheel, the sdist and `SHA256SUMS`, and every sum is OK; the notes equal the CHANGELOG section of that
version. Say so to the maintainer with what you checked, and that publishing cannot be undone, and ask whether to
publish it (AskUserQuestion, naming the tag). On a yes:

```bash
gh release edit <tag> --draft=false
gh release view <tag> --json isDraft,publishedAt
```

The rule `Bash(gh release edit * --draft=false)` in `.claude/settings.local.json` allows that one command.

### 7. Next

Once it is published, confirm it (`gh release view <tag> --json isDraft`) and go back to step 4 for the
distributions it unblocked. Update `~/rl-experiments/STATUS.md` at each hand-over.

## Never

- `git tag` or a tag push without the maintainer's yes for that tag; moving, deleting or force-pushing a tag; `gh
  release create` or `gh release delete`.
- `gh release edit --draft=false` without the maintainer's yes for that tag, or any other `gh release edit`.
- Editing `.github/workflows/release.yml` without the maintainer's go (anything under `.github/` is theirs).
- `pip install -U` of an `rl-*` name: these names are not on PyPI, so it asks an index anyone could publish to.
