# Pull Request Template

## Description
<!-- Provide a clear and concise description of the changes introduced in this PR. Why was this change made, and what does it accomplish? -->

## Related Issues / Roadmap Items
<!-- Link to any related issues, user requests, or the item in docs/roadmap.md; once complete it moves to CHANGELOG.md -->

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Refactoring (pure code improvement, no behavior changes)
- [ ] Test additions / updates (no production code changes)
- [ ] Documentation / CI / Config updates (no production code changes)

## Key Implementation Details
<!-- List the main files, classes, or functions changed, and any design/architectural decisions made. -->

## Verification & Testing

### Automated Tests

- [ ] All unit tests passed (`pytest`)
- [ ] Code formatting verified (`ruff format --check .`)
- [ ] Linter checks passed (`ruff check .`)
- [ ] Type checking passed (`mypy src`)
- [ ] Security audit passed (`bandit -r src/ -ll`)

### Manual Verification
<!-- Describe the manual testing steps performed and the results. Include terminal outputs or metrics if applicable. -->
1. Steps to reproduce/verify:
2. Expected behavior:
3. Actual outcome:

## PR Checklist

- [ ] The roadmap item is complete: removed from `docs/roadmap.md` and condensed into `CHANGELOG.md` (or, for one PR of
  several, updated with its progress).
- [ ] The change is free of code duplication.
- [ ] All new and existing tests pass cleanly.
- [ ] Manual test procedures are documented in `TESTING.md` (if new procedures were introduced).
- [ ] Workflow documentation is synchronized (`python scripts/sync_workflow_docs.py`).
