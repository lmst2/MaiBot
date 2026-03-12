# GitHub-Side Localization Workflow Report

## Scope

This document summarizes the current GitHub-side localization workflow in this repository.
It focuses on how GitHub Actions, branch conventions, pull requests, and validation gates work together around Crowdin.

This is intentionally a repository-specific operational report, not a generic Crowdin or GitHub Actions tutorial.

## Current Branch Model

- Source branches covered by the localization workflow:
  - `main`
  - `r-dev`
- Crowdin return branches:
  - `l10n_main`
  - `l10n_r-dev`
- Merge strategy:
  - translations do not go directly into `main` or `r-dev`
  - Crowdin updates return through pull requests and can be reviewed before merge

## Source of Truth

- Source locale for JSON translations: `locales/zh-CN/*.json`
- Source locale for prompt templates: `prompts/zh-CN/**/*.prompt`
- Current prompt template extension in the repository: `.prompt` only

GitHub Actions currently treat `zh-CN` assets as the source-language side of the workflow.
Translated target assets are not used to trigger source uploads back to Crowdin.

## Workflows Involved

### 1. `crowdin-sync.yml`

Role:
- uploads source-language assets to Crowdin
- downloads currently available translations from Crowdin when the workflow runs
- creates or updates localization pull requests back to the matching base branch

Current triggers:
- manual dispatch
- scheduled sync every 6 hours: `17 */6 * * *` UTC
- push to `main` or `r-dev` when one of these paths changes:
  - `crowdin.yml`
  - `locales/zh-CN/*.json`
  - `prompts/zh-CN/**/*.prompt`

Current branch behavior:
- push-triggered runs sync the current Git branch and use a matching localization branch name:
  - `main -> l10n_main -> PR into main`
  - `r-dev -> l10n_r-dev -> PR into r-dev`
- scheduled runs explicitly cover both `main` and `r-dev`

Current permissions and credentials:
- `contents: write`
- `pull-requests: write`
- `GITHUB_TOKEN`
- `CROWDIN_PROJECT_ID`
- `CROWDIN_PERSONAL_TOKEN`

Important boundary:
- the workflow keeps the PR-based return flow intact
- it does not directly push translated content into `main` or `r-dev`

### 2. `i18n-validate.yml`

Role:
- runs repository-side localization validation
- blocks structurally invalid or policy-breaking localization changes

Current triggers:
- pull requests that touch:
  - `locales/**/*.json`
  - `prompts/**/*.prompt`
  - `scripts/i18n_validate.py`
  - `src/common/i18n/**/*.py`
  - `src/common/prompt_i18n.py`
  - `src/prompt/prompt_manager.py`
- pushes to `main` or `r-dev` for the same path set

Current validation scope:
- JSON locale key alignment against `zh-CN`
- placeholder consistency
- plural structure consistency
- prompt placeholder consistency
- English locale protection against Chinese source-language leakage
- rejection of non-`zh-CN` entries that directly preserve Chinese source text

Prompt behavior note:
- missing target prompt files currently produce warnings, not hard failures
- runtime still falls back to `zh-CN` prompt templates when localized prompt files are absent

### 3. `precheck.yml`

Role:
- checks whether a pull request conflicts with its real target branch
- preserves the existing conflict-label behavior

Current triggers:
- all pull requests

Current behavior:
- checks out the PR head commit
- fetches the actual PR base branch from `github.event.pull_request.base.ref`
- performs a merge simulation against that real base branch
- marks the PR as conflicted only if the merge simulation produces unmerged files

This means:
- feature branches into `main` are checked against `main`
- feature branches into `r-dev` are checked against `r-dev`
- `l10n_main` PRs are checked against `main`
- `l10n_r-dev` PRs are checked against `r-dev`

### 4. `ruff-pr.yml`

Role:
- runs Ruff lint and format checks for pull requests that are relevant to Python code quality

Current triggers:
- pull requests that touch:
  - `*.py`
  - `**/*.py`
  - `pyproject.toml`
  - `ruff.toml`
  - `.ruff.toml`
  - `setup.cfg`
  - `tox.ini`
  - `.pre-commit-config.yaml`

Effect:
- translation-only localization pull requests do not run Ruff by default
- Python or Ruff-related pull requests still run the existing Ruff checks

## End-to-End GitHub Flow

### A. Source-language update on `r-dev` or `main`

1. A source-language change is pushed to `main` or `r-dev`.
2. `crowdin-sync.yml` uploads source assets to Crowdin.
3. The same workflow may also download any translations currently available in Crowdin when that workflow run executes.
4. A localization PR is opened or updated:
   - `l10n_main -> main`
   - `l10n_r-dev -> r-dev`

### B. Localization PR created by Crowdin branch

1. A localization PR is opened from `l10n_${branch}` into its matching base branch.
2. `precheck.yml` validates conflicts against the real PR base branch.
3. `i18n-validate.yml` validates localization structure and repository-specific locale-content policy.
4. `ruff-pr.yml` does not run if the PR only changes translation assets.
5. Maintainers review and merge the localization PR in the normal PR flow.

### C. Scheduled sync

1. Every 6 hours, GitHub Actions runs a scheduled localization sync.
2. The workflow explicitly processes both `main` and `r-dev`.
3. If Crowdin currently has downloadable translation updates, GitHub updates or creates the corresponding `l10n_` PRs.

## How the Current Setup Avoids Sync Loops

- Crowdin source uploads are triggered only from source-language assets:
  - `locales/zh-CN/*.json`
  - `prompts/zh-CN/**/*.prompt`
- translated target files do not trigger another source upload cycle
- translations return through `l10n_` branches and PRs instead of direct pushes to base branches
- translation-only PRs do not trigger Ruff, which reduces unnecessary CI noise without weakening Python quality gates

## Current Locale Content Policy on GitHub

This section describes repository-specific validation policy layered on top of the baseline Crowdin PR workflow.
It is not a default Crowdin rule.

The repository now enforces a stricter GitHub-side policy for committed target locale files:

- `zh-CN` remains the source language
- non-`zh-CN` locale files must not carry over Chinese source text unchanged
- English locale files must not retain Chinese characters

This policy is enforced by `scripts/i18n_validate.py` and therefore applies to localization pull requests on GitHub before merge.

## What Was Verified Against Repository Reality

- The repository currently uses `.prompt` as the only i18n-relevant prompt template extension.
- `i18n-validate.yml` already watches `prompts/**/*.prompt`, so no broader prompt-file trigger was needed.
- The only committed non-`zh-CN` locale directory currently present in the repository is `locales/en-US`.
- Chinese text previously found in `locales/en-US/startup.json` has been removed.

## Practical Maintainer Expectations

- A translation-only PR should normally trigger:
  - precheck
  - i18n validation
- A translation-only PR should normally not trigger:
  - Ruff PR checks
- A Python code PR should normally trigger:
  - precheck
  - Ruff PR checks
  - i18n validation if it touches i18n-related code or locale/prompt assets

## Open Discussion Topics

These are not current defects, but they may still be useful discussion topics for the team:

- whether scheduled source uploads should remain enabled together with scheduled translation downloads
- whether GitHub-side workflow linting should be added explicitly in the future
- whether additional locale-specific content rules should be introduced once more target locales are committed in the repository

## Bottom Line

The current GitHub-side localization workflow is now centered on a stable model:

- `zh-CN` is the source language
- `main` and `r-dev` are the source branches covered by Crowdin sync
- Crowdin returns through `l10n_${branch}` pull requests
- PR conflict checks now use the real base branch
- translation-only PRs no longer run Ruff by default
- GitHub-side i18n validation now also protects against source-language leakage in committed target locales

This keeps the existing Crowdin branch strategy intact while making pull request validation more accurate and less noisy.
