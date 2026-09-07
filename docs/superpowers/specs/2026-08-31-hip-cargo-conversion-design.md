# Design: Converting jackknify to the hip-cargo 0.3.0 project structure

**Date:** 2026-08-31
**Status:** Approved for planning
**Target:** hip-cargo 0.3.0 (`~/software/hip-cargo`, commit `c1a75b5`)

## 1. Background

jackknify was partially converted to the hip-cargo layout against an older
hip-cargo release. The skeleton landed correctly — `src/` layout,
`cli/` + `core/` + `cabs/` split, `@stimela_cab` / `@stimela_output`
decorators, and the four GitHub Actions workflows. Everything hip-cargo has
added since then is missing, and several pieces that did land now use
mechanisms hip-cargo has replaced.

This document specifies the conversion to full compliance with hip-cargo
0.3.0, as defined by `hip_cargo.core.init` and the templates under
`src/hip_cargo/templates/`.

## 2. Goals and non-goals

**Goals**

- The project matches what `hip-cargo init` produces today, in structure,
  packaging, tooling and CI.
- `tests/test_roundtrip.py` passes for every command: each `cli/<name>.py`
  regenerates byte-identically through its cab.
- The lightweight install (`pip install jackknify`) is genuinely usable —
  every command dispatches into the project container when heavy
  dependencies are absent.
- Cab YAML consumed by Stimela recipes stays semantically compatible.

**Non-goals**

- Changing what the science code computes.
- Adding monitoring, diagnostics or progress instrumentation.
- Using `hip-cargo transpile`.
- Fixing the hip-cargo bugs identified in §8 (they are reported, and worked
  around on the jackknify side).

## 3. Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Conversion method | In-place, incremental, on a branch | Preserves git history, the README and `recipes/`; each phase is independently verifiable. |
| Auto-changelog | Off | Matches jackknify's current setup; avoids imposing conventional-commit enforcement on existing contributors. `tbump.toml` gets the changelog hooks stripped, no `cliff.toml`, no `CHANGELOG.md`. |
| Positional CLI arguments | Accept the break | hip-cargo's canonical form emits `typer.Option(...)` for every required parameter; `typer.Argument` does not survive the round trip. `jackknify realise foo.ms` becomes `jackknify realise --ms-file foo.ms`. |

## 4. What does not change

Verified empirically by running `generate-cabs` on the current sources: the
regenerated `realise.yml` differs from the committed one only by the `image:`
line and a redundant `required: false`. Cab input names, dtypes, defaults and
`policies: {positional: true}` are all preserved, because `positional` is
inferred from `required`, not from the use of `typer.Argument`.

Consequence: `recipes/make-noise-map.yml` and any external Stimela recipe
consuming these cabs keep working unchanged (with the single exception in
§8.1). The breaking change is confined to the Python CLI surface.

## 5. Design

### 5.1 Packaging

`pyproject.toml`:

- Top-level `dependencies` reduces to `hip-cargo>=0.3.0`.
- `numpy`, `jax[cpu]`, `python-casacore`, `astropy`, `tqdm` move to
  `[project.optional-dependencies].full`. The Dockerfile already installs
  `.[full]`; that extra does not currently exist, so the container is being
  built without the heavy dependencies declared.
- `license = { text = "MIT" }` becomes `license = "MIT"` plus
  `license-files = ["LICENSE"]`.
- `uv_build` bound widens to `<0.12.0` to match hip-cargo's own constraint.
- Remove the duplicate bare `hip-cargo` entry and the now-unneeded `N999`
  ruff ignore.

`setup.py` is deleted. The build backend is `uv_build`; the file is a
leftover whose metadata actively contradicts `pyproject.toml` (version
0.1.0, `click` instead of `typer`, entry point `jackknify.Cli:cli`).

Added: `.python-version` (3.11), `uv.lock` (CI caches on it and the new
dependabot `uv` ecosystem requires it), `.github/CODEOWNERS`. `.gitignore`
is replaced with the current template.

### 5.2 Version and release mechanics

`tbump.toml` is rewritten to the current template shape. The old
`[[before_push]]` hooks and the `.tbump_version` sentinel file are removed
entirely. The new `[[before_commit]]` hooks:

1. Rewrite the tag in `src/jackknify/_container_image.py` to the release
   version.
2. Run `hip-cargo generate-cabs --module 'src/jackknify/cli/*.py'
   --output-dir src/jackknify/cabs`.
3. Stage the cabs and `_container_image.py`.
4. `uv lock` and stage `uv.lock`.

Changelog hooks are stripped (§3). `message_template` becomes
`chore(release): bump version to {new_version}`.

### 5.3 Container image as single source of truth

New file `src/jackknify/_container_image.py`:

```python
CONTAINER_IMAGE = "ghcr.io/talonmyburgh/jackknify:latest"
```

plus the commented-out `GPU` and `RUN_ARGS_*` stanzas from the template.

`scripts/generate_cabs.py` is deleted. Cab generation now calls
`hip-cargo generate-cabs` directly from pre-commit, `tbump`, and the
`update-cabs` workflow.

This file is load-bearing and its absence is the root cause of the largest
gap. `hip_cargo.core.generate_cabs` resolves the image by importing
`<package>._container_image` from the **installed** package
(`hip_cargo.utils.config.get_container_image`). With no such module, no
`image:` field is written to the cab; with no `image:` field,
`generate-function` emits no `--backend` parameter and no container-fallback
body at all. Verified by reproducing both states.

A corollary: cab generation requires jackknify to be installed in the active
environment. `uv sync` covers this, and the pre-commit hook runs with
`language: system`, so the project venv must be active — the same contract
hip-cargo uses on itself.

### 5.4 Pre-commit

The `generate-cabs` local hook's entry changes from
`python scripts/generate_cabs.py` to
`hip-cargo generate-cabs --module src/jackknify/cli/*.py --output-dir src/jackknify/cabs`.
`default_install_hook_types: [pre-commit, commit-msg]` is added at the top of
the file, matching the template.

### 5.5 CLI wrappers

The canonical form is derived mechanically rather than hand-written, and
iterated to a fixpoint:

1. Add `_container_image.py`; `uv sync --all-extras`.
2. Run `generate-cabs` on the current `cli/*.py`. Cabs now carry `image:`.
3. Run `generate-function` on each cab to produce the canonical
   `cli/<name>.py`, replacing the originals.
4. Regenerate cabs from the new sources and assert both cab YAML and CLI
   source are stable under a second pass.

Each wrapper gains:

- `backend: Literal["auto", "native", "apptainer", "singularity", "docker",
  "podman"] = "auto"` and `always_pull_images: bool = False`, both annotated
  `StimelaMeta(skip=True)` so they appear in the Python CLI but not in cab
  YAML.
- A `try/except ImportError` native attempt followed by
  `run_in_container(...)`, with `get_container_image("jackknify")` resolving
  the image.
- A `preflight_remote_must_exist(...)` call before native dispatch.

Hand-rolled `NewType("MS", Path)` / `NewType("Directory", Path)` declarations
are replaced by the generated equivalents carrying `parser=parse_upath`,
which is what enables `s3://`, `gs://` and `az://` inputs.

**Renames.** `tests/test_roundtrip.py` requires `cli/<name>.py` to pair with
`cabs/<name>.yml`. `cli/make_test_ms.py` declares `@stimela_cab(name="make_ms")`
and therefore produces `cabs/make_ms.yml`. So:

- `cli/make_test_ms.py` → `cli/make_ms.py`
- `core/make_test_ms.py` → `core/make_ms.py` (the cab's `command:` field is
  derived by substituting `core` for `cli` in the module path, so the two
  must stay in lockstep)
- The registered command name becomes `make-ms` rather than `make-test-ms`.

`cli/__init__.py` adopts the `cli_multi.py` template shape, with subcommand
imports at the bottom of the file to avoid circular imports.

### 5.6 core/ adjustments

- Core functions currently annotate paths as `str`. The generated wrappers
  pass `Path`/`UPath` objects through. Signatures widen accordingly, with
  `str()` applied only at the boundaries that require it — `casacore.tables.table()`
  in `ms_handler.py`, and the `os.path` usage in `calcnoise.py`.
- `core/__init__.py` currently runs `import jax` and
  `jax.config.update("jax_enable_x64", True)` at module scope. That makes
  importing *any* core module — including `core/onboard.py`, which only calls
  `print()` — depend on jax. Under a lightweight install this raises
  `ImportError`, which the wrapper catches, so `jackknify onboard` would
  silently dispatch into a container to print text. The jax configuration
  moves to `core/jackknife.py`, where jax is actually used, and
  `core/__init__.py` becomes a plain docstring module.

### 5.7 Tests

`tests/test_roundtrip.py` is added from the template with a case per command:
`realise`, `noise`, `make_ms`, `onboard`. This is the compliance gate — it
runs `generate-cabs` then `generate-function` and asserts the regenerated
source is byte-identical to the committed `cli/<name>.py`.

`tests/test_install.py` is unchanged.

### 5.8 CI/CD and agent documentation

The four workflows are refreshed from the current templates:

| Change | Files |
|---|---|
| `actions/checkout@v4` → `@v6`, `setup-python@v5` → `@v6`, `setup-uv@v5` → `@v7`, `actions/cache@v4` → `@v5` | all four |
| `docker/login-action@v3` → `@v4`, `metadata-action@v5` → `@v6`, `build-push-action@v6` → `@v7` | `publish-container.yml` |
| `type=semver` → `type=pep440` (so `v0.2.0rc1` tags work) | `publish-container.yml` |
| `[skip ci]` → `[skip checks]`, plus the skip-checks guard step on both jobs | `ci.yml`, `update-cabs.yml` |
| `create-github-app-token@v1` → `@v3`, `app-id: secrets.APP_ID` → `client-id: secrets.APP_CLIENT_ID` | `update-cabs.yml` |
| Image-tag reset step before cab regeneration; commit `_container_image.py` alongside the cabs | `update-cabs.yml` |
| `uv sync --group test` → `uv sync --all-extras --group test`; drop the now-redundant CLI smoke test | `ci.yml` |
| Drop the `test` job (tests already run on every push/PR); publish gates on `quality` only | `publish.yml` |
| Drop `master` from branch triggers | `publish-container.yml` |

`.github/dependabot.yml` moves from the `pip` ecosystem to `uv`, and adopts
the template's `python-minor-patch` / `python-major` grouping. The
`reviewers:` key is dropped in favour of `.github/CODEOWNERS`.

Agent documentation is added: `CLAUDE.md` and
`.claude/rules/{architecture,python-standards,testing-and-ci}.md`, from the
templates with placeholders substituted.

`README.md` gains a lightweight-vs-full installation section and a note
recording the CLI change from positional arguments to options.

## 6. Sequencing

Seven phases, each independently verifiable:

1. Packaging: `pyproject.toml`, delete `setup.py`, `tbump.toml`,
   `.pre-commit-config.yaml`, `.gitignore`, `.python-version`.
2. `_container_image.py`; delete `scripts/generate_cabs.py` and
   `.tbump_version` handling.
3. Regenerate `cli/*.py` via the generate-cabs → generate-function fixpoint;
   rewrite `cli/__init__.py`. The `make_test_ms` → `make_ms` rename lands in
   this phase and must be applied atomically across `cli/`, `core/` and
   `cabs/` — the cab's `command:` field is derived from the CLI module path,
   so a half-applied rename leaves the cab pointing at a module that does not
   exist.
4. `core/` signature widening and the `core/__init__.py` jax fix.
5. `tests/test_roundtrip.py`.
6. Workflows, dependabot, `CODEOWNERS`, `CLAUDE.md`, `.claude/rules/`,
   README.
7. `uv lock`; full verification sweep.

## 7. Verification

Per phase, and again at the end:

- `uv sync --all-extras`
- `uv run ruff format --check . && uv run ruff check .`
- `uv run pytest -v` — round-trip tests must pass
- `uv run pre-commit run --all-files` — clean
- `git diff --exit-code src/jackknify/cabs/` after a fresh `generate-cabs`
- `uv run jackknify --help`, and `--help` for each of `realise`, `noise`,
  `make-ms`, `onboard`

## 8. Known obstacles

### 8.1 `noise`: input and output share the name `out`

`cli/noise.py` declares `out` both as an input option with default
`Path("noise_cube.fits")` and as an `@stimela_output(name="out")`.

`_command_spec_to_cab_def` drops any parameter whose hyphenated name matches
an output key:

```python
if param_spec.name.replace("_", "-") in outputs:
    continue
```

For `out` this matches, so the input — and its `noise_cube.fits` default — is
dropped from the cab. The committed `noise.yml` confirms this: it has no
`out` input. Round-tripping therefore regenerates `out` as
`File | None = None`, carrying the *output's* info text. Left alone, the
mechanical regeneration would silently change the default output filename
from `noise_cube.fits` to `None`.

**Resolution:** adopt the pattern `make_ms` already uses for the same
situation — rename the output and link it back to the input implicitly:

```python
@stimela_output(
    name="out_cube",
    dtype="File",
    info="The resulting noise cube FITS file.",
    implicit="{out}",
)
```

`out` then survives as a normal input with its default intact. Verified
end-to-end: this shape generates a cab carrying both the `out` input (with
`default: noise_cube.fits`) and the `out_cube` output, and regenerates back
to byte-identical source.

`recipes/make-noise-map.yml` passes `out: =recipe.output_img` to the cab,
which still resolves — as an input rather than an output. Nothing in the
repo references the cab's output by name, so this rename is contained.

### 8.2 The same collision check is asymmetric (hip-cargo bug)

`realise` declares an input `out_dir` and an `@stimela_output(name="out_dir")`
— the identical situation — but is *not* affected, because the check
hyphenates the parameter name (`out-dir`) before testing membership in the
output keys (`out_dir`). Names containing an underscore therefore escape the
check and names without one do not.

This is a hip-cargo bug and is reported here rather than fixed. Note that if
it is fixed by normalising both sides, `realise`'s `out-dir` input would then
be dropped too, and `realise` would need the same `implicit="{out_dir}"`
treatment as §8.1.

### 8.3 `typer.Argument` does not round-trip

`generate-function` emits `typer.Option(...)` for every required parameter;
`policies: {positional: true}` in the cab is inferred from `required` on the
way out but is not translated back into `typer.Argument` on the way in.
Accepted as a breaking CLI change per §3.

### 8.4 GitHub secret rename (requires repository owner)

`update-cabs.yml` moves to `actions/create-github-app-token@v3`, which takes
`client-id` rather than `app-id`. The repository secret `APP_ID` must be
replaced with `APP_CLIENT_ID`, holding the GitHub App's **Client ID** (not
its numeric App ID). This is a repository settings change that cannot be made
from the working tree; the `update-cabs` workflow will fail on merge to
`main` until it is done.

### 8.5 Container fallback cannot be verified locally

Whether `ghcr.io/talonmyburgh/jackknify:latest` exists and is pullable in the
development environment is unknown. Verification covers the *shape* of the
generated fallback code; an end-to-end `--backend docker` smoke test is left
to the repository owner.
