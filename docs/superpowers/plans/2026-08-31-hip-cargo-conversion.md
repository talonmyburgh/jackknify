# hip-cargo 0.3.0 Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert jackknify in place to the project structure `hip-cargo init` produces at version 0.3.0, so that cab definitions, container fallback, release mechanics and CI all match the current hip-cargo contract.

**Architecture:** Seven sequential tasks on the `hip-cargo-conversion` branch. Packaging and the container-image single-source-of-truth land first, because cab generation depends on both. The CLI wrappers are then *derived* mechanically (generate-cabs → generate-function) rather than hand-written, gated by a round-trip test written first. `core/` is realigned to the new call convention, then CI, docs and the lockfile.

**Tech Stack:** Python 3.10+, uv (build backend `uv_build`, workspace tool), hip-cargo 0.3.0, typer, ruff, pytest, pre-commit, tbump, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-31-hip-cargo-conversion-design.md`

## Global Constraints

- Reference implementation: hip-cargo 0.3.0 at `~/software/hip-cargo`, templates under `~/software/hip-cargo/src/hip_cargo/templates/`.
- Top-level runtime dependency is `hip-cargo>=0.3.0` and nothing else. Every heavy dependency lives in `[project.optional-dependencies].full`.
- `requires-python = ">=3.10"`; ruff `line-length = 120`, `target-version = "py310"`.
- Build backend `uv_build`, constraint `requires = ["uv_build>=0.8.3,<0.12.0"]`.
- Container image: `ghcr.io/talonmyburgh/jackknify:latest`, declared **only** in `src/jackknify/_container_image.py`.
- Auto-changelog is OFF: no `cliff.toml`, no `CHANGELOG.md`, no `conventional-pre-commit` hook.
- Commands: `realise`, `noise`, `make_ms`, `onboard`. Every `cli/<name>.py` must pair with `cabs/<name>.yml`.
- `src/jackknify/cabs/*.yml` are generated artefacts. Never hand-edit them.
- Work on branch `hip-cargo-conversion`. Conventional Commit prefixes on every commit.
- After every code change: `uv run ruff format . && uv run ruff check . --fix`.

## Template substitution helper

Several tasks copy a hip-cargo template and substitute placeholders. Define this
helper once per shell session and reuse it:

```bash
HC=~/software/hip-cargo/src/hip_cargo/templates
hcsub() {
  sed -e 's|<PROJECT_NAME>|jackknify|g' \
      -e 's|<PACKAGE_NAME>|jackknify|g' \
      -e 's|<GITHUB_USER>|talonmyburgh|g' \
      -e 's|<GITHUB_URL>|https://github.com/talonmyburgh/jackknify|g' \
      -e 's|<CLI_COMMAND>|jackknify|g' \
      -e 's|<DEFAULT_BRANCH>|main|g' \
      -e 's|<LICENSE_TYPE>|MIT|g' \
      -e 's|<INITIAL_VERSION>|0.2.0|g' \
      -e 's|<AUTHOR_NAME>|Talon Myburgh|g' \
      -e 's|<AUTHOR_EMAIL>|myburgh.talon@gmail.com|g' \
      -e 's|<YEAR>|2024|g' \
      -e 's|<DESCRIPTION>|A Python-based package for jackknifing Measurement Set visibilities to create noise realisations|g' \
      "$1"
}
```

---

### Task 1: Packaging foundation

Split the dependency list so the lightweight install is real, delete the stale
setuptools shim, and add the missing tooling files.

**Files:**
- Modify: `pyproject.toml`
- Delete: `setup.py`
- Create: `.python-version`
- Replace: `.gitignore`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a `full` extra containing every heavy dependency; a project resolvable by `uv sync` with `hip-cargo>=0.3.0` available in `.venv`. Task 2 onwards run tooling via `uv run`.

- [ ] **Step 1: Rewrite the `[project]` block of `pyproject.toml`**

Replace the `license`, `dependencies` and (new) `[project.optional-dependencies]` sections. Note that the `License :: OSI Approved :: MIT License` classifier is **dropped**: under PEP 639 a `license` expression plus a license classifier is a rejected combination, and the hip-cargo template omits the classifier for exactly this reason.

```toml
[project]
name = "jackknify"
version = "0.2.0"
description = "A Python-based package for jackknifing Measurement Set visibilities to create noise realisations"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
license-files = ["LICENSE"]
authors = [
    { name = "Talon Myburgh", email = "myburgh.talon@gmail.com" }
]
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "hip-cargo>=0.3.0",
]

# Lightweight install (default):  pip install jackknify
#     Only pulls hip-cargo + typer. Suitable for cab consumers (e.g. stimela)
#     and for running commands via container fallback. Generated CLI wrappers
#     lazy-import core/, so when heavy deps are missing the wrapper drops into
#     run_in_container() automatically.
# Full install:                    pip install jackknify[full]
#     Adds the heavy runtime deps that core/ implementations need to execute
#     natively.
[project.optional-dependencies]
full = [
    "numpy",
    "jax[cpu]",
    "python-casacore",
    "astropy",
    "tqdm",
]
```

- [ ] **Step 2: Update the build-system constraint and ruff ignore list**

```toml
[build-system]
# NOTE: This uv_build constraint is intentionally aligned with hip-cargo's own pyproject.toml.
# If you bump uv_build in hip-cargo, update this template constraint to match.
requires = ["uv_build>=0.8.3,<0.12.0"]
build-backend = "uv_build"
```

and in `[tool.ruff.lint]` change `ignore = ["N999",]` to `ignore = []`. `N999` (invalid-module-name) was needed for the pre-conversion `Cli.py` module, which no longer exists.

Leave `[project.urls]`, `[project.scripts]`, `[tool.ruff]`, `[tool.pytest.ini_options]` and `[dependency-groups]` exactly as they are — they already match the template.

- [ ] **Step 3: Delete `setup.py`**

```bash
git rm setup.py
```

It is a leftover from before the `uv_build` migration and its metadata actively contradicts `pyproject.toml`: version `0.1.0`, `click` instead of `typer`, and an entry point `jackknify.Cli:cli` pointing at a module that no longer exists.

- [ ] **Step 4: Add `.python-version` and replace `.gitignore`**

```bash
hcsub $HC/python-version > .python-version
hcsub $HC/gitignore > .gitignore
```

- [ ] **Step 5: Verify the project resolves**

```bash
uv sync
```

Expected: succeeds, creating `.venv` with `hip-cargo` 0.3.0 and `typer`. It does **not** install jax/casacore — that is the point of the split.

If you also want the heavy dependencies for native execution, `uv sync --all-extras` is optional and may fail here: `python-casacore` needs the casacore C++ libraries present on the host. Nothing in this plan's test suite requires them, so a failure at `--all-extras` is not a blocker — fall back to plain `uv sync`.

- [ ] **Step 6: Confirm the package still imports and the CLI still runs**

```bash
uv run python -c "import jackknify; print(jackknify.__version__)"
uv run jackknify --help
```

Expected: prints `0.2.0`, then the Typer help listing `realise`, `noise`, `make-test-ms`, `onboard`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .python-version .gitignore
git add -u
git commit -m "build: split heavy deps into the full extra and drop setup.py"
```

---

### Task 2: Container image single source of truth

Introduce `_container_image.py` and repoint every cab-generation path at
`hip-cargo generate-cabs`, retiring the bespoke script and version sentinel.

**Files:**
- Create: `src/jackknify/_container_image.py`
- Delete: `scripts/generate_cabs.py` (and the `scripts/` directory)
- Modify: `.pre-commit-config.yaml`
- Rewrite: `tbump.toml`

**Interfaces:**
- Consumes: the resolvable project from Task 1.
- Produces: `CONTAINER_IMAGE = "ghcr.io/talonmyburgh/jackknify:latest"`, readable by `hip_cargo.utils.config.get_container_image("jackknify")`. Task 3 depends on this: without it, `generate-cabs` writes no `image:` field and `generate-function` emits no container-fallback body.

- [ ] **Step 1: Create `src/jackknify/_container_image.py`**

```python
CONTAINER_IMAGE = "ghcr.io/talonmyburgh/jackknify:latest"

# Optional GPU passthrough for the container-fallback path.
# Set GPU = True for a CUDA/GPU image, or "auto" to request a GPU only
# when one is detected (and, for docker/podman, the NVIDIA Container
# Toolkit is present). Absent => no GPU flags (the default).
# GPU = True

# Optional per-backend extra arguments, passed verbatim to the container
# runtime during fallback. Example: RUN_ARGS_APPTAINER = ["--ipc=host"].
# RUN_ARGS_DOCKER = []
# RUN_ARGS_PODMAN = []
# RUN_ARGS_APPTAINER = []
# RUN_ARGS_SINGULARITY = []
```

- [ ] **Step 2: Verify hip-cargo can resolve the image**

```bash
uv run python -c "
from hip_cargo.utils.config import get_container_image
print(get_container_image('jackknify'))
"
```

Expected: `ghcr.io/talonmyburgh/jackknify:latest`

If this prints `None`, the package is not installed into `.venv` — re-run `uv sync`. Every later step depends on this returning the image.

- [ ] **Step 3: Delete the bespoke generation script**

```bash
git rm scripts/generate_cabs.py
rmdir scripts 2>/dev/null || true
```

This script and its `.tbump_version` sentinel were the pre-0.2 mechanism for injecting a branch- or release-derived image tag. `_container_image.py` replaces both.

- [ ] **Step 4: Update the pre-commit hook**

In `.pre-commit-config.yaml`, add the install-hook-types line directly after the header comment block:

```yaml
# Wire both stages so a plain `pre-commit install` installs the commit-msg hook
# too (used by the conventional-pre-commit hook when --auto-changelog is on).
default_install_hook_types: [pre-commit, commit-msg]
```

and change the `generate-cabs` hook's entry from `python scripts/generate_cabs.py` to:

```yaml
      - id: generate-cabs
        name: Generate Stimela cab definitions
        entry: hip-cargo generate-cabs --module src/jackknify/cli/*.py --output-dir src/jackknify/cabs
        language: system
        always_run: true
        pass_filenames: false
```

`language: system` means the hook runs whatever `hip-cargo` is on `PATH`, so the project venv must be active when committing. This is the same contract hip-cargo uses on itself.

- [ ] **Step 5: Rewrite `tbump.toml`**

Generate it from the template, then strip the changelog hooks (auto-changelog is off per the spec):

```bash
hcsub $HC/tbump.toml > tbump.toml
python3 - <<'PY'
import re
from pathlib import Path
p = Path("tbump.toml")
t = p.read_text()
t = re.sub(
    r'\[\[before_commit\]\]\nname = "[^"]*[Cc]hangelog[^"]*"\ncmd = "[^"]*"\n\n?',
    "",
    t,
)
t = t.replace(
    "# Generate changelog, update container image tag, and regenerate cabs with the release version.",
    "# Update container image tag and regenerate cabs with the release version.",
)
p.write_text(t)
PY
```

- [ ] **Step 6: Verify the rewritten tbump config**

```bash
grep -c changelog tbump.toml
grep -n "before_push\|tbump_version" tbump.toml
uv run tbump --help >/dev/null && echo "tbump config parses"
```

Expected: `0` changelog references, no `before_push` or `.tbump_version` references, and the `current = "0.2.0"` version matching `pyproject.toml`. Fix `current` by hand if the substitution left `0.2.0` out of sync.

- [ ] **Step 7: Remove the stale sentinel file if present**

```bash
rm -f .tbump_version
```

- [ ] **Step 8: Commit**

```bash
git add src/jackknify/_container_image.py .pre-commit-config.yaml tbump.toml
git add -u
git commit -m "feat: adopt _container_image.py as the single source of truth for cab generation"
```

---

### Task 3: Round-trip test and canonical CLI wrappers

The compliance gate goes in first and fails; the wrappers are then regenerated
until it passes.

**Files:**
- Create: `tests/test_roundtrip.py`
- Rewrite: `src/jackknify/cli/realise.py`, `src/jackknify/cli/noise.py`, `src/jackknify/cli/onboard.py`
- Rename + rewrite: `src/jackknify/cli/make_test_ms.py` → `src/jackknify/cli/make_ms.py`
- Rename: `src/jackknify/core/make_test_ms.py` → `src/jackknify/core/make_ms.py`
- Rewrite: `src/jackknify/cli/__init__.py`
- Regenerate: `src/jackknify/cabs/*.yml`
- Delete: `src/jackknify/cabs/make_ms.yml` is regenerated in place; no stale cab files may remain

**Interfaces:**
- Consumes: `get_container_image("jackknify")` from Task 2.
- Produces: four CLI wrappers whose signatures each end with `backend: Literal[...] = "auto"` and `always_pull_images: bool = False`, and whose bodies call `run_in_container(<func>, dict(...), image=image, backend=backend, always_pull_images=always_pull_images)`. Core call convention becomes: first required parameter positional, all others keyword — e.g. `realise_core(ms_file, col=col, n_samples=n_samples, seed=seed, mode=mode, out_dir=out_dir)`. Task 4 matches `core/` to this.

- [ ] **Step 1: Write the failing round-trip test**

Create `tests/test_roundtrip.py` from the template with a case per command:

```bash
hcsub $HC/test_roundtrip.py > tests/test_roundtrip.py
cat >> tests/test_roundtrip.py <<'EOF'


def test_roundtrip_realise() -> None:
    """The realise command must round-trip cleanly through a cab."""
    _assert_roundtrip("realise")


def test_roundtrip_noise() -> None:
    """The noise command must round-trip cleanly through a cab."""
    _assert_roundtrip("noise")


def test_roundtrip_make_ms() -> None:
    """The make_ms command must round-trip cleanly through a cab."""
    _assert_roundtrip("make_ms")
EOF
```

The template already contains `_assert_roundtrip` and `test_roundtrip_onboard`. The helper runs `generate_cabs` on `src/jackknify/cli/<name>.py`, then `generate_function` on the resulting cab, then asserts the regenerated source is byte-identical line by line.

- [ ] **Step 2: Run the test and confirm it fails**

```bash
uv run pytest tests/test_roundtrip.py -v
```

Expected: all four FAIL. `test_roundtrip_make_ms` fails first with `Missing CLI module: src/jackknify/cli/make_ms.py` (the file is still named `make_test_ms.py`); the other three fail on line-count or line-content mismatches, because the committed sources use `typer.Argument`, hand-rolled `NewType` declarations without `parse_upath`, and have no `--backend` parameter or container-fallback body.

- [ ] **Step 3: Fix the `noise` input/output name collision before regenerating**

This must happen first, or the regeneration will bake in a silent behaviour change. `cli/noise.py` declares `out` as both an input option (default `noise_cube.fits`) and `@stimela_output(name="out")`. `hip_cargo.core.generate_cabs._command_spec_to_cab_def` drops any parameter whose hyphenated name matches an output key, so `out` is currently dropped from the cab's inputs and its default is lost. Round-tripping that cab yields `out: File | None = None`.

There is a second defect in the same file. The `out` default is written as
`Path("noise_cube.fits")` — a call expression. The introspector reads defaults
off the CST and emits them as literal source text, so that default reaches the
cab as `default: Path("noise_cube.fits")`, which is not a usable value. Verified
by reproducing it. The default must be a plain string.

Replace `src/jackknify/cli/noise.py` wholesale with the form below. It renames
the output to `out_cube` with `implicit="{out}"` — the same pattern `make_ms`
already uses for `ms_file`/`out_ms` — and fixes the default. This exact source
has been verified to generate a cab carrying both the `out` input (with
`default: noise_cube.fits`) and the `out_cube` output, and to regenerate back
byte-identically:

```python
from pathlib import Path
from typing import Annotated, NewType

import typer
from hip_cargo import parse_upath, stimela_cab, stimela_output

Directory = NewType("Directory", Path)
File = NewType("File", Path)


@stimela_cab(
    name="noise",
    info="Calculates a 'noise' cube (std dev) from a folder of FITS files.",
)
@stimela_output(
    dtype="File",
    name="out_cube",
    info="The resulting noise cube FITS file.",
    implicit="{out}",
)
def noise(
    folder_path: Annotated[
        Directory,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Folder containing input FITS files.",
        ),
    ],
    out: Annotated[
        File,
        typer.Option(
            parser=parse_upath,
            help="Output filename.",
        ),
    ] = "noise_cube.fits",
):
    """
    Calculates a 'noise' cube (std dev) from a folder of FITS files.
    """
    from jackknify.core.noise import noise as noise_core

    noise_core(folder_path, out=out)
```

This is an intermediate form: it has no `--backend` parameter yet, because that
is added by `generate-function` in Step 5 once the cab carries an `image:`
field. Step 5 will overwrite this file with the finished version.

Check the other three commands for the same call-expression default problem
before regenerating:

```bash
grep -n "= *Path(\|= *[A-Za-z_]*(" src/jackknify/cli/*.py
```

Expected: only `noise.py` was affected, and after this edit nothing matches.
Any other parameter defaulting to a call expression needs the same treatment.

- [ ] **Step 4: Rename the `make_ms` modules atomically**

The cab filename comes from `@stimela_cab(name=...)`, which is `make_ms`, and the cab's `command:` field is derived by substituting `core` for `cli` in the CLI module path. Both must move together or the cab points at a module that does not exist:

```bash
git mv src/jackknify/cli/make_test_ms.py src/jackknify/cli/make_ms.py
git mv src/jackknify/core/make_test_ms.py src/jackknify/core/make_ms.py
```

- [ ] **Step 5: Regenerate the cabs, then derive the canonical wrappers from them**

```bash
uv run hip-cargo generate-cabs --module 'src/jackknify/cli/*.py' --output-dir src/jackknify/cabs
for c in realise noise make_ms onboard; do
  uv run hip-cargo generate-function \
    --cab-file src/jackknify/cabs/$c.yml \
    --output-file src/jackknify/cli/$c.py \
    --config-file pyproject.toml
done
```

Confirm each cab carries the image line before proceeding:

```bash
grep -c "image: ghcr.io/talonmyburgh/jackknify:latest" src/jackknify/cabs/*.yml
```

Expected: `1` for each of the four files. A `0` means `_container_image.py` is not resolvable and Task 2 Step 2 must be revisited.

- [ ] **Step 6: Regenerate the cabs from the new sources and confirm the fixpoint**

```bash
uv run hip-cargo generate-cabs --module 'src/jackknify/cli/*.py' --output-dir src/jackknify/cabs
uv run pytest tests/test_roundtrip.py -v
```

Expected: all four PASS. If a test still fails, the mismatch it prints is authoritative — adjust `cli/<name>.py` to match the generated form, never the test.

- [ ] **Step 7: Rewrite `cli/__init__.py` in the template shape**

Subcommand imports move to the bottom of the file to avoid circular imports, and the command name changes from `make-test-ms` to `make-ms`:

```python
"""CLI for jackknify."""

import typer

app = typer.Typer(
    name="jackknify",
    help="Jackknife interferometric datasets using JAX.",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Jackknife interferometric datasets using JAX."""
    pass


# Register subcommands below. Imports go here (bottom) to avoid circular imports.
from jackknify.cli.make_ms import make_ms  # noqa: E402
from jackknify.cli.noise import noise  # noqa: E402
from jackknify.cli.onboard import onboard  # noqa: E402
from jackknify.cli.realise import realise  # noqa: E402

app.command(name="realise")(realise)
app.command(name="noise")(noise)
app.command(name="make-ms")(make_ms)
app.command(name="onboard")(onboard)

__all__ = ["app"]
```

- [ ] **Step 8: Lint, then verify the CLI surface**

```bash
uv run ruff format . && uv run ruff check . --fix
uv run jackknify --help
uv run jackknify realise --help
```

Expected: `jackknify --help` lists `realise`, `noise`, `make-ms`, `onboard`. `realise --help` shows `--ms-file` as an option (no longer positional), plus `--backend` and `--always-pull-images`.

- [ ] **Step 9: Re-run the full test suite and commit**

```bash
uv run pytest -v
git add -A src/jackknify tests/test_roundtrip.py
git commit -m "refactor!: regenerate CLI wrappers in canonical hip-cargo form

Required parameters become options, path types gain parse_upath, and every
command grows the --backend/--always-pull-images container fallback.
Renames make_test_ms to make_ms so the module pairs with its cab."
```

---

### Task 4: Align `core/` with the new call convention

**Files:**
- Modify: `src/jackknify/core/__init__.py`
- Modify: `src/jackknify/core/jackknife.py`
- Modify: `src/jackknify/core/realise.py`
- Modify: `src/jackknify/core/noise.py`
- Modify: `src/jackknify/core/make_ms.py`
- Rewrite: `src/jackknify/core/onboard.py`

**Interfaces:**
- Consumes: the call convention produced by Task 3 — path arguments arrive as `Path`/`UPath` objects, not `str`.
- Produces: core entrypoints accepting `str | os.PathLike` and coercing to `str` at the top of the function body, so `casacore.tables.table()` and `shutil` keep working unchanged.

- [ ] **Step 1: Move the jax configuration out of `core/__init__.py`**

Currently `core/__init__.py` runs `import jax` at module scope, which means importing *any* core module — including `core/onboard.py`, which only calls `print()` — requires jax. Under a lightweight install that raises `ImportError`, which the CLI wrapper catches, so `jackknify onboard` would silently dispatch into a container just to print text.

Replace `src/jackknify/core/__init__.py` with:

```python
"""Core implementations for jackknify."""
```

and add the configuration to the top of `src/jackknify/core/jackknife.py`, which is the module that actually uses jax:

```python
import jax
import jax.numpy as jnp

# Ensure JAX uses 64-bit precision globally
jax.config.update("jax_enable_x64", True)
```

- [ ] **Step 2: Widen the path parameters on the three core entrypoints**

`src/jackknify/core/realise.py` — change the signature and coerce at the top:

```python
def realise(
    ms_file: str | os.PathLike,
    col: str,
    n_samples: int,
    seed: int,
    mode: str,
    out_dir: str | os.PathLike | None = None,
):
    """Generates jackknife noise realisations from an MS."""
    ms_file = str(ms_file)
    out_dir = str(out_dir) if out_dir is not None else None
```

(`os` is already imported in this module. Drop the now-unused `from typing import Optional` import if ruff flags it.)

`src/jackknify/core/noise.py`:

```python
def noise(folder_path: str | os.PathLike, out: str | os.PathLike):
    """Calculates a 'noise' cube (std dev) from a folder of FITS files."""
    folder_path = str(folder_path)
    out = str(out)
```

Add `import os` at the top of this module — it does not currently import it.

`src/jackknify/core/make_ms.py`:

```python
def make_ms(ms_file: str | os.PathLike, rows: int, chans: int):
    """Creates a simple mock MS filled with 1s for testing."""
    ms_file = str(ms_file)
```

Add `import os` at the top of this module too.

Coercing at the entrypoint rather than threading `Path` through `ms_handler.py` and `calcnoise.py` is deliberate: `casacore.tables.table()` requires a `str`, and the helper modules are unchanged by this conversion.

- [ ] **Step 3: Refresh `core/onboard.py` from the current template**

The committed version predates two changes it needs to describe — the secret is now `APP_CLIENT_ID` holding the App's Client ID, and the image-tag workflow section did not exist:

```bash
hcsub $HC/onboard_core.py > src/jackknify/core/onboard.py
```

- [ ] **Step 4: Verify the onboard text and that nothing imports jax eagerly**

```bash
uv run ruff format . && uv run ruff check . --fix
uv run jackknify onboard | head -5
grep -n "APP_CLIENT_ID" src/jackknify/core/onboard.py
```

Expected: `onboard` prints the setup instructions **natively** — not via a container — proving the jax fix in Step 1 worked, and `APP_CLIENT_ID` appears in the refreshed text. On a lightweight install this is the concrete regression test for Step 1: before the fix it would have attempted a container run.

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -v
git add -A src/jackknify/core
git commit -m "fix: accept path objects in core entrypoints and stop importing jax package-wide"
```

---

### Task 5: Refresh CI/CD

**Files:**
- Replace: `.github/workflows/ci.yml`, `publish.yml`, `publish-container.yml`, `update-cabs.yml`
- Replace: `.github/dependabot.yml`
- Create: `.github/CODEOWNERS`

**Interfaces:**
- Consumes: `src/jackknify/_container_image.py` from Task 2 — `update-cabs.yml` rewrites its tag by regex.
- Produces: no Python interface. `update-cabs.yml` now commits `_container_image.py` alongside the cab YAML.

- [ ] **Step 1: Replace all four workflows and the dependabot config from the templates**

```bash
for w in ci publish publish-container update-cabs; do
  hcsub $HC/workflows/$w.yml > .github/workflows/$w.yml
done
hcsub $HC/dependabot.yml > .github/dependabot.yml
hcsub $HC/CODEOWNERS > .github/CODEOWNERS
```

This carries in, all at once: `actions/checkout@v6`, `setup-python@v6`, `setup-uv@v7`, `actions/cache@v5`, `docker/login-action@v4`, `metadata-action@v6`, `build-push-action@v7`, `create-github-app-token@v3`, `type=pep440` container tags, the `[skip checks]` guard jobs, `uv sync --all-extras --group test`, the image-tag reset step in `update-cabs.yml`, and the `uv` dependabot ecosystem replacing `pip`.

- [ ] **Step 2: Verify every workflow is valid YAML and the substitutions took**

```bash
uv run python -c "
import yaml, pathlib
for p in sorted(pathlib.Path('.github').rglob('*.yml')):
    yaml.safe_load(p.read_text())
    print('ok', p)
"
grep -rn "<[A-Z_]*>" .github/ && echo "UNSUBSTITUTED PLACEHOLDERS FOUND" || echo "no placeholders remain"
```

Expected: all five files parse, and no `<PLACEHOLDER>` survives.

- [ ] **Step 3: Confirm the secret rename is visible in the diff**

```bash
grep -n "client-id\|APP_CLIENT_ID\|APP_ID" .github/workflows/update-cabs.yml
```

Expected: `client-id: ${{ secrets.APP_CLIENT_ID }}` and **no** occurrence of the bare `APP_ID`. This workflow will fail on the first merge to `main` until the repository secret is renamed — that is Talon's action, tracked in spec §8.4, and is not a blocker for this branch.

- [ ] **Step 4: Commit**

```bash
git add .github
git commit -m "ci: refresh workflows and dependabot to the hip-cargo 0.3.0 templates"
```

---

### Task 6: Agent documentation and README

**Files:**
- Create: `CLAUDE.md`
- Create: `.claude/rules/architecture.md`, `.claude/rules/python-standards.md`, `.claude/rules/testing-and-ci.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing structural.
- Produces: documentation only.

- [ ] **Step 1: Write `CLAUDE.md` and the rules files from the templates**

```bash
mkdir -p .claude/rules
hcsub $HC/CLAUDE.md > CLAUDE.md
for r in architecture python-standards testing-and-ci; do
  hcsub $HC/claude/rules/$r.md > .claude/rules/$r.md
done
grep -rn "<[A-Z_]*>" CLAUDE.md .claude/ && echo "UNSUBSTITUTED PLACEHOLDERS FOUND" || echo "no placeholders remain"
```

- [ ] **Step 2: Add the installation-modes section to `README.md`**

Replace everything from the `Installation` heading down to (but not including) the `## Dependancies` heading with:

```markdown
Installation
============

``jackknify`` ships in two install modes.

**Lightweight** (default) pulls only ``hip-cargo`` and ``typer``:

    pip install jackknify

Every command still works in this mode — the CLI wrappers dispatch into the
project's container image (``ghcr.io/talonmyburgh/jackknify``) when the heavy
dependencies are not importable. This is the right mode for Stimela and for
machines that only need to launch commands.

**Full** adds the runtime dependencies needed to execute natively:

    pip install jackknify[full]

or from source:

    git clone https://github.com/talonmyburgh/jackknify
    cd jackknify
    uv sync --all-extras

Use ``--backend native`` to force in-process execution and surface an
``ImportError`` rather than falling back to a container, or ``--backend
docker`` / ``apptainer`` / ``podman`` / ``singularity`` to skip the native
attempt entirely.
```

- [ ] **Step 3: Record the CLI change in `README.md`**

Append this section immediately after the Installation section:

```markdown
Command-line interface
======================

Commands are defined once and exposed both as a CLI and as Stimela cabs
(generated into ``src/jackknify/cabs/``). Required parameters are passed as
options rather than positionally:

    jackknify realise --ms-file /path/to/observation.ms --n-samples 5
    jackknify noise --folder-path noise_images --out noise_cube.fits
    jackknify make-ms --ms-file mock.ms --rows 100 --chans 16

Path parameters accept remote URIs (``s3://``, ``gs://``, ``az://``) as well
as local paths.

**Note:** prior to the hip-cargo 0.3.0 conversion, ``ms-file`` and
``folder-path`` were positional arguments and ``make-ms`` was called
``make-test-ms``.
```

- [ ] **Step 4: Verify the README renders and the commands it documents exist**

```bash
uv run jackknify realise --help | grep -- "--ms-file"
uv run jackknify noise --help | grep -- "--folder-path"
uv run jackknify make-ms --help | grep -- "--rows"
```

Expected: each grep matches, confirming the README's examples are accurate.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md .claude README.md
git commit -m "docs: add hip-cargo agent rules and document install modes"
```

---

### Task 7: Lockfile and full verification sweep

**Files:**
- Create: `uv.lock`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: a committed lockfile. CI caches on `hashFiles('uv.lock')` and the `uv` dependabot ecosystem requires it.

- [ ] **Step 1: Generate the lockfile**

```bash
uv lock
```

- [ ] **Step 2: Install pre-commit hooks and run them over the whole tree**

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Expected: all hooks pass. The `generate-cabs` hook may rewrite cab YAML on its first run — if it does, that is a real finding: it means the committed cabs were stale. Inspect `git diff src/jackknify/cabs/`, stage the result, and re-run until clean.

- [ ] **Step 3: Confirm cab generation is idempotent**

```bash
uv run hip-cargo generate-cabs --module 'src/jackknify/cli/*.py' --output-dir src/jackknify/cabs
git diff --exit-code src/jackknify/cabs/ && echo "cabs are stable"
```

Expected: no diff. A diff here means the committed cabs do not match what the sources generate.

- [ ] **Step 4: Run the full verification sweep from the spec**

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -v
uv run jackknify --help
for c in realise noise make-ms onboard; do uv run jackknify $c --help >/dev/null && echo "ok: $c"; done
```

Expected: format check clean, lint clean, all tests pass (2 install tests + 4 round-trip tests), and every subcommand's help renders.

- [ ] **Step 5: Confirm no artefacts of the old structure survive**

```bash
for f in setup.py scripts/generate_cabs.py .tbump_version src/jackknify/cli/make_test_ms.py src/jackknify/core/make_test_ms.py; do
  test -e "$f" && echo "STILL PRESENT: $f" || echo "gone: $f"
done
grep -rn "tbump_version\|scripts/generate_cabs" . --exclude-dir=.git --exclude-dir=.venv --exclude-dir=docs && echo "STALE REFERENCES FOUND" || echo "no stale references"
```

Expected: all five gone, no stale references outside `docs/` (the spec and this plan legitimately mention them).

- [ ] **Step 6: Compare the result against a reference scaffold**

```bash
uv run hip-cargo init jackknify-reference \
  --github-user talonmyburgh \
  --description "reference scaffold" \
  --project-dir /tmp/jackknify-reference 2>/dev/null || echo "init needs network/git; skip and diff by hand"
diff -rq --exclude=.git /tmp/jackknify-reference/.github .github || true
```

This is a cross-check, not a gate: jackknify legitimately differs from a fresh scaffold (four commands instead of one, a real README, `recipes/`). Only unexplained differences in `.github/` and the tooling dotfiles are worth acting on. Delete `/tmp/jackknify-reference` afterwards.

- [ ] **Step 7: Commit and report**

```bash
git add uv.lock
git add -u
git commit -m "build: add uv.lock and finalise hip-cargo 0.3.0 conversion"
git log --oneline main..HEAD
```

Report to the user: the branch is ready for review; `APP_ID` → `APP_CLIENT_ID` is still outstanding on the repository (Talon); and the container-fallback path has been verified only in the shape of the generated code, not end to end.

---

## Deferred / not in this plan

- **Fixing the asymmetric input/output collision check in hip-cargo** (spec §8.2). Reported, not fixed. If it is later fixed by normalising both sides of the comparison, `realise` will begin losing its `out-dir` input and will need the same `implicit="{out_dir}"` treatment applied to `noise` in Task 3 Step 3.
- **Renaming the `APP_ID` repository secret** (spec §8.4) — repository settings, owner action.
- **End-to-end container fallback smoke test** (spec §8.5) — requires a pullable `ghcr.io/talonmyburgh/jackknify:latest`.
- **`[tool.hip-cargo].recipes_dir`** — `recipes/` at the repo root is already the default discovery location, so no declaration is needed.
