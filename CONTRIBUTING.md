# Contributing Guidelines

Thank you for your interest in contributing to the ICEYE QGIS plugins! We appreciate your support.

These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Setting Up for Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/iceye-ltd/iceye-qgis-plugins.git
   cd iceye-qgis-plugins
   ```

2. **Prerequisites**: QGIS 3.44 or later

3. Install from source (development-friendly): The install scripts create a link from your local QGIS plugins directory to `ICEYE_toolbox` in this repository so you can modify the code directly.

   **Linux**
   ```bash
   ./install.sh linux
   ```

   **macOS**
   ```bash
   ./install.sh macos
   ```

   **Windows (PowerShell)**
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\install.ps1
   ```
   If symbolic links are restricted on your machine, use:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\install.ps1 -UseJunction
   ```

4. Restart QGIS and enable the plugin via the Plugin Manager. For fast development iteration, we recommend the [Plugin Reloader](https://plugins.qgis.org/plugins/plugin_reloader/) plugin to reload your plugin without restarting QGIS.

## Git Hooks

We use git hooks through [pre-commit](https://pre-commit.com/) to enforce and automatically check some rules (code style and [Conventional Commits](https://www.conventionalcommits.org/) on `git commit`). Set this up once per clone before you push.

### Installing the `pre-commit` command

You need the `pre-commit` CLI available on your `PATH`. Use either:

**Option A — with the project’s dev dependencies (recommended)**
From the repository root, in a [virtual environment](https://docs.python.org/3/library/venv.html):

```bash
python -m venv .venv
source .venv/bin/activate
# Windows (cmd): .venv\Scripts\activate.bat
# Windows (PowerShell): .venv\Scripts\Activate.ps1

pip install -e ".[dev]"
pip install qgis-stubs
```

The `pip install -e ".[dev]"` command installs `pre-commit`, `ruff`, and related dev packages declared in [`pyproject.toml`](pyproject.toml). `qgis-stubs` is still installed separately so ruff understands QGIS imports (see [Testing](#testing)).

**Option B — standalone tool**
If you only want the hook runner globally: `pip install pre-commit` or `pipx install pre-commit` (see [pre-commit installation](https://pre-commit.com/#install)).

### Connecting pre-commit to Git

Register the hooks so Git runs them when you commit:

```bash
pre-commit install
pre-commit install-hooks
```

- **`pre-commit install`** creates scripts under `.git/hooks/` (for example `pre-commit` and `commit-msg`) that call the `pre-commit` framework. Those are the Git hooks; you normally do not edit them by hand.
- **`pre-commit install-hooks`** downloads and builds the environments for each hook in [.pre-commit-config.yaml](.pre-commit-config.yaml) (ruff, conventional-pre-commit, etc.) so the first real commit is not stuck downloading.

The repo sets `default_install_hook_types` in [.pre-commit-config.yaml](.pre-commit-config.yaml), so `pre-commit install` should register both the **pre-commit** stage (format/lint before the commit is created) and the **commit-msg** stage (validate the commit message). If your `pre-commit` version ignores that, run:

```bash
pre-commit install --hook-type pre-commit --hook-type commit-msg
pre-commit install-hooks
```

### Checking that it works

After installing, you should have `.git/hooks/pre-commit` and `.git/hooks/commit-msg`. Optionally run `pre-commit run --all-files` to run every hook on the whole tree once (fix any failures before committing). A normal `git commit` will run the pre-commit hooks automatically; a non-conventional message will be rejected by the commit-msg hook.

If [.pre-commit-config.yaml](.pre-commit-config.yaml) changes on `main`, run `pre-commit install-hooks` again (or `pre-commit clean` then reinstall) so your local hook environments stay in sync.

## Code Style

Make sure your code roughly follows [PEP-8](https://www.python.org/dev/peps/pep-0008/) and keeps things consistent with the rest of the code:

- **Docstrings**: [Sphinx-style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format) is used to write technical documentation.
- **Formatting**: [ruff](https://docs.astral.sh/ruff/formatter/) is used to automatically format the code.
- **Static analysis**: [ruff](https://docs.astral.sh/ruff/) is used for linting.

## Code Contribution Process

1. Ensure your contribution addresses an existing issue or discussion topic in the repository. If it does not, please open an issue to discuss your idea before starting.

2. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes and ensure they follow the project's coding standards.

4. Test your changes locally (see Testing section below).

5. Commit your changes using [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add a clear description of your changes"
   ```
   Commit prefixes drive version bumps and changelog entries:
   - `feat:` — new feature (minor version bump)
   - `fix:` — bug fix (patch version bump)
   - `feat!:` or `fix!:` — breaking change (major version bump)
   - `docs:`, `chore:`, `refactor:` — no version bump, may appear in changelog

6. Push your changes to your forked repository:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Open a Pull Request (PR) against the main repository. Ensure your PR description includes the problem it solves, a summary of changes, and any additional notes.

   CI validates that the **PR title** follows Conventional Commits (for example `feat: ...`, `fix: ...`). If your repository uses **squash merge** and defaults the squash commit message to the PR title, that title is what lands on `main` and drives [Release Please](#releases)—keep it conventional before merging.

## Testing

### Prerequisites

Install development dependencies:

```bash
pip install ruff qgis-stubs
```

`qgis-stubs` is required so ruff recognizes imports from QGIS. If you already set up [Git Hooks](#git-hooks) with `pip install -e ".[dev]"`, you have `ruff` from that install; add `qgis-stubs` if you have not already.

### Code Formatting

Format code with ruff:

```bash
ruff format .
```

### Linting and Auto-fix

Check for issues and auto-fix:

```bash
ruff check --fix .
```

### Tests

Build the Docker image for testing:

```bash
docker build -t qgis-test .
```

Run tests:

```bash
./run_tests.sh
```

**Headless processing (video / color / focus)** on a GeoTIFF, optionally with a KML ROI, uses the same Docker image and `PYTHONPATH` as tests:

```bash
# crop requires a KML (or OGR vector) ROI path as the second positional argument.
# Data outside the repo is not visible in Docker unless you mount it, for example:
#   ICEYE_CLI_DATA_ROOT=/path/to/folder/containing/tifs ./run_cli.sh crop --output-dir /tmp/out \
#     /iceye_data/ICEYE_..._SLC.tif /plugins/ICEYE_toolbox/test/fixtures/minimal_roi.kml
./run_cli.sh crop --output-dir /path/to/out /path/to/full.tif /path/to/roi.kml
./run_cli.sh video --frames 4 --output-dir /path/to/out test/ICEYE_*_CROP_*.tif
./run_cli.sh color --mode slow_time test/ICEYE_*_CROP_*.tif
./run_cli.sh focus /path/to/full.tif /path/to/roi.kml
```

See [`scripts/iceye_cli.py`](scripts/iceye_cli.py) for full usage.

If possible, add tests to cover your changes and ensure they pass.

## Releases

Releases are automated with [Release Please](https://github.com/googleapis/release-please). When you merge pull requests whose **merged commits** use conventional messages (`feat:`, `fix:`, etc.)—typically by using a conventional **PR title** with squash merge—Release Please opens a **Release PR** that updates `CHANGELOG.md` and bumps the version. Merging that Release PR creates the GitHub release and tag.

**Configuration:** Release Please [manifest mode](https://github.com/googleapis/release-please/blob/main/docs/manifest-releaser.md) requires [`.release-please-manifest.json`](.release-please-manifest.json) (current version for the root package, key `"."`) alongside [`release-please-config.json`](release-please-config.json). Release Please updates the manifest when release PRs merge; keep the `"."` version in sync with `pyproject.toml` / `metadata.txt` if you adjust things manually.

**Bootstrap (first-time setup):** Create tag `v1.0.0` on `main` so Release Please knows the baseline:
```bash
git tag v1.0.0
git push origin v1.0.0
```

## Questions or Help?

If you have any questions or need assistance, feel free to reach out by opening an [issue](https://github.com/iceye-ltd/iceye-qgis-plugins/issues).
