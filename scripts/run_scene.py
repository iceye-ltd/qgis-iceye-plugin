"""Run ``scripts/shear_averaging.py`` over every ``.npy`` patch in a directory.

Each patch is processed in its own subprocess (the heavy lifting and the
PNG saving all live inside ``shear_averaging.py``). Unknown CLI arguments
to this wrapper are forwarded verbatim to ``shear_averaging.py``, so e.g.

    python scripts/run_scene.py --patches-dir /path/to/patches \\
        --n-subaperture 8 --cov-th-mult 1.2

passes ``--n-subaperture 8 --cov-th-mult 1.2`` to every per-patch
invocation. ``--no-show`` is always added and ``MPLBACKEND=Agg`` is set
so the loop runs without an X display.

NOTE: do **not** pass ``--x-c-min`` / ``--x-c-max`` — that range-pixel
band gate was removed permanently (it silently emptied cropped patches
narrower than its hard-coded default).

A per-patch summary line is printed (rc + elapsed); the final line is a
roll-up. Exit code is non-zero iff any patch failed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PATCHES_DIR = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/patches"
)
SHEAR_SCRIPT = Path(__file__).resolve().parent / "shear_averaging.py"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--patches-dir", type=Path, default=DEFAULT_PATCHES_DIR,
        help=f"Directory containing patches. Default: {DEFAULT_PATCHES_DIR}",
    )
    parser.add_argument(
        "--glob", default="*.npy",
        help="Glob pattern relative to --patches-dir. Default: *.npy",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands without running them.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Don't bail on the first non-zero return code; finish the loop "
             "and report failures at the end.",
    )
    known_args, forward_args = parser.parse_known_args()

    patches = sorted(known_args.patches_dir.glob(known_args.glob))
    if not patches:
        print(
            f"No patches matched {known_args.patches_dir}/{known_args.glob}",
            file=sys.stderr,
        )
        return 1

    if not SHEAR_SCRIPT.exists():
        print(f"shear_averaging.py not found at {SHEAR_SCRIPT}", file=sys.stderr)
        return 1

    # Headless matplotlib so we don't depend on a display in the subprocesses.
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    print(f"Found {len(patches)} patch(es) in {known_args.patches_dir}")
    if forward_args:
        print(f"Forwarded args: {' '.join(forward_args)}")
    print(f"MPLBACKEND={env['MPLBACKEND']}")

    n_ok = 0
    failures: list[tuple[Path, int]] = []
    t0_all = time.monotonic()
    for i, patch in enumerate(patches, 1):
        cmd = [
            sys.executable, str(SHEAR_SCRIPT), str(patch),
            "--no-show", *forward_args,
        ]
        print(f"\n[{i}/{len(patches)}] {patch.name}")
        print(f"  $ {' '.join(cmd)}")
        if known_args.dry_run:
            continue
        t0 = time.monotonic()
        # Stream the child's output as it runs; don't capture it.
        rc = subprocess.call(cmd, env=env)
        elapsed = time.monotonic() - t0
        if rc == 0:
            n_ok += 1
            print(f"  -> rc=0  ({elapsed:.1f}s)")
        else:
            failures.append((patch, rc))
            print(f"  -> rc={rc}  ({elapsed:.1f}s)")
            if not known_args.continue_on_error:
                elapsed_all = time.monotonic() - t0_all
                print(
                    f"\nAborted after {patch.name} "
                    f"(use --continue-on-error to keep going). "
                    f"Total elapsed {elapsed_all:.1f}s, "
                    f"{n_ok} ok, {len(failures)} failed."
                )
                return rc

    elapsed_all = time.monotonic() - t0_all
    print(
        f"\nDone in {elapsed_all:.1f}s: {n_ok} ok, {len(failures)} failed "
        f"(of {len(patches)})"
    )
    if failures:
        print("Failed patches:")
        for patch, rc in failures:
            print(f"  rc={rc:<4d} {patch}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
