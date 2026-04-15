#!/usr/bin/env bash
# Run iceye_cli.py inside the same QGIS Docker image as tests (headless).
#
# Examples (from repo root, after: docker build -t qgis-test .):
#   # Mount a host data directory at /iceye_data inside the container (read-only):
#   ICEYE_CLI_DATA_ROOT=/path/to/data ./run_cli.sh crop --output-dir /tmp/out \
#     /iceye_data/ICEYE_..._SLC.tif /plugins/ICEYE_toolbox/test/fixtures/minimal_roi.kml
#   ./run_cli.sh crop --output-dir /tmp/out path/to/full.tif path/to/roi.kml  # KML required
#   ./run_cli.sh video test/ICEYE_D1X6JD_20251107T033407Z_6934716_X50_SLH_CROP_6717cb0a.tif
#   ./run_cli.sh color --mode slow_time --output-dir /tmp/out test/ICEYE_*_CROP_*.tif
#   ./run_cli.sh focus /path/to/full.tif /path/to/roi.kml

set -euo pipefail
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

EXTRA_DOCKER_ARGS=()
if [[ -n "${ICEYE_CLI_DATA_ROOT:-}" ]]; then
    EXTRA_DOCKER_ARGS+=( -v "${ICEYE_CLI_DATA_ROOT}:/iceye_data:ro" )
fi

docker run --rm -t \
    --shm-size=2g \
    "${EXTRA_DOCKER_ARGS[@]}" \
    -v "${SCRIPT_DIR}:/plugins/ICEYE_toolbox" \
    -e QT_QPA_PLATFORM=offscreen \
    -e PYTHONPATH=/plugins:/usr/share/qgis/python/:/usr/share/qgis/python/plugins:/usr/lib/python3/dist-packages/qgis:/usr/share/qgis/python/qgis \
    -w /plugins/ICEYE_toolbox \
    qgis-test:latest \
    python3 scripts/iceye_cli.py "$@"
