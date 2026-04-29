#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
EXTRA_ARGS="${*:-}"

docker run --rm -t \
    --shm-size=2g \
    -v "${SCRIPT_DIR}:/plugins/iceye_toolbox" \
    -e QT_QPA_PLATFORM=offscreen \
    -e PYTHONPATH=/plugins:/usr/share/qgis/python/:/usr/share/qgis/python/plugins:/usr/lib/python3/dist-packages/qgis:/usr/share/qgis/python/qgis \
    -w /plugins/iceye_toolbox \
    qgis-test:latest \
    pytest test -v -s ${EXTRA_ARGS}