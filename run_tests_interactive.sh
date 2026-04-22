#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Start interactive Docker container
docker run -it \
    -v "${SCRIPT_DIR}:/plugins/iceye_toolbox" \
    -e DISPLAY=:99 \
    qgis-test:latest \
    bash -c "Xvfb :99 -screen 0 1024x768x24 & bash"