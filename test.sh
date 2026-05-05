#!/usr/bin/env bash

set -euo pipefail

echo $(pwd)

xvfb-run pytest test -v -s 
