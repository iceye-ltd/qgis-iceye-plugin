#!/usr/bin/env bash

set -e

# Usage: ./install.sh [linux|macos|windows]
os="${1:-$(uname -s)}"
os="$(printf '%s' "$os" | tr '[:upper:]' '[:lower:]')"

case "$os" in
  linux)
    dst="$HOME/.local/share/QGIS/QGIS3/profiles/default/python/plugins"
    ;;
  macos|darwin)
    dst="$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins"
    ;;
  windows|win)
    dst="$APPDATA/QGIS/QGIS3/profiles/default/python/plugins"
    ;;
  *)
    echo "Unknown OS: '$os'. Use 'linux', 'macos', or 'windows'."
    exit 1
    ;;
esac

echo "Target plugins directory: $dst"

if [ ! -d "$dst" ]; then
    echo "Plugins directory doesn't exist. Creating it now..."
    mkdir -p "$dst"
fi

ln -sfn "${PWD}" "${dst}/iceye_toolbox"

echo "All plugins linked for '$os'."
