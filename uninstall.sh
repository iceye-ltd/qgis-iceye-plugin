#!/usr/bin/env bash

set -e

# Usage: ./uninstall.sh [linux|macos|windows]
# Removes the symlink created by install.sh from the QGIS plugins directory.
# Run this from the same plugin repository root where you ran install.sh.

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

link="${dst}/iceye_toolbox"

if [ ! -e "$link" ] && [ ! -L "$link" ]; then
  echo "Nothing to remove: $link does not exist."
  exit 0
fi

if [ ! -L "$link" ]; then
  echo "Refusing to remove $link: it is not a symlink (install.sh only creates a symlink)."
  exit 1
fi

repo_root="$(cd "$(dirname "$0")" && pwd -P)"

if command -v realpath >/dev/null 2>&1; then
  target="$(realpath "$link")"
elif target="$(readlink -f "$link" 2>/dev/null)"; then
  :
else
  target="$(readlink "$link")"
  case "$target" in
    /*) ;;
    *) target="$(cd "$(dirname "$link")" && pwd -P)/${target}" ;;
  esac
fi

if [ "$target" != "$repo_root" ]; then
  echo "Refusing to remove $link: it points to:"
  echo "  $target"
  echo "Expected this repository root:"
  echo "  $repo_root"
  exit 1
fi

rm "$link"
echo "Removed symlink $link (uninstall complete for '$os')."
