#!/usr/bin/env bash
 
set -e
 
# Usage: ./install.sh [linux|macos|windows]
os="${1:-$(uname -s)}"
os="$(printf '%s' "$os" | tr '[:upper:]' '[:lower:]')"
 
case "$os" in
  linux)
    base="$HOME/.local/share/QGIS"
    ;;
  macos|darwin)
    base="$HOME/Library/Application Support/QGIS"
    ;;
  windows|win)
    base="$APPDATA/QGIS"
    ;;
  *)
    echo "Unknown OS: '$os'. Use 'linux', 'macos', or 'windows'."
    exit 1
    ;;
esac
 
qgis_dir="QGIS3"
for v in QGIS4 QGIS3; do
  if [ -d "$base/$v/profiles/default/python" ]; then
    qgis_dir="$v"
    break
  fi
done
 
dst="$base/$qgis_dir/profiles/default/python/plugins"
 
echo "Target plugins directory: $dst"
 
if [ ! -d "$dst" ]; then
    echo "Plugins directory doesn't exist. Creating it now..."
    mkdir -p "$dst"
fi
 
ln -sfn "${PWD}" "${dst}/iceye_toolbox"
 
echo "All plugins linked for '$os' ($qgis_dir)."
 