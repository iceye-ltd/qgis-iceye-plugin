
#!/usr/bin/env bash
 
set -e
 
# Usage: ./uninstall.sh [linux|macos|windows]
# Removes the symlink created by install.sh from the QGIS plugins directory.
# Run this from the same plugin repository root where you ran install.sh.
 
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
 
# Match install.sh: use whichever profile folder exists (QGIS4 or QGIS3).
qgis_dir="QGIS3"
for v in QGIS4 QGIS3; do
  if [ -d "$base/$v/profiles/default/python" ]; then
    qgis_dir="$v"
    break
  fi
done
 
dst="$base/$qgis_dir/profiles/default/python/plugins"
 
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