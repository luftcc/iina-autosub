#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_DIR="$ROOT_DIR/plugin"
DIST_DIR="$ROOT_DIR/dist"
OUT_FILE="$DIST_DIR/iina-autosub.iinaplgz"

mkdir -p "$DIST_DIR"
rm -f "$OUT_FILE"

(
  cd "$PLUGIN_DIR"
  zip -r -q "$OUT_FILE" Info.json main.js global.js ui/
)

echo "Packed IINA plugin: $OUT_FILE"
