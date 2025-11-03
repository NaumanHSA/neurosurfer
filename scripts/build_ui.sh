#!/usr/bin/env bash
# Build React app in ./neurosurferui and sync compiled assets into ./neurosurfer/ui_build
# Usage: ./scripts/build_ui.sh [--no-install] [--no-clean] [--prod] [--out-dir dist|build]

set -euo pipefail

# -------- Resolve repo root (works no matter where you run from) --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# -------- Config --------
UI_DIR="$REPO_ROOT/neurosurferui"
PY_PKG_DIR="$REPO_ROOT/neurosurfer"
TARGET_DIR="$PY_PKG_DIR/ui_build"

DO_INSTALL=true
DO_CLEAN=true
NPM_FLAGS=()
OUT_DIR_OVERRIDE=""

for arg in "$@"; do
  case "$arg" in
    --no-install) DO_INSTALL=false ;;
    --no-clean)   DO_CLEAN=false ;;
    --prod)       NPM_FLAGS+=("--production") ;;
    --out-dir)
      echo "Error: --out-dir needs a value (dist|build)" >&2; exit 2 ;;
    --out-dir=*)  OUT_DIR_OVERRIDE="${arg#*=}" ;;
    *)
      echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

# -------- Helpers --------
need() { command -v "$1" >/dev/null 2>&1 || { echo "Error: '$1' is required."; exit 1; }; }
log()  { printf "\033[1;34m[build-ui]\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m[build-ui]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[build-ui]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[build-ui]\033[0m %s\n" "$*"; }

need node; need npm

[[ -d "$UI_DIR" ]] || { err "Missing UI dir: $UI_DIR"; exit 1; }
[[ -d "$PY_PKG_DIR" ]] || { err "Missing Python pkg dir: $PY_PKG_DIR"; exit 1; }

# -------- Build --------
pushd "$UI_DIR" >/dev/null

if $DO_INSTALL; then
  if [[ -f package-lock.json ]]; then
    log "npm ci --force"
    npm ci --force
  else
    warn "No package-lock.json; running npm install"
    npm install --force
  fi
fi

export NODE_ENV=production
log "npm run build ${NPM_FLAGS[*]:-}"
npm run build "${NPM_FLAGS[@]:-}"

popd >/dev/null

# -------- Detect build output folder --------
UI_BUILD_DIR=""
if [[ -n "$OUT_DIR_OVERRIDE" ]]; then
  if [[ -d "$UI_DIR/$OUT_DIR_OVERRIDE" ]]; then
    UI_BUILD_DIR="$UI_DIR/$OUT_DIR_OVERRIDE"
  else
    err "Override out dir '$OUT_DIR_OVERRIDE' not found under $UI_DIR"
    exit 1
  fi
else
  # Try Vite first, then CRA
  for cand in "dist" "build"; do
    if [[ -d "$UI_DIR/$cand" ]]; then
      UI_BUILD_DIR="$UI_DIR/$cand"
      break
    fi
  done
fi

[[ -n "$UI_BUILD_DIR" ]] || { err "Could not find build output (tried 'dist' and 'build')."; exit 1; }
log "Using UI build dir: $UI_BUILD_DIR"

# -------- Sync to Python package --------
log "Preparing target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

if command -v rsync >/dev/null 2>&1; then
  log "Syncing with rsync (deleting removed files, excluding maps)"
  rsync -a --delete \
    --exclude='*.map' \
    --exclude='.DS_Store' \
    "$UI_BUILD_DIR"/ "$TARGET_DIR"/
else
  warn "rsync not found; falling back to rm+cp"
  if $DO_CLEAN; then
    log "Cleaning existing target directory"
    rm -rf "${TARGET_DIR:?}/"*
  fi
  cp -a "$UI_BUILD_DIR"/. "$TARGET_DIR"/
  find "$TARGET_DIR" -type f -name '*.map' -delete || true
fi

# -------- Post checks --------
if [[ ! -f "$TARGET_DIR/index.html" ]]; then
  err "index.html not found in '$TARGET_DIR'. Did the build succeed?"
  exit 1
fi

ok "UI build synced to '$TARGET_DIR'"
ok "Remember to include in packaging:
  - pyproject.toml -> [tool.setuptools.package-data] neurosurfer = [\"ui_build/**\"]
  - MANIFEST.in    -> graft neurosurfer/ui_build"
