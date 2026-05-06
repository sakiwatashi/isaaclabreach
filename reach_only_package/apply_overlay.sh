#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <IsaacLab_path>" >&2
  exit 1
fi

REPO="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
cp -a "$HERE/code/." "$REPO/"
