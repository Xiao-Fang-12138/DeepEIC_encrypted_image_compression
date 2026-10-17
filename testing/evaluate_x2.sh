#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
python -m testing.evaluate --config configs/x2.json "$@"
