#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
python -m training.train --config configs/x4.json "$@"
