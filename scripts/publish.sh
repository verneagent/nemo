#!/usr/bin/env bash
# Publish captain-nemo to PyPI.
#
# Deterministic release flow. This exists because a manual release once ran
# `python3 -m build --wheel` (wheel only) and then `twine upload <whl> <sdist>`,
# which aborted with "Cannot find file ...tar.gz" — the sdist was never built.
# This script makes that impossible: it always builds BOTH artifacts before
# uploading, and `--skip-existing` makes retries safe.
#
# Usage:
#   scripts/publish.sh        # build wheel + sdist, upload to PyPI
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Building wheel + sdist =="
rm -rf dist/ build/
python3 -m build --wheel --sdist

echo "== Uploading to PyPI =="
python3 -m twine upload --skip-existing dist/captain_nemo-*.whl dist/captain_nemo-*.tar.gz

echo "== Published =="
ls -la dist/
