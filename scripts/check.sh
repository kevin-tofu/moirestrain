#!/usr/bin/env bash
set -euo pipefail

poetry check
poetry run python scripts/update_version.py --check
poetry run pytest -q
poetry build
