#!/usr/bin/env bash
set -euo pipefail

poetry check
poetry run pytest -q
poetry build
