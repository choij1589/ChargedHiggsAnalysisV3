#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python python/parseSFToCommonData.py
