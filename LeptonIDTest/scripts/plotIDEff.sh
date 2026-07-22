#!/bin/bash
export PATH="$PWD/python:$PATH"
ERA=$1

if [[ -z "$ERA" ]]; then
    echo "Usage: bash scripts/plotIDEff.sh <era>"
    exit 1
fi

for OBJECT in muon electron; do
    plotIDEff.py --era "$ERA" --object "$OBJECT"
done
