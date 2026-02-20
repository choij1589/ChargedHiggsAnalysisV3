#!/bin/bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
SRC_DIR="${WORKDIR}/TriLepton/results"
DST_DIR="${WORKDIR}/Common/Data/ConvSF"

# Run2 eras
RUN2_ERAS=("2016preVFP" "2016postVFP" "2017" "2018")
# Run3 eras
RUN3_ERAS=("2022" "2022EE" "2023" "2023BPix")

# Channels: source directory -> destination directory
declare -A CHANNEL_MAP
CHANNEL_MAP["ZG1E2Mu"]="1E2Mu"
CHANNEL_MAP["ZG3Mu"]="3Mu"

# Copy Run2
for src_channel in "${!CHANNEL_MAP[@]}"; do
    dst_channel="${CHANNEL_MAP[$src_channel]}"
    dst_path="${DST_DIR}/Run2/${dst_channel}"
    mkdir -p "$dst_path"
    for era in "${RUN2_ERAS[@]}"; do
        src_file="${SRC_DIR}/${src_channel}/${era}/ConvSF.json"
        if [[ -f "$src_file" ]]; then
            cp "$src_file" "${dst_path}/${era}.json"
            echo "Copied ${src_file} -> ${dst_path}/${era}.json"
        else
            echo "WARNING: ${src_file} not found"
        fi
    done
done

# Copy Run3
for src_channel in "${!CHANNEL_MAP[@]}"; do
    dst_channel="${CHANNEL_MAP[$src_channel]}"
    dst_path="${DST_DIR}/Run3/${dst_channel}"
    mkdir -p "$dst_path"
    for era in "${RUN3_ERAS[@]}"; do
        src_file="${SRC_DIR}/${src_channel}/${era}/ConvSF.json"
        if [[ -f "$src_file" ]]; then
            cp "$src_file" "${dst_path}/${era}.json"
            echo "Copied ${src_file} -> ${dst_path}/${era}.json"
        else
            echo "WARNING: ${src_file} not found"
        fi
    done
done

echo "Done. ConvSF files copied to ${DST_DIR}"
