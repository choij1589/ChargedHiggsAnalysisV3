#!/bin/bash
PORT=22
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in 
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        *) # positional argument
            break
            ;;
    esac
done

REMOTE="knu"
REMOTE_BASE="/d0/scratch/choij/ChargedHiggsAnalysisV3/SignalRegionStudyV2"
LOCAL_BASE="$PWD"

echo "Starting rsync with port $PORT"
[[ -n "$DRY_RUN" ]] && echo "Dry run mode enabled"

rsync -avz --progress --size-only ${DRY_RUN} -e "ssh -p ${PORT}" \
    "${REMOTE}:${REMOTE_BASE}/templates" "${LOCAL_BASE}/"
echo "Done."
