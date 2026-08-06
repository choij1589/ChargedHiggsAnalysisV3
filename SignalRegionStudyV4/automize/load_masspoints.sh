#!/bin/bash
# Source this file to load mass point arrays from configs/masspoints.json.
# Name-keyed parsing: each line is "<KEY>=<space-separated list>", so adding
# or reordering keys can never silently shift another array.
MASSPOINTS_JSON="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/masspoints.json"

_mp_all=$(python3 -c "
import json
d = json.load(open('$MASSPOINTS_JSON'))
for key in ('baseline', 'particlenet', 'limits'):
    print(f\"{key}=\" + ' '.join(d[key]))
")

while IFS='=' read -r _mp_key _mp_values; do
    case "$_mp_key" in
        baseline)    read -ra MASSPOINTs_BASELINE    <<< "$_mp_values" ;;
        particlenet) read -ra MASSPOINTs_PARTICLENET <<< "$_mp_values" ;;
        limits)      read -ra MASSPOINTs_LIMITS      <<< "$_mp_values" ;;
    esac
done <<< "$_mp_all"
unset _mp_all _mp_key _mp_values

if [[ ${#MASSPOINTs_BASELINE[@]} -eq 0 || ${#MASSPOINTs_PARTICLENET[@]} -eq 0 ]]; then
    echo "Error: failed to load mass points from $MASSPOINTS_JSON" >&2
    return 1 2>/dev/null || exit 1
fi
