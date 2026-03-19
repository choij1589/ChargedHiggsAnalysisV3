#!/bin/bash
# Source this file to load mass point arrays from configs/masspoints.json
MASSPOINTS_JSON="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/masspoints.json"

# Parse all arrays in a single Python invocation
_mp_all=$(python3 -c "
import json
d = json.load(open('$MASSPOINTS_JSON'))
print(' '.join(d['baseline']))
print(' '.join(d['particlenet']))
print(' '.join(d['run3_real_mc']))
print(' '.join(d['partial_unblind']))
print(' '.join(d['impact']['baseline']))
print(' '.join(d['impact']['particlenet']))
print(' '.join(d['signal_injection']['baseline']))
print(' '.join(d['signal_injection']['particlenet']))
print(' '.join(d['hybridnew']['baseline']))
print(' '.join(d['hybridnew']['particlenet']))
print(' '.join(d['gof']['baseline']))
print(' '.join(d['gof']['particlenet']))
")

read -ra MASSPOINTs_BASELINE        <<< "$(sed -n '1p' <<< "$_mp_all")"
read -ra MASSPOINTs_PARTICLENET     <<< "$(sed -n '2p' <<< "$_mp_all")"
read -ra MASSPOINTs_Run3            <<< "$(sed -n '3p' <<< "$_mp_all")"
read -ra MASSPOINTs_PARTIAL_UNBLIND <<< "$(sed -n '4p' <<< "$_mp_all")"
read -ra MASSPOINTs_IMPACT_BASELINE <<< "$(sed -n '5p' <<< "$_mp_all")"
read -ra MASSPOINTs_IMPACT_PN       <<< "$(sed -n '6p' <<< "$_mp_all")"
read -ra MASSPOINTs_SIGINJ_BASELINE <<< "$(sed -n '7p' <<< "$_mp_all")"
read -ra MASSPOINTs_SIGINJ_PN       <<< "$(sed -n '8p' <<< "$_mp_all")"
read -ra MASSPOINTs_HYBRIDNEW_BASELINE <<< "$(sed -n '9p' <<< "$_mp_all")"
read -ra MASSPOINTs_HYBRIDNEW_PN    <<< "$(sed -n '10p' <<< "$_mp_all")"
read -ra MASSPOINTs_GOF_BASELINE    <<< "$(sed -n '11p' <<< "$_mp_all")"
read -ra MASSPOINTs_GOF_PN          <<< "$(sed -n '12p' <<< "$_mp_all")"
unset _mp_all
