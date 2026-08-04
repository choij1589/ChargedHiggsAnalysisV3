#!/bin/bash
# Source this file to load mass point arrays from configs/masspoints.json
MASSPOINTS_JSON="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/masspoints.json"

# Parse all arrays in a single Python invocation
_mp_all=$(python3 -c "
import json
d = json.load(open('$MASSPOINTS_JSON'))
baseline = d.get('baseline', [])
particlenet = d.get('particlenet', [])
ptoptimized = d.get('ptoptimized', [])
impact = d.get('impact', {})
signal_injection = d.get('signal_injection', {})
hybridnew = d.get('hybridnew', {})
gof = d.get('gof', {})
lee = d.get('LEE', [])

print(' '.join(baseline))
print(' '.join(particlenet))
print(' '.join(d.get('partial_unblind', particlenet)))
print(' '.join(impact.get('baseline', baseline)))
print(' '.join(impact.get('particlenet', particlenet)))
print(' '.join(signal_injection.get('baseline', impact.get('baseline', baseline))))
print(' '.join(signal_injection.get('particlenet', impact.get('particlenet', particlenet))))
print(' '.join(hybridnew.get('baseline', baseline)))
print(' '.join(hybridnew.get('particlenet', particlenet)))
print(' '.join(gof.get('baseline', baseline)))
print(' '.join(gof.get('particlenet', particlenet)))
print(' '.join(lee))
print(' '.join(ptoptimized))
")

read -ra MASSPOINTs_BASELINE        <<< "$(sed -n '1p' <<< "$_mp_all")"
read -ra MASSPOINTs_PARTICLENET     <<< "$(sed -n '2p' <<< "$_mp_all")"
read -ra MASSPOINTs_PARTIAL_UNBLIND <<< "$(sed -n '3p' <<< "$_mp_all")"
read -ra MASSPOINTs_IMPACT_BASELINE <<< "$(sed -n '4p' <<< "$_mp_all")"
read -ra MASSPOINTs_IMPACT_PN       <<< "$(sed -n '5p' <<< "$_mp_all")"
read -ra MASSPOINTs_SIGINJ_BASELINE <<< "$(sed -n '6p' <<< "$_mp_all")"
read -ra MASSPOINTs_SIGINJ_PN       <<< "$(sed -n '7p' <<< "$_mp_all")"
read -ra MASSPOINTs_HYBRIDNEW_BASELINE <<< "$(sed -n '8p' <<< "$_mp_all")"
read -ra MASSPOINTs_HYBRIDNEW_PN    <<< "$(sed -n '9p' <<< "$_mp_all")"
read -ra MASSPOINTs_GOF_BASELINE    <<< "$(sed -n '10p' <<< "$_mp_all")"
read -ra MASSPOINTs_GOF_PN          <<< "$(sed -n '11p' <<< "$_mp_all")"
read -ra MASSPOINTs_LEE             <<< "$(sed -n '12p' <<< "$_mp_all")"
# Appended last on purpose: new entries must not shift the line numbers above.
read -ra MASSPOINTs_PTOPTIMIZED     <<< "$(sed -n '13p' <<< "$_mp_all")"
unset _mp_all
