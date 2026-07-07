#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
condor_submit_dag -config dagman.config dag.dag
