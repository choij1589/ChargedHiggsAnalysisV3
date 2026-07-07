#!/bin/bash
cd "$(dirname "$0")"
echo "DAG status:"
if [[ -f dag.dag.dagman.out ]]; then
    done=$(grep -c "ULOG_JOB_TERMINATED" dag.dag.dagman.out 2>/dev/null || echo 0)
    total=$(grep -c "^JOB " dag.dag 2>/dev/null || echo 0)
    echo "  $done / $total jobs completed"
else
    echo "  not started"
fi
condor_q -dag 2>/dev/null || true
