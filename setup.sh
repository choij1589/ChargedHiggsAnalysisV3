#!/bin/bash
HOST=`hostname`
echo "Hello, $HOST"
 
if [[ $HOST == *"ai-tamsa"* ]]; then
    export WORKDIR="/data9/Users/choij/ChargedHiggsAnalysisV3"
    source ~/.conda-activate
    conda activate pyg
else
    echo "Environment not configured for $HOST"
fi
echo "@@@@ WORKDIR: $WORKDIR"

export PYTHONPATH=$WORKDIR/Common/Tools/tdr-style:$PYTHONPATH
