#!/bin/bash
HOST=`hostname`
echo "Hello, $HOST"
 
if [[ $HOST == *"tamsa"* ]]; then
    export WORKDIR="/data9/Users/choij/Sync/workspace/ChargedHiggsAnalysisV3"
    export PATH=$HOME/micromamba/bin:$PATH
    export MAMBA_ROOT_PREFIX=$HOME/micromamba
    eval "$(micromamba shell hook -s zsh)"
    micromamba activate Nano
elif [[ $HOST == *"Mac"* ]]; then
    export WORKDIR="/Users/choij/Sync/workspace/ChargedHiggsAnalysisV3"
    export PATH=$HOME/micromamba/bin:$PATH
    export MAMBA_ROOT_PREFIX=$HOME/micromamba
    eval "$(micromamba shell hook -s zsh)"
    micromamba activate Nano
elif [[ $HOST == *"private-snu"* ]]; then
    export WORKDIR="/home/choij/Sync/workspace/ChargedHiggsAnalysisV3"
    export PATH=$HOME/micromamba/bin:$PATH
    export MAMBA_ROOT_PREFIX=$HOME/micromamba
    eval "$(micromamba shell hook -s zsh)"
    micromamba activate Nano
else
    echo "Is it nersc?"
    export WORKDIR="/pscratch/sd/c/choij/workspace/ChargedHiggsAnalysisV3"
    source $HOME/miniconda3/bin/activate
    conda activate Nano
fi
echo "@@@@ WORKDIR: $WORKDIR"

export PYTHONPATH=$WORKDIR/Common/Tools:$PYTHONPATH
