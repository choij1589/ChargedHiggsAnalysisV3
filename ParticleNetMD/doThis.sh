#!/bin/bash

python3 python/resaveTree.py --signal MHc130_MA90 --channel Combined \
          --base-dir LambdaSweep/Combined/MHc130_MA90/fold-4 --device cuda:0 &

python3 python/resaveTree.py --signal MHc160_MA85 --channel Combined \
          --base-dir LambdaSweep/Combined/MHc160_MA85/fold-4 --device cuda:1 &

python3 python/resaveTree.py --signal MHc100_MA95 --channel Combined \
          --base-dir LambdaSweep/Combined/MHc100_MA95/fold-4 --device cuda:2 &
