#!/bin/bash
ERA=$1
export PATH=$WORKDIR/ExampleRun/python:$PATH

plot.py --era $ERA --histkey Central/ZCand_Mass_Central
