#!/bin/bash
ERA=$1
export PATH=$WORKDIR/ExampleRun/python:$PATH

plot.py --era $ERA --histkey POGMedium_Central/ZCand_Mass_POGMedium_Central --debug
plot.py --era $ERA --histkey POGTight_Central/ZCand_Mass_POGTight_Central

