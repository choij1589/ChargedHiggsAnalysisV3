#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(dirname "$SCRIPT_DIR")"

cd "$CPP_DIR"
rm -rf build lib
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX="$CPP_DIR" \
      -DROOT_DIR="$ROOTSYS/cmake" \
      "$CPP_DIR"
make -j4 && make install
cd -
echo "Build complete: $CPP_DIR/lib/libCommonTools.so"
