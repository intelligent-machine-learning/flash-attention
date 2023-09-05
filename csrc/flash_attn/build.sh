#!/usr/bin/bash
set -e

# cmake -S . -B build
cmake --build build -j48

# /usr/bin/cp build/fmhalib.cpython-38-x86_64-linux-gnu.so /opt/conda/lib/python3.8/site-packages/apex-0.1-py3.8-linux-x86_64.egg/

/usr/bin/cp build/flash_attn_cuda.cpython-38-x86_64-linux-gnu.so /opt/conda/lib/python3.8/site-packages/flash_attn-0.2.6.post2-py3.8-linux-x86_64.egg/

