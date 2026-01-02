#!/bin/bash

python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" --no-build-isolation
python -m pip install vllm==0.10.2

python -m pip install --upgrade -r requirements.txt