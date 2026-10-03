#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
python3 -m venv Optional/vibevoice-1.5b-env
Optional/vibevoice-1.5b-env/bin/python -m pip install \
  'transformers @ git+https://github.com/huggingface/transformers.git@c587bc884db2c2e31fc2b8102314656b17aa07b1' \
  'torch==2.14.0' 'accelerate==1.15.0' 'diffusers==0.40.0' \
  'soundfile==0.14.0' 'librosa==1.0.0'
Optional/vibevoice-1.5b-env/bin/python Scripts/prepare-vibevoice-1.5b.py
