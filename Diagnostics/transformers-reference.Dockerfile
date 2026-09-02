FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

RUN python -m pip install --break-system-packages --no-cache-dir \
    transformers==5.7.0 \
    accelerate==1.10.1 \
    sentencepiece==0.2.1
