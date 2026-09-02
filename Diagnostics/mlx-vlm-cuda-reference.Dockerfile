FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

RUN python -m pip install --break-system-packages --no-cache-dir \
    "mlx[cuda13]==0.32.1" \
    mlx-vlm==0.6.16
