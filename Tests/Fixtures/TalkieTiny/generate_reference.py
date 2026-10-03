#!/usr/bin/env python3
"""Generate synthetic Talkie fixtures independently of Midnight's Swift model.

Requires Python MLX (generated with 0.31.2). No model download, PyTorch, NumPy,
random seed, or mlx_lm imports. Equations follow talkie-lm/talkie model.py:
https://github.com/talkie-lm/talkie/blob/main/src/talkie/model.py
The context-cache extension concatenates independently rotated/normalized K/V.
"""

import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parent
TOKENS = [3, 1, 4, 1, 5, 9, 2]
CONFIG = {
    "model_type": "talkie",
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "head_dim": 8,
    "intermediate_size": 32,
    "vocab_size": 32,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1.1920928955078125e-7,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
}


def matrix(shape, index, dtype):
    count = math.prod(shape)
    # Deterministic, nonseparable, unequal Q/K/V and gate/up projections.
    values = [
        0.19 * math.sin((i + 1) * (0.173 + index * 0.019))
        + 0.11 * math.cos((i + 3) * (0.071 + index * 0.013))
        for i in range(count)
    ]
    return mx.array(values, dtype=mx.float32).reshape(shape).astype(dtype)


def weights(config, dtype):
    h, f, v = (config[k] for k in ["hidden_size", "intermediate_size", "vocab_size"])
    result = {
        "model.embed.weight": matrix((v, h), 1, dtype),
        "lm_head": matrix((v, h), 2, dtype),
        "lm_head_gain.w_g": mx.array([1.375], dtype=dtype),
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.blocks.{layer}."
        for index, name in enumerate(["query", "key", "value", "resid"]):
            result[prefix + f"attn.attn_{name}.weight"] = matrix(
                (h, h), 3 + 7 * layer + index, dtype
            )
        for index, name in enumerate(["gate", "linear", "resid"]):
            result[prefix + f"mlp.mlp_{name}.weight"] = matrix(
                (h, f) if name == "resid" else (f, h), 7 + 7 * layer + index, dtype
            )
        result[prefix + "attn.head_gain.head_g"] = mx.array(
            [0.625 + layer * 0.125, 1.625 - layer * 0.125], dtype=dtype
        )
        for name, value in [
            ("attn_gain", 0.625 - layer * 0.125),
            ("mlp_gain", 0.375 + layer * 0.125),
            ("embed_skip", 0.25 if layer == 0 else -0.375),
        ]:
            result[prefix + name + ".a_g"] = mx.array([value], dtype=dtype)
    return result


def norm(x, eps):
    full = x.astype(mx.float32)
    return (full * mx.rsqrt(mx.mean(full * full, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def rotary(x, offset, theta, reverse=True):
    d = x.shape[-1]
    positions = mx.arange(offset, offset + x.shape[-2], dtype=mx.float32)
    freq = theta ** (-mx.arange(0, d, 2, dtype=mx.float32) / d)
    angle = positions[:, None] * freq[None, :]
    # Explicit inverse NeoX rotation, not mx.fast.rope from the Swift port.
    c, s = mx.cos(angle), mx.sin(angle)
    if not reverse:
        s = -s
    left, right = x[..., : d // 2].astype(mx.float32), x[..., d // 2 :].astype(mx.float32)
    return mx.concatenate([left * c + right * s, right * c - left * s], axis=-1).astype(x.dtype)


def forward(tokens, w, config, cache=None, *, reverse=True, embedding_skip=True, head_gain=True):
    def project(x, key):
        if key + ".scales" not in w:
            return x @ w[key + ".weight"].T
        # CPU affine QMM has a BF16 accumulator on the reference platform.
        # Dense GEMM over dequantized BF16 weights does not reproduce its
        # rounding. Only this primitive is shared; model equations stay explicit.
        return mx.quantized_matmul(
            x, w[key + ".weight"], w[key + ".scales"], w[key + ".biases"],
            transpose=True, group_size=config["quantization"]["group_size"], bits=4,
        )

    ids = mx.array([tokens], dtype=mx.int32)
    embed = w["model.embed.weight"]
    if "model.embed.scales" in w:
        embed = mx.dequantize(
            embed, w["model.embed.scales"], w["model.embed.biases"],
            group_size=config["quantization"]["group_size"], bits=4,
        )
    x = norm(embed[ids], config["rms_norm_eps"])
    embedding = x
    count = len(tokens)
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.blocks.{layer}."
        pre = norm(x, config["rms_norm_eps"])
        q, k, v = [
            project(pre, prefix + f"attn.attn_{name}")
            .reshape(1, count, config["num_attention_heads"], config["head_dim"])
            .transpose(0, 2, 1, 3)
            for name in ["query", "key", "value"]
        ]
        old = None if cache is None else cache[layer]
        offset = 0 if old is None else old[0].shape[2]
        q = norm(rotary(q, offset, config["rope_theta"], reverse), config["rms_norm_eps"])
        k = norm(rotary(k, offset, config["rope_theta"], reverse), config["rms_norm_eps"])
        if head_gain:
            q = q * w[prefix + "attn.head_gain.head_g"].reshape(1, -1, 1, 1)
        if old is not None:
            k = mx.concatenate([old[0], k], axis=2)
            v = mx.concatenate([old[1], v], axis=2)
        if cache is not None:
            cache[layer] = (k, v)
        # Independent dense attention, with explicit rectangular causal mask.
        scores = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) / math.sqrt(config["head_dim"])
        causal = mx.arange(k.shape[2])[None, :] <= mx.arange(offset, offset + count)[:, None]
        scores = mx.where(causal, scores, -float("inf"))
        attention = (mx.softmax(scores, axis=-1).astype(v.dtype) @ v)
        attention = attention.transpose(0, 2, 1, 3).reshape(1, count, config["hidden_size"])
        x = x + project(attention, prefix + "attn.attn_resid") * w[prefix + "attn_gain.a_g"]
        pre = norm(x, config["rms_norm_eps"])
        gate = project(pre, prefix + "mlp.mlp_gate")
        gate32 = gate.astype(mx.float32)
        gate = (gate32 * mx.sigmoid(gate32)).astype(gate.dtype)
        hidden = gate * project(pre, prefix + "mlp.mlp_linear")
        x = x + project(hidden, prefix + "mlp.mlp_resid") * w[prefix + "mlp_gain.a_g"]
        if embedding_skip:
            x = x + embedding * w[prefix + "embed_skip.a_g"]
    # Official PyTorch folds gain into the head weight before multiplication.
    if "lm_head.scales" in w:
        return project(norm(x, config["rms_norm_eps"]), "lm_head")
    head = w.get("lm_head.weight")
    if head is None:
        head = w["lm_head"] * w["lm_head_gain.w_g"]
    return norm(x, config["rms_norm_eps"]) @ head.T


def expected(w, config):
    result = {"full": forward(TOKENS, w, config)}
    cache = [None] * config["num_hidden_layers"]
    result["prefill"] = forward(TOKENS[:2], w, config, cache)
    result["continuation"] = forward(TOKENS[2:5], w, config, cache)
    result["decode"] = forward(TOKENS[5:6], w, config, cache)
    result["last"] = forward(TOKENS[6:], w, config, cache)
    return result


def save(name, arrays):
    mx.eval(list(arrays.values()))
    mx.save_safetensors(str(ROOT / name), arrays)


def main():
    # CPU reference avoids shape-dependent GPU GEMM rounding between prefill
    # and one-token decoding, keeping the golden calculation deterministic.
    mx.set_default_device(mx.cpu)
    (ROOT / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    manifest = {"generator": "generate_reference.py", "mlx_version": mx.__version__, "tokens": TOKENS}
    for name, dtype in [("fp32", mx.float32), ("bf16", mx.bfloat16)]:
        w = weights(CONFIG, dtype)
        save(f"weights-{name}.safetensors", w)
        out = expected(w, CONFIG)
        save(f"expected-{name}.safetensors", out)
        full = out["full"].astype(mx.float32)
        joined = mx.concatenate([out[k] for k in ["prefill", "continuation", "decode", "last"]], axis=1)
        manifest[name + "_reference_cache_max_error"] = float(mx.max(mx.abs(full - joined)))
        if name == "fp32":
            manifest["sensitivity_max_logit_error"] = {
                change: float(mx.max(mx.abs(full - forward(TOKENS, w, CONFIG, **kwargs))))
                for change, kwargs in {
                    "wrong_rope_sign": {"reverse": False},
                    "missing_embedding_skip": {"embedding_skip": False},
                    "missing_query_head_gain": {"head_gain": False},
                }.items()
            }

    # MLX affine groups require a 32-wide input; a second tiny shape covers the
    # packed checkpoint loader, including the pre-folded quantized LM head.
    qc = dict(CONFIG, hidden_size=32, head_dim=16, intermediate_size=64)
    qc["quantization"] = {"group_size": 32, "bits": 4, "mode": "affine"}
    (ROOT / "config-q4.json").write_text(json.dumps(qc, indent=2) + "\n")
    w = weights(qc, mx.bfloat16)
    w["lm_head.weight"] = w.pop("lm_head") * w.pop("lm_head_gain.w_g")
    packed, dense = {}, {}
    for key, value in w.items():
        if key.endswith(".weight"):
            qw, scales, biases = mx.quantize(value, group_size=32, bits=4)
            packed[key] = qw
            packed[key.removesuffix("weight") + "scales"] = scales
            packed[key.removesuffix("weight") + "biases"] = biases
            dense[key] = mx.dequantize(qw, scales, biases, group_size=32, bits=4)
        else:
            packed[key] = dense[key] = value
    save("weights-q4.safetensors", packed)
    quantized_out = expected(packed, qc)
    save("expected-q4.safetensors", quantized_out)
    manifest["q4_bf16_packed_vs_dense_max_error"] = float(mx.max(mx.abs(
        quantized_out["full"].astype(mx.float32)
        - forward(TOKENS, dense, qc).astype(mx.float32)
    )))
    full = quantized_out["full"].astype(mx.float32)
    manifest["q4_bf16_sensitivity_max_token_relative_rms"] = {}
    for change, kwargs in {
        "wrong_rope_sign": {"reverse": False},
        "missing_embedding_skip": {"embedding_skip": False},
        "missing_query_head_gain": {"head_gain": False},
    }.items():
        changed = forward(TOKENS, packed, qc, **kwargs).astype(mx.float32)
        relative_rms = mx.sqrt(mx.mean((full - changed) ** 2, axis=-1) / mx.mean(full ** 2, axis=-1))
        manifest["q4_bf16_sensitivity_max_token_relative_rms"][change] = float(mx.max(relative_rms))

    # The SAME packed integers with FP32 scales/activations give an independent,
    # tight reference for quantized loading: reconstruct affine weights through
    # explicit nibble shifts rather than using quantized matmul or dequantize.
    dense32 = {}
    for key, value in packed.items():
        if key.endswith(".weight"):
            prefix = key.removesuffix("weight")
            codes = ((value[..., None] >> mx.arange(0, 32, 4, dtype=mx.uint32)) & 15)
            codes = codes.reshape(value.shape[0], -1).astype(mx.float32)
            scale = mx.repeat(packed[prefix + "scales"].astype(mx.float32), 32, axis=-1)
            bias = mx.repeat(packed[prefix + "biases"].astype(mx.float32), 32, axis=-1)
            dense32[key] = codes * scale + bias
        elif not key.endswith((".scales", ".biases")):
            dense32[key] = value.astype(mx.float32)
    float_out = expected(dense32, qc)
    save("expected-q4-fp32.safetensors", float_out)
    packed32 = {k: v if v.dtype == mx.uint32 else v.astype(mx.float32) for k, v in packed.items()}
    manifest["q4_fp32_manual_unpack_vs_packed_max_error"] = float(mx.max(mx.abs(
        float_out["full"] - forward(TOKENS, packed32, qc)
    )))
    manifest["sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ROOT.glob("*.safetensors"))
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
