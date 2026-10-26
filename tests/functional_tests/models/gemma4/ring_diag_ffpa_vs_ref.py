#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scratch diagnostic for the FFPA ring parity failure seen on sm80.

For every shard layout of ``run_ffpa_vs_flex_throughput_cp.py`` this reports, per rank and per
tensor (out, dQ, dK, dV), the error of the FFPA ring and of the Flex ring against an fp32
full-sequence masked-SDPA reference, plus FFPA-vs-Flex directly. It then sweeps the cp_size=1
dense causal path and the raw dense non-causal kernel over S to separate "kernel wrong at this
shape" from "ring plumbing wrong".

Run::

    torchrun --standalone --nproc-per-node=4 \
        tests/functional_tests/models/gemma4/ring_diag_ffpa_vs_ref.py

Env: ``S_LOCAL`` (default 2048), ``KINDS`` (default ``padded,multidoc,straddle``),
``SWEEP`` (default ``256,512,1024,2048,4096``; empty string skips the sweep).
"""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import flex_attention

from nemo_automodel.components.attention.ffpa_attention import _ffpa_dense_bwd, _ffpa_dense_fwd
from nemo_automodel.components.models.gemma4_moe import cp_attention as cpa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ffpa_vs_flex_throughput_cp import _ctx, _scenario_ids  # noqa: E402

ATOL, RTOL = 3e-2, 2e-2


def _err(got: torch.Tensor, ref: torch.Tensor) -> str:
    got, ref = got.float(), ref.float()
    d = (got - ref).abs()
    ok = torch.allclose(got, ref, atol=ATOL, rtol=RTOL)
    rel = (d.norm() / ref.norm().clamp_min(1e-12)).item()
    viol = (d > ATOL + RTOL * ref.abs()).float().mean().item() * 100
    return (
        f"max|d|={d.max().item():.3e} rel_l2={rel:.2e} viol={viol:.2f}% "
        f"ref_max={ref.abs().max().item():.2e} {'ok' if ok else 'FAIL'}"
    )


def _reference(q_full, k_full, v_full, ids_full, grad_full, *, scale, Hq, Hkv):
    """fp32 masked SDPA over the full sequence (same-doc & causal; pad rows attend pos 0, discarded)."""
    S_full = q_full.shape[2]
    dev = q_full.device
    qr, kr, vr = (t.float().clone().requires_grad_(True) for t in (q_full, k_full, v_full))
    rep = Hq // Hkv
    kk = kr.repeat_interleave(rep, dim=1)
    vv = vr.repeat_interleave(rep, dim=1)
    pos = torch.arange(S_full, device=dev)
    d = ids_full[0]
    allowed = (pos.view(1, -1) <= pos.view(-1, 1)) & (d.view(-1, 1) == d.view(1, -1)) & (d.view(-1, 1) > 0)
    real = d > 0
    allowed[~real, 0] = True
    mask = torch.zeros(S_full, S_full, device=dev).masked_fill(~allowed, float("-inf"))
    out = torch.nn.functional.scaled_dot_product_attention(qr, kk, vv, attn_mask=mask[None, None], scale=scale)
    g = grad_full.float().clone()
    g[:, :, ~real] = 0
    out.backward(g)
    out = out.detach()
    out[:, :, ~real] = 0
    return out, qr.grad, kr.grad, vr.grad, real


def _dense_ref(q, k, v, g, *, scale, causal):
    """fp32 reference for one dense chunk: out, natural-log LSE, dq, dk, dv."""
    Hq, Hkv = q.shape[1], k.shape[1]
    qr, kr, vr = (t.float().clone().requires_grad_(True) for t in (q, k, v))
    rep = Hq // Hkv
    s = torch.einsum("bhqd,bhkd->bhqk", qr, kr.repeat_interleave(rep, dim=1)) * scale
    if causal:
        Tq, Tk = q.shape[2], k.shape[2]
        m = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(Tk - Tq)
        s = s.masked_fill(~m, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    out = torch.softmax(s, dim=-1) @ vr.repeat_interleave(rep, dim=1)
    out.backward(g.float())
    return out.detach(), lse.detach(), qr.grad, kr.grad, vr.grad


def main() -> int:
    dist.init_process_group("nccl")
    rank, cp = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dt = torch.bfloat16
    B, Hq, Hkv, D = 1, 8, 4, 512
    S_local = int(os.environ.get("S_LOCAL", "2048"))
    S_full = S_local * cp
    scale = 1.0 / math.sqrt(256)
    s0 = rank * S_local
    if rank == 0:
        print(
            f"ring diag cp={cp} S_local={S_local} S_full={S_full} Hq={Hq} Hkv={Hkv} D={D} {dt} "
            f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability()} "
            f"ffpa_varlen={cpa._ffpa_varlen_ring_available()}",
            flush=True,
        )

    torch.manual_seed(0)
    q_full = torch.randn(B, Hq, S_full, D, device=dev, dtype=dt)
    k_full = torch.randn(B, Hkv, S_full, D, device=dev, dtype=dt)
    v_full = torch.randn(B, Hkv, S_full, D, device=dev, dtype=dt)
    torch.manual_seed(1)
    grad = torch.randn(B, Hq, S_local, D, device=dev, dtype=dt)
    grad_full = grad.repeat(1, 1, cp, 1)
    module = SimpleNamespace(
        sliding_window=None,
        config=SimpleNamespace(use_bidirectional_attention=None),
        _gemma4_cp_compiled_flex_attn=torch.compile(flex_attention, dynamic=True),
    )

    def make(ids, *, cp_size=cp, rank_=rank, S=S_local, src=(q_full, k_full, v_full)):
        start = rank_ * S
        q = src[0][:, :, start : start + S].clone().requires_grad_(True)
        k = src[1][:, :, start : start + S].clone().requires_grad_(True)
        v = src[2][:, :, start : start + S].clone().requires_grad_(True)
        ctx = _ctx(module, q, k, v, rank=rank_, cp_size=cp_size, S_local=S, scale=scale, Hq=Hq, Hkv=Hkv, ids=ids)
        return q, k, v, ctx

    kinds = [k for k in os.environ.get("KINDS", "padded,multidoc,straddle").split(",") if k]
    sl = slice(s0, s0 + S_local)
    for kind in kinds:
        ids_full = _scenario_ids(kind, S_full, S_local, cp, dev)
        ids = ids_full[:, sl].contiguous()
        out_ref, dq_ref, dk_ref, dv_ref, real = _reference(
            q_full, k_full, v_full, ids_full, grad_full, scale=scale, Hq=Hq, Hkv=Hkv
        )
        real_l = real[sl]
        res = {}
        for name, fn in (("ffpa", cpa._Gemma4FFPAVarlenRingAttention), ("flex", cpa._Gemma4FlexRingAttention)):
            q, k, v, ctx = make(ids)
            o = fn.apply(q, k, v, ctx)
            o.backward(grad)
            res[name] = (o.detach(), q.grad, k.grad, v.grad)
        for r in range(cp):
            if r == rank:
                print(f"[{kind}] rank {rank}: real_q={int(real_l.sum())}/{S_local}", flush=True)
                for name in ("ffpa", "flex"):
                    o, dq, dk, dv = res[name]
                    print(f"  {name:4s} vs ref  out {_err(o[:, :, real_l], out_ref[:, :, sl][:, :, real_l])}")
                    print(f"  {name:4s} vs ref  dq  {_err(dq[:, :, real_l], dq_ref[:, :, sl][:, :, real_l])}")
                    print(f"  {name:4s} vs ref  dk  {_err(dk, dk_ref[:, :, sl])}")
                    print(f"  {name:4s} vs ref  dv  {_err(dv, dv_ref[:, :, sl])}")
                    if (~real_l).any():
                        pad_o = o[:, :, ~real_l].abs().max().item()
                        pad_q = dq[:, :, ~real_l].abs().max().item()
                        print(f"  {name:4s} pad rows: |out|max={pad_o:.2e} |dq|max={pad_q:.2e}")
                of, dqf, dkf, dvf = res["ffpa"]
                ox, dqx, dkx, dvx = res["flex"]
                print(
                    f"  ffpa vs flex  out {_err(of, ox)}\n"
                    f"  ffpa vs flex  dq  {_err(dqf, dqx)}\n"
                    f"  ffpa vs flex  dk  {_err(dkf, dkx)}\n"
                    f"  ffpa vs flex  dv  {_err(dvf, dvx)}",
                    flush=True,
                )
            dist.barrier()
        del out_ref, dq_ref, dk_ref, dv_ref, res
        torch.cuda.empty_cache()

    sweep = [int(s) for s in os.environ.get("SWEEP", "256,512,1024,2048,4096").split(",") if s]
    if rank == 0 and sweep:
        print("--- cp_size=1 dense causal ring path (single doc) vs fp32 SDPA ---", flush=True)
        for S in sweep:
            torch.manual_seed(0)
            qs = torch.randn(B, Hq, S, D, device=dev, dtype=dt)
            ks = torch.randn(B, Hkv, S, D, device=dev, dtype=dt)
            vs = torch.randn(B, Hkv, S, D, device=dev, dtype=dt)
            g = torch.randn(B, Hq, S, D, device=dev, dtype=dt)
            doc = torch.ones(B, S, dtype=torch.long, device=dev)
            q, k, v, ctx = make(doc, cp_size=1, rank_=0, S=S, src=(qs, ks, vs))
            o = cpa._Gemma4FFPAVarlenRingAttention.apply(q, k, v, ctx)
            o.backward(g)
            o_ref, _lse_ref, dq_ref, dk_ref, dv_ref = _dense_ref(qs, ks, vs, g, scale=scale, causal=True)
            print(
                f"[cp1 causal] S={S}: out {_err(o, o_ref)} | dq {_err(q.grad, dq_ref)} | "
                f"dk {_err(k.grad, dk_ref)} | dv {_err(v.grad, dv_ref)}",
                flush=True,
            )
        print("--- raw dense kernel, causal and non-causal (cross chunk) vs fp32 math ---", flush=True)
        for S in sweep:
            torch.manual_seed(0)
            qs = torch.randn(B, Hq, S, D, device=dev, dtype=dt)
            ks = torch.randn(B, Hkv, S, D, device=dev, dtype=dt)
            vs = torch.randn(B, Hkv, S, D, device=dev, dtype=dt)
            g = torch.randn(B, Hq, S, D, device=dev, dtype=dt)
            for causal in (True, False):
                o, lse = _ffpa_dense_fwd(qs, ks, vs, scale=scale, causal=causal)
                dq, dk, dv = _ffpa_dense_bwd(g, qs, ks, vs, o, lse, scale=scale, causal=causal)
                o_ref, lse_ref, dq_ref, dk_ref, dv_ref = _dense_ref(qs, ks, vs, g, scale=scale, causal=causal)
                print(
                    f"[dense causal={causal}] S={S}: out {_err(o, o_ref)} | lse {_err(lse, lse_ref)} | "
                    f"dq {_err(dq, dq_ref)} | dk {_err(dk, dk_ref)} | dv {_err(dv, dv_ref)}",
                    flush=True,
                )
                if torch.cuda.get_device_capability()[0] == 8:
                    # Hypothesis check: the SM80 Split-D backward accumulates dK/dV through an HBM buffer in
                    # the activation dtype; its grad_kv_storage_dtype option widens that buffer to fp32.
                    from ffpa_attn.cute import _ffpa_attn_backward_cute

                    dq32, dk32, dv32 = _ffpa_attn_backward_cute(
                        g, qs, ks, vs, o, lse, scale, causal, grad_kv_storage_dtype=torch.float32
                    )
                    print(
                        f"[dense causal={causal} fp32-kv-storage] S={S}: dq {_err(dq32, dq_ref)} | "
                        f"dk {_err(dk32, dk_ref)} | dv {_err(dv32, dv_ref)}",
                        flush=True,
                    )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
