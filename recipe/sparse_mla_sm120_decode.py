# SPDX-License-Identifier: Apache-2.0
"""FlashInfer BatchSparseMLAPagedAttentionWrapper decode adapter for upstream.

Routes the upstream DSV4 SM120 decode through the fork's
``BatchSparseMLAPagedAttentionWrapper`` (grafted into flashinfer 0.6.14), which
dispatches the multi-query case (num_tokens<=64) that the raw
``trtllm_batch_decode_sparse_mla_dsv4`` call rejects with ``num_tokens > 64``.

This is the fork's actual working speculative-decode path (the fork's
DeepseekV4SM120SparseImpl calls this wrapper). Enabled with
VLLM_DSV4_SM120_WRAPPER_DECODE=1.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

_DECODE_SPLIT_TILE = 64


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _decode_num_splits(topk: int, extra_topk: int = 0) -> int:
    return _cdiv(topk, _DECODE_SPLIT_TILE) + _cdiv(extra_topk, _DECODE_SPLIT_TILE)


def use_sm120_wrapper_decode() -> bool:
    v = os.getenv("VLLM_DSV4_SM120_WRAPPER_DECODE", "0").strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _get_wrapper(layer, *, padded_heads):
    ws = getattr(layer, "_sparse_mla_sm120_wrapper", None)
    if ws is not None and getattr(layer, "_sparse_mla_sm120_wrapper_heads", 0) == padded_heads:
        return ws
    from flashinfer.sparse_mla_sm120 import BatchSparseMLAPagedAttentionWrapper

    # max_num_tokens is only a pre-allocation bound for the wrapper's out_lse;
    # decode num_tokens never exceeds max_num_batched_tokens. The vLLM config
    # context is not set during worker execution, so fall back to the env/default.
    max_num_tokens = 0
    try:
        from vllm.config import get_current_vllm_config

        max_num_tokens = int(
            get_current_vllm_config().scheduler_config.max_num_batched_tokens
        )
    except Exception:
        pass
    if max_num_tokens <= 0:
        max_num_tokens = int(os.getenv("MAX_NUM_BATCHED_TOKENS", "8192") or "8192")
    ws = BatchSparseMLAPagedAttentionWrapper(
        max_num_tokens=max_num_tokens,
        max_num_heads=int(padded_heads),
        d_v=512,
    )
    layer._sparse_mla_sm120_wrapper = ws
    layer._sparse_mla_sm120_wrapper_heads = int(padded_heads)
    logger.info_once(
        "DeepSeek V4 SM120 sparse-MLA WRAPPER decode enabled "
        "(multi-query capable; replaces raw trtllm_batch_decode_sparse_mla_dsv4)."
    )
    return ws


def sm120_wrapper_decode(
    layer,
    *,
    q: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    indexed_kv_cache: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_sparse_indices: torch.Tensor | None,
    extra_sparse_lengths: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    """Decode via the sparse-MLA SM120 wrapper. ``q`` is pre-padded to
    ``output.shape[1]`` heads by the caller (_prepare_query)."""
    num_decode_tokens = q.shape[0]
    padded_heads = output.shape[1]
    topk = int(swa_indices.shape[-1])
    # The decode_dsv4/dsv3_2 kernels only support swa topk in {128, 512, 1024}
    # (see _DECODE_DSV4_DISPATCH). The spec-verify step produces topk=256, which
    # is unsupported and falls through to the paged kernel (asserts num_tokens>64).
    # Pad the swa indices up to the next supported width with -1 (invalid slots
    # the kernel skips); swa_lens (valid counts) is unchanged so results match.
    _SUPPORTED_TOPK = (128, 512, 1024)
    if topk not in _SUPPORTED_TOPK:
        target = next((t for t in _SUPPORTED_TOPK if t >= topk), None)
        if target is not None and target > topk:
            pad = swa_indices.new_full(
                (*swa_indices.shape[:-1], target - topk), -1
            )
            swa_indices = torch.cat([swa_indices, pad], dim=-1)
            topk = target
    extra_topk = (
        int(extra_sparse_indices.shape[-1])
        if extra_sparse_indices is not None
        else 0
    )
    num_splits = _decode_num_splits(topk, extra_topk)
    mid_out, mid_lse = current_workspace_manager().get_simultaneous(
        ((num_decode_tokens, padded_heads, num_splits, output.shape[-1]), torch.bfloat16),
        ((num_decode_tokens, padded_heads, num_splits), torch.float32),
    )

    wrapper = _get_wrapper(layer, padded_heads=padded_heads)
    # b12x/flashinfer wrapper wants the raw cache with a singleton dim:
    # [num_pages, page_block_size, 1, kv_bytes_per_token].
    swa_cache = swa_kv_cache.unsqueeze(-2)
    extra_cache = (
        indexed_kv_cache.unsqueeze(-2) if indexed_kv_cache is not None else None
    )
    _key = (num_decode_tokens, padded_heads, topk, extra_topk,
            int(swa_cache.size(-3)) if swa_cache.ndim >= 3 else -1)
    _seen = getattr(sm120_wrapper_decode, "_seen_shapes", None)
    if _seen is None:
        _seen = set()
        sm120_wrapper_decode._seen_shapes = _seen
    if _key not in _seen:
        _seen.add(_key)
        logger.info(
            "SM120WRAP SHAPES: num_tokens=%d padded_heads=%d topk=%d extra_topk=%d "
            "kv_pbs=%d swa_cache=%s extra_cache=%s",
            num_decode_tokens, padded_heads, topk, extra_topk,
            _key[4], tuple(swa_cache.shape),
            (tuple(extra_cache.shape) if extra_cache is not None else None),
        )
    wrapper.run(
        q=q,
        kv_cache=swa_cache,
        indices=swa_indices,
        output=output,
        sm_scale=layer.scale,
        topk_length=swa_lens,
        attn_sink=layer.attn_sink,
        extra_kv_cache=extra_cache,
        extra_indices=extra_sparse_indices,
        extra_topk_length=extra_sparse_lengths,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )
