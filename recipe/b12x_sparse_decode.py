# SPDX-License-Identifier: Apache-2.0
"""B12X (TileLang) sparse-MLA decode adapter for the upstream GB10 image.

The upstream branch's DSV4 decode calls FlashInfer's SM120 sparse-MLA kernel,
which supports only single-query decode (q_len=1) and prefill (>64 tokens) --
not the multi-query verify shape (1 < q_len <= 64) that MTP speculative decode
needs. This module routes decode through the fork's b12x TileLang kernel
(``compressed_mla_decode_forward``), which DOES cover the 2..64 band and
supports fp8_ds_mla (the upstream KV dtype).

Enabled with VLLM_DSV4_B12X_COMPRESSED_MLA=1. The b12x package is copied into
the image; all inputs come from the same metadata upstream already computes in
DeepseekV4FlashInferSM120Attention._forward_decode.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

_DECODE_MAX_TOKENS = 64
_DECODE_SPLIT_TILE = 64


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _decode_num_splits(topk: int, extra_topk: int = 0) -> int:
    return _cdiv(topk, _DECODE_SPLIT_TILE) + _cdiv(extra_topk, _DECODE_SPLIT_TILE)


def use_b12x_compressed_mla() -> bool:
    v = os.getenv("VLLM_DSV4_B12X_COMPRESSED_MLA", "0").strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _b12x_index_matrix(indices: torch.Tensor | None) -> torch.Tensor | None:
    if indices is None:
        return None
    if indices.ndim == 3:
        assert indices.shape[1] == 1
        return indices.squeeze(1)
    return indices


def _get_decode_scratch(num_tokens, num_heads, d_v, topk, extra_topk):
    num_splits = _decode_num_splits(topk, extra_topk)
    return current_workspace_manager().get_simultaneous(
        ((num_tokens, num_heads, num_splits, d_v), torch.bfloat16),
        ((num_tokens, num_heads, num_splits), torch.float32),
    )


def _get_b12x_decode_workspace(layer, *, padded_heads, window_size,
                               swa_page_size, extra_topk):
    from b12x.attention.workspace import B12XAttentionWorkspace

    total_topk = int(window_size) + int(extra_topk)
    # decode is capped at 64 tokens; workspace is sized once for that bound.
    max_rows = _DECODE_MAX_TOKENS
    max_chunks = _decode_num_splits(window_size, extra_topk)

    ws = getattr(layer, "_b12x_compressed_mla_workspace", None)
    if (
        ws is None
        or int(ws.topk) < total_topk
        or int(ws.max_total_q) < max_rows
        or int(ws.max_chunks_per_row) < max_chunks
        or int(ws.num_q_heads) != int(padded_heads)
    ):
        device = layer.attn_sink.device
        if device.type != "cuda":
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        ws = B12XAttentionWorkspace(
            mode="decode",
            device=device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=int(padded_heads),
            head_dim=512,
            v_head_dim=512,
            topk=total_topk,
            max_total_q=max_rows,
            max_batch=max_rows,
            max_page_table_width=total_topk,
            max_paged_q_rows=max_rows,
            page_size=int(swa_page_size),
            padded_heads=int(padded_heads),
            max_chunks_per_row=max_chunks,
        )
        ws.kv_chunk_size_ptr = torch.empty((1,), dtype=torch.int32, device=device)
        ws.num_chunks_ptr = torch.empty((1,), dtype=torch.int32, device=device)
        layer._b12x_compressed_mla_workspace = ws
        logger.info_once(
            "DeepSeek V4 SM120 b12x compressed-MLA decode enabled "
            "(multi-query capable; replaces FlashInfer sparse decode)."
        )
    return ws


def b12x_compressed_decode(
    layer,
    *,
    q: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    indexed_kv_cache: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_sparse_indices: torch.Tensor | None,
    extra_sparse_lengths: torch.Tensor | None,
    swa_page_size: int,
    indexed_page_size: int | None,
    output: torch.Tensor,
) -> None:
    """Run the b12x SWA+compressed sparse-MLA decode into ``output``.

    ``q`` is already padded to output.shape[1] heads by the caller.
    """
    from b12x.attention.mla.compressed_api import compressed_mla_decode_forward

    num_decode_tokens = q.shape[0]
    padded_heads = output.shape[1]
    window_size = int(layer.window_size)
    extra_topk = (
        int(extra_sparse_indices.shape[-1])
        if extra_sparse_indices is not None
        else 0
    )

    mid_out, mid_lse = _get_decode_scratch(
        num_decode_tokens, padded_heads, output.shape[-1],
        window_size, extra_topk,
    )
    ws = _get_b12x_decode_workspace(
        layer, padded_heads=padded_heads, window_size=window_size,
        swa_page_size=swa_page_size, extra_topk=extra_topk,
    )
    ws.tmp_output = mid_out
    ws.tmp_lse = mid_lse
    ws.output_buffer = output

    result = compressed_mla_decode_forward(
        q_all=q,
        swa_k_cache=swa_kv_cache,
        swa_indices=_b12x_index_matrix(swa_indices),
        swa_topk_lengths=swa_lens,
        workspace=ws,
        sm_scale=layer.scale,
        swa_page_size=int(swa_page_size),
        indexed_k_cache=indexed_kv_cache,
        indexed_indices=_b12x_index_matrix(extra_sparse_indices),
        indexed_topk_lengths=extra_sparse_lengths,
        indexed_page_size=indexed_page_size,
        attn_sink=layer.attn_sink,
        expected_num_q_heads=padded_heads,
        backend="sm120_unified",
    )
    if result.data_ptr() != output.data_ptr():
        output.copy_(result)
