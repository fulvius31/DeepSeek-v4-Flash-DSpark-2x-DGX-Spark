sparse_mla_sm120_decode_dsv4_trace = TraceTemplate(
    op_type="sparse_mla_decode_dsv4_sm120",
    name_prefix="sparse_mla_sm120_decode_dsv4",
    description=(
        "Sparse-MLA paged decode (DSv4 standalone kernel) on SM120 with "
        "optional dual-cache support and AutoTuner-driven chunks_per_block "
        "tactic selection."
    ),
    axes={
        "num_tokens": Var(description="Number of query tokens."),
        "num_heads": Const(
            description="Number of query heads after TP split.", abbrev="h"
        ),
        "head_dim_qk": Const(
            description="Query head dim. 512 for the DSv4 family.",
            abbrev="dqk",
        ),
        "head_dim_v": Const(description="Value head dim. 512.", abbrev="dv"),
        "topk": Const(
            description="Number of top-K paged slots per query token "
            "({128, 512, 1024}).",
            abbrev="topk",
        ),
        "page_block_size": Const(
            description="KV cache page block size (64).", abbrev="ps"
        ),
        "num_pages": Var(description="Total allocated pages in the KV cache."),
        "extra_num_pages": Var(
            description="Pages in the optional secondary KV cache (dual-cache mode)."
        ),
        "extra_topk": Const(
            description="Top-K width for the secondary cache.",
            abbrev="xtopk",
        ),
        "extra_page_block_size": Const(
            description="Page block size of the secondary cache.",
            abbrev="xps",
        ),
        "kv_bytes_per_token": Const(
            description="Byte-packed FP8 FOOTER stride (584 = 448 nope + 128 rope + 8 scales).",
            abbrev="kvb",
        ),
        "num_splits": Var(
            description="ceil(topk / 64) + ceil(extra_topk / 64); mid_out splits.",
        ),
    },
    inputs={
        "q": Tensor(
            ["num_tokens", "num_heads", "head_dim_qk"],
            description="Query tensor, dtype bf16.",
        ),
        "kv_cache": Tensor(
            ["num_pages", "page_block_size", "1", "kv_bytes_per_token"],
            dtype="uint8",
            description="Paged main KV cache, FP8 FOOTER layout.",
        ),
        "indices": Tensor(
            ["num_tokens", "topk"],
            dtype="int32",
            description="Paged slot IDs. -1 marks invalid / out-of-window.",
        ),
        "mid_out": Tensor(
            ["num_tokens", "num_heads", "num_splits", "head_dim_v"],
            dtype_from="q",
            description="Per-split partial outputs (bf16 scratch).",
        ),
        "mid_lse": Tensor(
            ["num_tokens", "num_heads", "num_splits"],
            dtype="float32",
            description="Per-split LSE (float32 scratch).",
        ),
        "output": Tensor(
            ["num_tokens", "num_heads", "head_dim_v"],
            dtype_from="q",
            description="In-place output buffer.",
        ),
        "out_lse": Tensor(
            ["num_tokens", "num_heads"],
            dtype="float32",
            description="In-place log-sum-exp (2-based; merges attn_sink when present).",
        ),
        "sm_scale": Scalar(
            "float32", description="Softmax scale, typically 1/sqrt(head_dim_qk)."
        ),
        "topk_length": Tensor(
            ["num_tokens"],
            dtype="int32",
            optional=True,
            description="Effective top-k length per query token.",
        ),
        "extra_kv_cache": Tensor(
            [
                "extra_num_pages",
                "extra_page_block_size",
                "1",
                "kv_bytes_per_token",
            ],
            dtype="uint8",
            optional=True,
            description="Optional secondary KV cache, FP8 FOOTER layout.",
        ),
        "extra_indices": Tensor(
            ["num_tokens", "extra_topk"],
            dtype="int32",
            optional=True,
            description="Paged slot IDs for the secondary cache.",
        ),
        "extra_topk_length": Tensor(
            ["num_tokens"],
            dtype="int32",
            optional=True,
            description="Effective top-k length per query token for the secondary cache.",
        ),
        "attn_sink": Tensor(
            ["num_heads"],
            dtype="float32",
            optional=True,
            description=(
                "Per-head learnable bias added pre-softmax. FlashMLA V4 convention: "
                "output *= sigmoid(lse - sink), lse' = log(exp(lse) + exp(sink))."
            ),
        ),
    },
    outputs={
        "output": Tensor(
            ["num_tokens", "num_heads", "head_dim_v"],
            dtype_from="q",
            description="Attention output (also mutated in place above).",
        ),
        "out_lse": Tensor(
            ["num_tokens", "num_heads"],
            dtype="float32",
            description="The 2-based log-sum-exp of attention logits (sink-merged).",
        ),
    },
    constraints=[
        "indices.shape[0] == num_tokens",
        "indices.shape[-1] == topk",
        "kv_cache.shape[1] == page_block_size",
        "head_dim_qk == 512",
        "head_dim_v == 512",
        "topk in (128, 512, 1024)",
        "extra_indices.shape[0] == num_tokens",
        "extra_indices.shape[-1] == extra_topk",
        "extra_kv_cache.shape[1] == extra_page_block_size",
    ],
    tags=[
        "status:wip",
        "sparse:topk",
        "backend:sm120",
        "model:dsv4",
    ],
)
