# llama.cpp patches for the diffusion-LM reference backend

Two patches against llama.cpp at `bb4caa75`. They touch different files, so each applies on
its own or together, and both do nothing unless their environment variable is set:

    git -C <llama.cpp at bb4caa75> apply llama-nonconsecutive-pos.patch
    git -C <llama.cpp at bb4caa75> apply llama-block-routing.patch

Neither has been compiled or run yet.

## llama-nonconsecutive-pos.patch (`LLAMA_ALLOW_NONCONSECUTIVE_POS=1`)

WeDLM writes rows whose positions are out of logical order: after a prefix that ends at 99 it
may decode a row at 103 and then one at 101. The driver sends one row per `llama_decode`
with `llama_set_causal_attn(ctx, false)`, so each row sees every cell the cache holds for its
sequence, and it removes rows later with `llama_memory_seq_rm`. Stock llama.cpp rejects such a
batch because its first position is not the cache's last position plus one. With the variable
set, `llama_batch_allocr::init` (`src/llama-batch.cpp`) skips that check and also the check
for gaps inside a batch. That is the only change. The rest of the decode path was audited and
needs nothing. The cache stores positions in a counted map, finds slots by emptiness alone,
and builds the non-causal mask from sequence membership only. The `seq_pos_min - n_swa - 32`
filter in `set_input_kq_mask_impl` can't hide a cell: it only decides which cells are
re-checked for later tokens of the same sequence within one ubatch, every cell is fully checked
for the first token, and a one-row ubatch never reuses a row.

## llama-block-routing.patch (`LLAMA_BLOCK_ROUTING=<block>:<pool>`, e.g. `32:48`)

LLaDA2.2-mini picks its experts per block of 32 tokens, not per token
(`LLaDA2MoeGate.block_routing` in `modeling_llada2_moe.py`). Within each block, the 48
experts with the highest biased score (taking each expert's maximum over the block's tokens)
form a pool. Each token then takes its top 8 from that pool. The weights still come from the
unbiased sigmoid scores, normalised and scaled by 2.5. With the variable set,
`build_moe_ffn` (`src/llama-graph.cpp`) masks the selection scores to the pool before its
usual top-k, using reshape, transpose, cont, `pool_1d` (max), `argsort_top_k`, `get_rows`,
`fill`, `set_rows` and a broadcast `add`. CPU and Vulkan support all of these. The mask
applies only to `bailingmoe2` with sigmoid gating and an expert-bias tensor, and only to a
ubatch whose size is a whole number of blocks. Any other ubatch falls back to normal per-token
routing, including the n_tokens = 1 graph llama.cpp reserves at startup. The driver therefore
has to submit whole blocks, starting at a block boundary, with `n_ubatch` a multiple of the
block size. When block routing applies, it also skips group-limited routing, which the
reference never uses. The patch makes one more change in `src/models/bailingmoe2.cpp`.
Normally the last layer drops the rows that need no logits before its MoE. With the variable
set, it drops them after the MoE, because the pool depends on every token in the block.

## What could not be checked without compiling

- Neither patch has been built. The new code uses only ops that already exist at this
  commit, but MSVC and gcc have not seen it.
- Block routing has never been compared numerically with the reference. Test it against a
  dump from `modeling_llada2_moe.py` for one 32-token block and for a 64-token ubatch (two
  blocks), on both CPU and Vulkan. The new graph nodes are named `ffn_moe_block_max`,
  `ffn_moe_block_pool` and `ffn_moe_probs_block`, so `examples/eval-callback` can dump them
  next to the existing `ffn_moe_topk`.
- The Vulkan path. `pool_1d` has its own Vulkan shader, but few models use it, so it is the
  op most likely to misbehave. `set_rows` writing rows that are one element wide is new here
  too; the existing group-routing code writes wider rows. If either op proves faulty, the
  scheduler can't fall back on its own, since `supports_op` says yes. To compare, run the
  same block on CPU only (`n_gpu_layers = 0`, op offload off) and fully offloaded, and diff
  `ffn_moe_topk`.
- Vulkan's fused top-k-MoE kernel (`TOPK_MOE_SIGMOID_NORM_BIAS`) should stop matching once
  the mask nodes sit between the bias add and the argsort, because it checks graph edges.
  That should make the routing correct and slightly slower; confirm by logging the fusion.
- Ties. `torch.topk` and ggml's argsort may break exact ties differently.
- A GGUF that carries `expert_group_count` changes the fallback. A model converted with the
  stock base converter would carry it, because LLaDA2.2's `config.json` has `n_group: 8`
  and `topk_group: 4`. Ragged ubatches would then use group-limited routing, which the
  reference doesn't do. The GGUF in use was converted without them, so it isn't affected.
- With non-consecutive positions, a failed or aborted decode still removes `[min pos, +inf)`
  for the failed ubatch. That can remove cells at higher positions decoded earlier.
- Non-consecutive positions on a sliding-window model (WeDLM's base models don't use one).
  The SWA slot reuse in `find_slot` measures from the sequence's highest cached position, so
  it could free a cell that a later, lower-positioned row still needs. M-RoPE models and
  recurrent or hybrid memory are untouched and keep their own checks.
