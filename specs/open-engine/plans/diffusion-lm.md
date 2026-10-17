# Plan: diffusion language models on the open engine

**Status:** in progress. The policies below are built and tested; the engine side is planned.

Two diffusion LMs, each a decode policy over passes of several token rows:
- **SDAR-8B-Chat-b16:** Qwen3-8B shape; 16-token blocks with in-block bidirectional
  attention.
- **LLaDA2.2-mini:** a 16B MoE; 32-token blocks, block routing, DELETE/INSERT edits.

The policies are engine-independent (`src/open_dllm/`). The llama.cpp reference driver
(`utilities/dllm-ref/`) runs them first; the open engine runs them later.

## The loop a backend runs

```
policy->start(prompt);
while (!policy->done()) {
    Pass p = policy->next();
    // run p.rows: write each row's K/V at .slot, RoPE at .pos, attention mask per p.attn
    policy->feed(logits of p.logit_rows, vocab);
    // drop every cache row at slot >= policy->keep()
    // stream policy->take()
}
```

For both policies a row's slot is >= `keep()` exactly when its position is, so a backend whose
cache is keyed by position (llama.cpp) can trim by position.

## New requirements

### OPEN-DLLM-SDAR: the SDAR decode policy reproduces generate.py's passes
**Applies to:** openflowlm-next (`src/open_dllm/policy_sdar.cpp`)
**Test category:** test (`src/open_dllm/sdar_test.cpp`; fixture `dllm_sdar.json`, recorded by
`utilities/dllm-ref/ref_sdar.py` from JetAstra/SDAR `generate.py`, vendored verbatim)

`make_sdar` issues the same passes as `generate.py`:
- **Prefill:** one block-attention pass per full prompt block, aligned to absolute
  position 0.
- **The prompt's tail** rides in the first decode block.
- **Denoise passes** are not kept: `keep()` = block start.
- **Fill:** dynamic low-confidence filling with the per-step minimum, clamped to the masks
  left.
- **Commit:** one commit pass per finished block, `keep()` = block end.
- **Stopping:** after the block holding the first generated stop id, which is taken
  inclusive.

Greedy is T = 0. That's the published OpenCompass setting, T=1 / `top_k=1` with confidence
read before the filter, not `generate.py`'s own `top_k=1`.

**Acceptance criteria:** every recorded pass matches exactly across the 10 scenarios.
These cover dynamic and static modes, aligned and unaligned prompts, the clamp, a stop
mid-block, budget hits, and an empty prompt.

### OPEN-DLLM-LLADA2: the LLaDA2.2 decode policy reproduces the reference's passes
**Applies to:** openflowlm-next (`src/open_dllm/policy_llada2.cpp`)
**Test category:** test (`src/open_dllm/llada2_test.cpp`; fixture `dllm_llada2.json`, recorded
by `utilities/dllm-ref/ref_llada2.py` running `modeling_llada2_moe.py`'s own decode methods)

`make_llada2` reproduces the reference's per-block loop:
- mask→token fills by threshold with the spread schedule as the minimum;
- token→token edits above `editing_threshold`;
- the final round that re-picks DELETE/INSERT;
- seen-state resampling;
- DELETE drops a token and INSERT opens a mask slot, within the block.

A block's K/V is kept only from a pass over its final tokens: the normal exit's no-change
pass, or one extra commit pass on any other exit.

**Acceptance criteria:** every recorded pass matches exactly across the 16 scenarios. They
cover threshold and schedule fills, T2T edits, DELETE, INSERT on a token and on a mask, the
post-step budget, the step budget with leftover masks, sampled paths with a single
possible outcome, seeded resampling, and block 32 / steps 32.

## Planned requirements (not yet built)

- **OPEN-DLLM-PASS:** the engine's multi-row pass.
- **OPEN-DLLM-COMMIT:** commit and rewind.
- **OPEN-DLLM-BLOCK-PREFILL:** block-causal prefill.
- **OPEN-DLLM-DETECT:** choosing the policy from `model_type`.
- **OPEN-MOE-BLOCK-ROUTING:** block routing in the MoE.
- **OPEN-FAMILY-LLADA2MOE:** LLaDA2.2-mini on the open kernels.
- **OPEN-DLLM-SERVE:** streaming through `oflm serve`.

Each gets a definition and a test category here before its code lands.

## Spec merge

When a requirement's code is done, its section moves into `specs/open-engine/spec.md`, and
this plan is archived once all of them have moved.
