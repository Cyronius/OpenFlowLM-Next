#!/usr/bin/env python3
"""hf_check.py: replay dllm-ref's passes through each model's original HF modeling code on CPU
and compare the logits.

    python hf_check.py --model {wedlm,sdar,llada2} --hf <dir> --trace <trace.json> [more ...]
                       [--passes N] [--dtype bf16|fp32] [--attn sdpa|eager] [--dry-run]

WHAT IT CHECKS
    dllm-ref (this directory: llama.cpp driven by the src/open_dllm policies) writes one JSON
    trace per prompt with --trace. For the first N passes of a trace (default: the prefill
    passes plus the first 4 decode passes) this script rebuilds what each pass computed --
    every row's token, RoPE position and the set of rows it could see, including the rows it
    read from the KV cache -- runs that through the HF original in one forward without a KV
    cache, and compares each logit row with the top-8 the driver recorded:

      top-1     the HF argmax equals the driver's top-1. A disagreement is a "near-tie" when the
                two candidates' logit difference moved by <= --tie-eps between the runs, else
                "hard";
      ovl       |HF top-k  intersect  driver top-k| (k = 8);
      r         Pearson correlation of HF and driver logits over the driver's top-8 ids;
      max|d|    the largest absolute logit difference over those ids.

    It prints a per-pass table, the disagreements and an overall verdict, and exits 0 whatever
    it finds: it reports, it does not assert. Tokens come from the trace (teacher forcing), so
    a driver bug shows up at the pass where it starts instead of compounding. Passes without
    logit rows (prefill, commit) are not compared themselves; their K/V is read by every later
    pass, so an error there shows up in the next compared pass. What is under test is the
    driver: slots, positions, masks, cache trimming, LLaDA2's routing patch, the GGUF
    conversion and the quant -- not the policies.

TRACE FORMAT
    What dllm-ref is expected to write (the driver was written in parallel; the parser is
    lenient, and its key lists sit at the top of the TRACE PARSING section if a name differs):

      {"model": "...", "passes": [
         {"rows": [[token, pos, slot], ...], "attn": "causal" | "block",
          "logit_rows": [i, ...],                 # indices into rows
          "top8": [[[id, logit], ...8], ...],     # one list per logit row, same order
          "keep": n, "taken": [...]},
         ...],
       "output": [...]}

    Also accepted: rows as {"token","pos","slot"}; the top-k under top8/top/topk/top_k/tops/
    top_logits/logits, as [id, logit] pairs or {"id","logit"} objects; logit_rows as
    [{"row": i, "top8": [...]}, ...]; the pass list under passes/steps/trace; several traces
    in one file (a top-level list, or under traces/prompts/requests). A directory argument
    means every *.json in it. Several traces share one model load.

WHY ONE FORWARD WITHOUT A KV CACHE IS THE SAME COMPUTATION
    policy.hpp's pass: row i's K/V is written at cache row `slot` with RoPE at `pos`; a row
    sees the kept cache plus the rows of the pass its mode allows (causal: rows 0..j in
    physical order; block: all of them); then every cache row with slot >= keep is dropped.
    In these decoder-only models a row's hidden state at layer l depends only on its own state
    at layer l-1 and on the layer-l K/V of the rows it sees, and a row's K/V depends only on
    its own state and RoPE position. A cached row's K/V stays what its own pass computed.

    So the script tracks every row ever written as an "instance" (token, position, the set of
    instances it saw) while it walks the trace. For a compared pass it takes the instances the
    pass can reach -- the pass's rows, what they saw, what those saw, ... -- lays them out in
    creation order (the cache in slot order, then the pass in physical order), feeds their
    tokens with explicit position_ids, and gives every instance exactly its own visible set as
    an explicit 4D mask. By induction over layers, and then over passes, that one forward
    computes every instance's states exactly as its own pass did, so the pass's logit rows
    are the same function of the same inputs. The only cost is recomputing the context every
    time. Every op other than attention is row-local (embedding, RMSNorm, projections, MLP,
    LM head) -- except LLaDA2's block routing, below.

    For the three policies the reachable set is the kept cache plus the pass:
      WeDLM   the kept rows of a pass are its leading filled rows, which saw only kept rows,
              and their pos == slot. The mask is plain causal over [committed tokens at
              positions 0..ctx-1] + [the pass in physical order: filled rows, then masks], with
              non-monotonic position_ids -- WeDLM's own reordering (modeling_wedlm.py:664-682).
      SDAR,   whole block passes are kept (prefill blocks, then each block's commit pass), so
      LLaDA2  the mask is block-causal over [prefix blocks] + [current block].
    If a trace ever kept a row that had seen a since-dropped row, that dropped row is replayed
    as an extra "ghost" instance visible only to the rows that saw it; the table counts them
    (expected 0). The "mask" column says whether the rebuilt mask is plain causal or
    block-causal, as a cross-check of the trace against the policy.

    LLaDA2's block routing. LLaDA2MoeGate (modeling_llada2_moe.py:277-298) scores each token
    alone (fp32 linear + sigmoid + expert_bias), then block_routing (:244-275) views the flat
    token list as [n/32, 32, 256] (asserting 32 divides it), takes each block's max over its 32
    tokens per expert, keeps the top expert_capacity=48 experts per block, and lets each token
    take its top-8 among those. A block's pool therefore depends only on the hidden states of
    that block's own 32 tokens. dllm-ref runs every LLaDA2 pass as exactly one aligned 32-row
    block (policy_llada2.cpp) and the patched bailingmoe2 graph pools per 32 rows of the
    batch. The replay lays every reachable pass out as one contiguous 32-row group (prefix
    blocks in order, then the current block), which is also what the reference itself does:
    generate() forwards x[:, :block_end] with position_ids arange (:1505-1510, :1731-1743).
    A LLaDA2 pass whose reachable rows would not split into whole 32-row passes is refused.

HOW EACH MODEL IS FED  (transformers 5.14.1, torch 2.13 CPU; base model + LM head, no cache)
    WeDLM   model.model(input_ids, position_ids, attention_mask={"full_attention": M}).
            M is additive (0 / finfo.min) in the model dtype, which both attention paths take
            as is: transformers' sdpa passes it to F.scaled_dot_product_attention, eager adds
            it (modeling_wedlm.py:230-232). The dict skips the create_causal_mask call at
            :498-509, which passes input_embeds= and cache_position= -- keywords 5.14's
            create_causal_mask (masking_utils.py:871-881) does not take, so WeDLM's own mask
            path raises TypeError under 5.14.
    SDAR    model.model(input_ids, position_ids, attention_mask=M, store_kv=False). SDAR's
            mask creation is commented out (modeling_sdar.py:517-519) and the attention
            .bool()s whatever it gets (:275), so M is BOOLEAN, True = attend (an additive 0/-inf
            mask would be inverted). A block-causal M takes SDAR's own SDPA branch (:288-298);
            an all-True M (one block, nothing before it) takes the flash_attn_func branch
            (:276-286), which runs on the SDPA stand-in below.
    LLaDA2  model.model(input_ids, position_ids, attention_mask=M). create_bidirectional_mask
            (:887-891) returns a 4D mask unchanged (masking_utils.py:817-819). M is additive,
            the same form generate() builds (:1719-1729: log of a 0/1 block mask -> 0/-inf).
    The LM head is then applied to the logit rows only, in fp32 on the model's final hidden
    states. The HF code applies it in the model dtype; in bf16 that rounds logits to steps of
    0.06-0.125 around 10-20 and turns near-ties into exact ties. --native-head restores it.

CPU SWAPS AND 5.x FIXES  (monkeypatches; the downloaded files are not edited)
    SDAR (modeling_sdar.py)
      :49      `from flash_attn.ops.triton.layer_norm import rms_norm_fn` is unguarded and
               flash-attn is not installed (its Triton kernels don't run on CPU anyway). A stub
               flash_attn package sits in sys.modules for the import only (transformers'
               check_imports needs it too); its functions raise if ever called.
      :83-93   SDARRMSNorm.forward calls that Triton RMSNorm -> replaced by the eager RMSNorm
               commented out at :87-92, verbatim.
      :53,:280 flash_attn_func (imported in a try at :52-56) -> an SDPA wrapper: same
               (batch, seq, heads, dim) layout, GQA, non-causal, softmax_scale = scale.
      :58-62   liger_kernel's Triton SwiGLU forced off (it isn't installed here either).
      :46,:736 imports and subclasses transformers.utils.LossKwargs, gone in 5.x -> TypedDict.
      :411     ROPE_INIT_FUNCTIONS[self.rope_type] with rope_type "default", a key 5.x dropped
               (modeling_rope_utils.py:632-639) -> registered (inv_freq = theta^(-2i/d)).
      :384-395 _init_weights has no rotary branch. 5.x builds the model on the meta device and
               re-creates non-persistent buffers with torch.empty_like (modeling_utils.py:3899,
               :4876-4880), counting on _init_weights to fill them: SDAR's inv_freq would be
               uninitialised memory. -> _init_weights fills rotary buffers from config.json.
    WeDLM (modeling_wedlm.py)
      :122-145 names its RoPE helper _compute_default_rope_parameters, but 5.x's generic
               _init_weights (modeling_utils.py:2562-2570), which WeDLM inherits, calls
               compute_default_rope_parameters on every *RotaryEmbedding while loading ->
               AttributeError. -> the same _init_weights patch as SDAR.
      :498-509 see above (dict mask). No flash-attn dependency.
    LLaDA2 (modeling_llada2_moe.py)
      Written for 5.2; nothing in it needs patching as far as the code reads (its
      _init_weights calls super(), whose rotary branch matches its own
      compute_default_rope_parameters). MoE runs moe_infer (:357-387), plain torch.
    All three: after loading, every rotary inv_freq is checked against
    1/theta^(arange(0, d, 2)/d) from config.json (d = head_dim * partial_rotary_factor) and
    repaired if wrong; missing or mismatched checkpoint tensors, or anything left on the meta
    device, abort the run. If importing the modeling file fails, it is retried once with
    transformers' auto_docstring (a docstring generator) disabled.

WHAT STAYS APPROXIMATE
    - bf16 (the default) rounds activations, K/V and RoPE cos/sin to bf16 at every layer; the
      driver computes in f32 with Q8_0/Q4_1 weights. Expect logit differences of ~0.1 and
      top-1 flips at near-ties. --dtype fp32 removes the HF side's rounding and is the cleaner
      reference for WeDLM and SDAR (33 GB); for LLaDA2 it needs ~65 GB of the 88.
    - LLaDA2's routing is discrete: an expert near the edge of a block's top-48 pool, or of a
      token's top-8, can flip between HF and the driver and give one row (or one block) a
      larger deviation. The router itself runs in fp32 in both, but in bf16 mode its input
      is bf16-rounded. Isolated hard flips on LLaDA2 at Q8_0 are not by themselves a bug.
    - SDAR: the eager RMSNorm rounds the normalised x to bf16 before the weight multiply; the
      fused flash-attn kernel the reference runs on GPU rounds once. Identical in fp32.
      SDPA instead of flash_attn_func: same math, different summation order.
    - Nothing in the modeling code prevents an exact replay of these traces: the cache-free
      forward is the same computation (above), and LLaDA2's routing groups are reproduced
      exactly. The differences are rounding and quantization only.

MEMORY AND TIME (it will be slow)
    RAM: WeDLM/SDAR ~16.4 GB bf16, ~33 GB fp32; LLaDA2 ~32.5 GB bf16, ~65 GB fp32. In bf16 the
    fp32 LM head adds 2.5 GB (1.3 GB for LLaDA2). Activations are small below ~2K tokens; on
    torch's math SDPA path attention may hold [heads, L, L] floats per layer (0.5 GB at L=2K).
    Load: reading 16-33 GB, from tens of seconds to a few minutes.
    Each compared pass is one forward over L = context + pass rows. Dense 8B is ~16 GFLOP per
    token, so L=500 is ~8 TFLOP: roughly 10-60 s on this CPU in bf16, about twice that in fp32.
    LLaDA2 has fewer FLOPs (1.4B active) but moe_infer loops in Python over up to 256 experts in
    each of 19 layers: roughly 10-30 s. The default (prefill + 4 decode passes) is ~4 forwards
    per trace: a few minutes per trace on top of the load. Use --dry-run first: it parses the
    traces and prints every pass's replay layout without importing torch.
    The run keeps every core busy: don't overlap it with an NPU benchmark (a busy CPU throttles
    the NPU).

Plan: .claude/plans/diffusion-lm-support.md, Phase 1, "Checking the driver itself".
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

# Where the downloaded models live: $DLLM_ROOT (default ~/dllm), as in eval/common.py.
DLLM_ROOT = Path(os.environ.get("DLLM_ROOT") or Path.home() / "dllm")
DEFAULT_HF = {
    "wedlm": DLLM_ROOT / "hf" / "WeDLM-8B-Instruct",
    "sdar": DLLM_ROOT / "hf" / "SDAR-8B-Chat-b16",
    "llada2": DLLM_ROOT / "hf" / "LLaDA2.2-mini",
}
MODEL_TYPE = {"wedlm": "wedlm", "sdar": "sdar", "llada2": "llada2_moe"}
EXPECTED_ATTN = {"wedlm": "causal", "sdar": "block", "llada2": "block"}


# =============================================================================================
# TRACE PARSING
# =============================================================================================

PASS_LIST_KEYS = ("passes", "steps", "trace")
MULTI_TRACE_KEYS = ("traces", "prompts", "requests")
TOP_KEYS = ("top8", "top", "topk", "top_k", "tops", "top_logits", "logits")
ROW_TOKEN_KEYS = ("token", "tok", "id", "t")
ROW_POS_KEYS = ("pos", "position", "p")
ROW_SLOT_KEYS = ("slot", "cache_slot", "s")
PAIR_ID_KEYS = ("id", "token", "tok", "t")
PAIR_LOGIT_KEYS = ("logit", "value", "v", "l", "score")
ATTN_KEYS = ("attn", "mode", "attention")
PHASE_KEYS = ("phase", "kind", "stage")
META_KEYS = ("model", "gguf", "quant", "policy", "name", "id", "prompt_id", "settings", "seed")
OUTPUT_KEYS = ("output", "generated", "tokens_out")


class TraceError(Exception):
    pass


@dataclass
class Row:
    token: int
    pos: int
    slot: int


@dataclass
class TPass:
    index: int
    rows: list[Row]
    attn: str                                      # "causal" | "block"
    logit_rows: list[int]
    tops: list[list[tuple[int, float]]] | None     # per logit row, sorted by logit, descending
    keep: int
    phase: str | None


@dataclass
class Trace:
    path: Path
    label: str
    meta: dict
    passes: list[TPass]
    output: list | None


def _first(d: dict, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _as_int(x, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or int(x) != x:
        raise TraceError(f"{what}: expected an integer, got {x!r}")
    return int(x)


def _parse_row(r, where: str) -> Row:
    if isinstance(r, dict):
        tok, pos, slot = _first(r, ROW_TOKEN_KEYS), _first(r, ROW_POS_KEYS), _first(r, ROW_SLOT_KEYS)
    elif isinstance(r, (list, tuple)) and len(r) >= 3:
        tok, pos, slot = r[0], r[1], r[2]
    else:
        raise TraceError(f"{where}: a row must be [token, pos, slot] or an object, got {r!r}")
    return Row(_as_int(tok, where + " token"), _as_int(pos, where + " pos"), _as_int(slot, where + " slot"))


def _parse_pair(e, where: str) -> tuple[int, float]:
    if isinstance(e, dict):
        i, v = _first(e, PAIR_ID_KEYS), _first(e, PAIR_LOGIT_KEYS)
    elif isinstance(e, (list, tuple)) and len(e) >= 2:
        i, v = e[0], e[1]
    else:
        raise TraceError(f"{where}: a top-k entry must be [id, logit] or an object, got {e!r}")
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TraceError(f"{where}: logit {v!r} is not a number")
    return _as_int(i, where + " id"), float(v)


def _parse_top(e, where: str) -> tuple[list[tuple[int, float]], int | None]:
    """One logit row's top-k: a list of pairs, or an object holding one (and maybe its row)."""
    row = None
    if isinstance(e, dict):
        row = e.get("row")
        lst = _first(e, TOP_KEYS)
        if lst is None:
            raise TraceError(f"{where}: no top-k list (keys tried: {', '.join(TOP_KEYS)})")
    else:
        lst = e
    if not isinstance(lst, list) or not lst:
        raise TraceError(f"{where}: empty or malformed top-k list")
    pairs = [_parse_pair(x, where) for x in lst]
    pairs.sort(key=lambda t: -t[1])  # stable: equal logits keep the driver's order
    return pairs, (None if row is None else _as_int(row, where + " row"))


def _parse_attn(p: dict, where: str) -> str:
    v = _first(p, ATTN_KEYS)
    if isinstance(v, str):
        s = v.lower()
        if "causal" in s:
            return "causal"
        if "block" in s or "bidir" in s or "full" in s:
            return "block"
    if isinstance(v, int) and not isinstance(v, bool) and v in (0, 1):
        return "causal" if v == 0 else "block"   # policy.hpp's enum order: Causal, Block
    if v is None and isinstance(p.get("causal"), bool):
        return "causal" if p["causal"] else "block"
    raise TraceError(f"{where}: attention mode {v!r} is neither causal nor block")


def _parse_pass(i: int, p, where: str) -> TPass:
    if not isinstance(p, dict):
        raise TraceError(f"{where}: a pass must be an object")
    rows_raw = p.get("rows")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise TraceError(f"{where}: no rows")
    rows = [_parse_row(r, f"{where} row {j}") for j, r in enumerate(rows_raw)]
    attn = _parse_attn(p, where)
    keep = p.get("keep")
    if keep is None:
        raise TraceError(f"{where}: no 'keep' (the cache trim after the pass is needed to replay)")
    keep = _as_int(keep, where + " keep")

    lr_raw = p.get("logit_rows") or []
    if not isinstance(lr_raw, list):
        raise TraceError(f"{where}: logit_rows must be a list")
    logit_rows: list[int] = []
    tops: list[list[tuple[int, float]]] | None = None
    if lr_raw and isinstance(lr_raw[0], dict):           # [{"row": i, "top8": [...]}, ...]
        tops = []
        for k, e in enumerate(lr_raw):
            pairs, row = _parse_top(e, f"{where} logit row {k}")
            if row is None:
                raise TraceError(f"{where} logit row {k}: object without 'row'")
            logit_rows.append(row)
            tops.append(pairs)
    else:
        logit_rows = [_as_int(x, where + " logit_rows") for x in lr_raw]
        raw = None
        for key in TOP_KEYS:
            if isinstance(p.get(key), list):
                raw = p[key]
                break
        if raw is not None and logit_rows:
            if len(raw) != len(logit_rows):
                raise TraceError(f"{where}: {len(logit_rows)} logit rows but {len(raw)} top-k lists")
            tops = []
            for k, e in enumerate(raw):
                pairs, row = _parse_top(e, f"{where} logit row {k}")
                if row is not None and row != logit_rows[k]:
                    raise TraceError(f"{where} logit row {k}: top-k says row {row}, logit_rows says {logit_rows[k]}")
                tops.append(pairs)
    for r in logit_rows:
        if not 0 <= r < len(rows):
            raise TraceError(f"{where}: logit row {r} outside the pass's {len(rows)} rows")
    if len(set(logit_rows)) != len(logit_rows):
        raise TraceError(f"{where}: repeated logit row")
    phase = _first(p, PHASE_KEYS)
    return TPass(i, rows, attn, logit_rows, tops, keep, phase if isinstance(phase, str) else None)


def _passes_of(d) -> list | None:
    if isinstance(d, dict):
        for key in PASS_LIST_KEYS:
            if isinstance(d.get(key), list):
                return d[key]
    return None


def _split_docs(doc, f: Path) -> list[tuple[str, dict]]:
    if isinstance(doc, dict):
        for key in MULTI_TRACE_KEYS:
            v = doc.get(key)
            if isinstance(v, list) and v and _passes_of(v[0]) is not None:
                return [(f"{f.stem}#{i}", d) for i, d in enumerate(v)]
        return [(f.stem, doc)]
    if isinstance(doc, list):
        if doc and _passes_of(doc[0]) is not None:
            return [(f"{f.stem}#{i}", d) for i, d in enumerate(doc)]
        return [(f.stem, {"passes": doc})]
    raise TraceError("the top level is neither an object nor a list")


def _parse_trace(d: dict, f: Path, label: str) -> Trace:
    raw = _passes_of(d)
    if raw is None:
        raise TraceError(f"no pass list (keys tried: {', '.join(PASS_LIST_KEYS)})")
    passes = [_parse_pass(i, p, f"pass {i}") for i, p in enumerate(raw)]
    meta = {k: d[k] for k in META_KEYS if k in d and not isinstance(d[k], (list, dict))}
    name = _first(d, ("name", "id", "prompt_id"))
    if isinstance(name, (str, int)) and str(name) not in label:
        label = f"{label} ({name})"
    out = _first(d, OUTPUT_KEYS)
    return Trace(f, label, meta, passes, out if isinstance(out, list) else None)


def load_traces(paths: list[Path]) -> list[Trace]:
    files: list[Path] = []
    for p in paths:
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    traces: list[Trace] = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                doc = json.load(fh)
            for label, d in _split_docs(doc, f):
                traces.append(_parse_trace(d, f, label))
        except (OSError, json.JSONDecodeError, TraceError) as e:
            print(f"[skip] {f}: {e}")
    return traces


def default_pass_count(passes: list[TPass]) -> tuple[int, int]:
    """(prefill passes, default N). Prefill = passes flagged so, else the leading passes
    that ask for no logits."""
    n_pre = 0
    flagged = any(p.phase for p in passes)
    for p in passes:
        if flagged:
            if p.phase and "prefill" in p.phase.lower():
                n_pre += 1
                continue
        elif not p.logit_rows:
            n_pre += 1
            continue
        break
    return n_pre, n_pre + 4


# =============================================================================================
# REPLAY: which instances a pass reaches, in what order, under what mask
# =============================================================================================

@dataclass
class Inst:
    token: int
    pos: int
    slot: int
    pass_idx: int


@dataclass
class Plan:
    tokens: list[int]
    positions: list[int]
    mask: np.ndarray            # [L, L] bool, mask[a, b]: layout row a sees layout row b
    logit_index: list[int]      # layout index of each logit row, in logit_rows order
    n_ctx: int                  # cache rows visible to the pass
    ghosts: int                 # reachable instances that are neither in that cache nor the pass
    structure: str              # "causal", "block-causal/B" or "custom"


class Replay:
    def __init__(self, routing_block: int | None):
        self.routing_block = routing_block   # LLaDA2: each layout group of this many rows routes alone
        self.insts: list[Inst] = []
        self.pass_base: list[int] = []       # first instance id of each pass
        self.pass_len: list[int] = []
        self.pass_attn: list[str] = []
        self.pass_ctx: list[tuple[int, ...]] = []   # cache instances every row of the pass sees
        self.cache: dict[int, int] = {}       # slot -> instance id
        self.warnings: list[str] = []

    def own(self, i: int) -> range:
        """The rows of its own pass that instance i saw (itself included)."""
        q = self.insts[i].pass_idx
        b = self.pass_base[q]
        if self.pass_attn[q] == "causal":
            return range(b, i + 1)
        return range(b, b + self.pass_len[q])

    def add(self, p: TPass) -> int:
        q = len(self.pass_base)
        slots = [r.slot for r in p.rows]
        if len(set(slots)) != len(slots):
            raise TraceError(f"pass {p.index}: two rows write the same cache slot")
        written = set(slots)
        over = sorted(s for s in written if s in self.cache)
        if over:
            self.warnings.append(f"pass {p.index} rewrites {len(over)} kept cache slot(s) (first {over[0]}); "
                                 "policy.hpp never does -- the replay lets the new rows replace them")
        ctx = tuple(self.cache[s] for s in sorted(self.cache) if s not in written)
        base = len(self.insts)
        self.insts += [Inst(r.token, r.pos, r.slot, q) for r in p.rows]
        self.pass_base.append(base)
        self.pass_len.append(len(p.rows))
        self.pass_attn.append(p.attn)
        self.pass_ctx.append(ctx)
        return q

    def commit(self, q: int, keep: int) -> None:
        b = self.pass_base[q]
        for i in range(b, b + self.pass_len[q]):
            self.cache[self.insts[i].slot] = i
        for s in [s for s in self.cache if s >= keep]:
            del self.cache[s]

    def closure(self, q: int) -> list[int]:
        b, n = self.pass_base[q], self.pass_len[q]
        need = set(range(b, b + n))
        stack = list(need)
        ctx_done: set[int] = set()
        while stack:
            i = stack.pop()
            for k in self.own(i):
                if k not in need:
                    need.add(k)
                    stack.append(k)
            qq = self.insts[i].pass_idx
            if qq not in ctx_done:
                ctx_done.add(qq)
                for k in self.pass_ctx[qq]:
                    if k not in need:
                        need.add(k)
                        stack.append(k)
        return sorted(need)

    def plan(self, q: int, p: TPass) -> Plan:
        order = self.closure(q)
        L = len(order)
        index = {i: a for a, i in enumerate(order)}
        mask = np.zeros((L, L), dtype=bool)
        ctx_cols: dict[int, np.ndarray] = {}
        for a, i in enumerate(order):
            qq = self.insts[i].pass_idx
            cols = ctx_cols.get(qq)
            if cols is None:
                cols = np.fromiter((index[k] for k in self.pass_ctx[qq]), dtype=np.int64,
                                   count=len(self.pass_ctx[qq]))
                ctx_cols[qq] = cols
            if cols.size:
                mask[a, cols] = True
            own = self.own(i)
            # own is a run of consecutive instance ids, all reachable, so consecutive in the layout
            mask[a, index[own.start]:index[own.stop - 1] + 1] = True

        if self.routing_block:
            B = self.routing_block
            a = 0
            while a < L:
                qq = self.insts[order[a]].pass_idx
                e = a
                while e < L and self.insts[order[e]].pass_idx == qq:
                    e += 1
                if e - a != B or self.pass_len[qq] != B:
                    raise TraceError(
                        f"pass {p.index}: the forward would hold {e - a} row(s) of trace pass {qq} "
                        f"(a pass of {self.pass_len[qq]} rows) as one group; LLaDA2's block routing "
                        f"pools experts per {B} consecutive tokens, so the replay is only exact when every "
                        f"reachable pass is a whole {B}-row pass")
                a = e

        cur = set(range(self.pass_base[q], self.pass_base[q] + self.pass_len[q]))
        ghosts = len(set(order) - set(self.pass_ctx[q]) - cur)
        structure = "custom"
        if np.array_equal(mask, np.tril(np.ones((L, L), dtype=bool))):
            structure = "causal"
        elif p.attn == "block" and L % len(p.rows) == 0:
            blk = np.arange(L) // len(p.rows)
            if np.array_equal(mask, blk[:, None] >= blk[None, :]):
                structure = f"block-causal/{len(p.rows)}"
        return Plan(
            tokens=[self.insts[i].token for i in order],
            positions=[self.insts[i].pos for i in order],
            mask=mask,
            logit_index=[index[self.pass_base[q] + r] for r in p.logit_rows],
            n_ctx=len(self.pass_ctx[q]),
            ghosts=ghosts,
            structure=structure,
        )


# =============================================================================================
# MODEL LOADING: shims, CPU swaps, checks
# =============================================================================================

def read_config(hf: Path) -> dict:
    with open(hf / "config.json", encoding="utf-8") as fh:
        return json.load(fh)


def _unavailable(name: str):
    def f(*args, **kwargs):
        raise RuntimeError(f"{name} was called: hf_check.py's CPU replacement for it did not take")
    f.__name__ = name.rsplit(".", 1)[-1]
    return f


def _flash_attn_func_sdpa(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    """flash_attn_func on torch SDPA. flash-attn layout: q (B, Sq, Hq, D), k/v (B, Sk, Hkv, D),
    Hkv dividing Hq; returns (B, Sq, Hq, D). SDAR calls it with causal=False only (:280-286)."""
    import torch.nn.functional as F
    extra = {key: val for key, val in kwargs.items() if val not in (None, False, 0, 0.0, (-1, -1))}
    if dropout_p or extra:
        raise NotImplementedError(f"flash_attn_func stand-in: unsupported arguments {extra or {'dropout_p': dropout_p}}")
    if causal and q.shape[1] != k.shape[1]:
        raise NotImplementedError("flash_attn_func stand-in: causal with Sq != Sk (flash aligns bottom-right)")
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal, scale=softmax_scale,
                                         enable_gqa=qt.shape[1] != kt.shape[1])
    return out.transpose(1, 2)


def _install_flash_attn_stub(notes: list[str]) -> list[str]:
    """A stub flash_attn package, in sys.modules only while modeling_sdar.py is imported."""
    if "flash_attn" in sys.modules or importlib.util.find_spec("flash_attn") is not None:
        notes.append("flash_attn is importable here; SDAR's uses of it are still replaced")
        return []
    names = ["flash_attn", "flash_attn.ops", "flash_attn.ops.triton", "flash_attn.ops.triton.layer_norm",
             "flash_attn.bert_padding"]
    mods = {}
    for n in names:
        m = types.ModuleType(n)
        m.__spec__ = importlib.machinery.ModuleSpec(n, loader=None, is_package=True)
        m.__path__ = []
        mods[n] = m
    for n in names:
        parent, _, child = n.rpartition(".")
        if parent:
            setattr(mods[parent], child, mods[n])
    mods["flash_attn.ops.triton.layer_norm"].rms_norm_fn = _unavailable("flash_attn.ops.triton.layer_norm.rms_norm_fn")
    mods["flash_attn"].flash_attn_func = _flash_attn_func_sdpa
    mods["flash_attn"].flash_attn_varlen_func = _unavailable("flash_attn.flash_attn_varlen_func")
    for fn in ("index_first_axis", "pad_input", "unpad_input"):
        setattr(mods["flash_attn.bert_padding"], fn, _unavailable("flash_attn.bert_padding." + fn))
    sys.modules.update(mods)
    notes.append("SDAR: stub flash_attn package for the import (modeling_sdar.py:49, :53-54)")
    return names


def _shim_loss_kwargs(notes: list[str]) -> None:
    import transformers.utils as tu
    if hasattr(tu, "LossKwargs"):
        return

    class LossKwargs(TypedDict, total=False):
        num_items_in_batch: Any

    tu.LossKwargs = LossKwargs
    notes.append("SDAR: transformers.utils.LossKwargs (removed in 5.x) shimmed as a TypedDict (modeling_sdar.py:46, :736)")


def _rope_params(config) -> dict:
    params = getattr(config, "rope_parameters", None) or {}
    if isinstance(params, dict) and "rope_theta" not in params and len(params) == 1:
        only = next(iter(params.values()))
        if isinstance(only, dict):
            params = only
    return params if isinstance(params, dict) else {}


def _shim_default_rope(notes: list[str]) -> None:
    import torch
    from transformers import modeling_rope_utils as mru
    if "default" in mru.ROPE_INIT_FUNCTIONS:
        return

    def _default_rope(config, device=None, seq_len=None, layer_type=None):
        # transformers 4.x _compute_default_rope_parameters, which SDAR was written against
        params = _rope_params(config)
        base = params.get("rope_theta", getattr(config, "rope_theta", None))
        partial = params.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", None)) or 1.0
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0

    mru.ROPE_INIT_FUNCTIONS["default"] = _default_rope
    notes.append('SDAR: ROPE_INIT_FUNCTIONS["default"] registered (5.x dropped it; modeling_sdar.py:411)')


def expected_inv_freq(cfg: dict):
    import torch
    if cfg.get("rope_scaling"):
        raise SystemExit(f"config.json sets rope_scaling={cfg['rope_scaling']!r}; this check only models plain RoPE")
    theta = cfg["rope_theta"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    dim = int(head_dim * float(cfg.get("partial_rotary_factor") or 1.0))
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))


def _fill_rotary(mod, inv) -> None:
    import torch
    buf = mod.inv_freq
    if not isinstance(buf, torch.Tensor) or buf.is_meta or tuple(buf.shape) != tuple(inv.shape):
        return  # checked and repaired after loading
    with torch.no_grad():
        buf.copy_(inv.to(buf.dtype))
        orig = mod._buffers.get("original_inv_freq") if "original_inv_freq" in mod._buffers else None
        if orig is not None and not orig.is_meta:
            orig.copy_(inv.to(orig.dtype))
        elif "original_inv_freq" not in mod._buffers and hasattr(mod, "original_inv_freq"):
            mod.original_inv_freq = buf  # the plain alias the modeling file's __init__ made


def _patch_rotary_init(module, cfg: dict, notes: list[str], why: str) -> None:
    """Route *RotaryEmbedding modules around the modeling file's _init_weights during loading."""
    from transformers import PreTrainedModel
    inv = expected_inv_freq(cfg)
    targets = [obj for obj in vars(module).values()
               if isinstance(obj, type) and issubclass(obj, PreTrainedModel) and obj.__module__ == module.__name__
               and ("_init_weights" in obj.__dict__ or obj.__name__.endswith("PreTrainedModel"))]
    if not targets:
        raise SystemExit(f"{module.__name__}: no PreTrainedModel subclass to patch")
    for cls in targets:
        orig = cls._init_weights

        def _init_weights(self, mod, _orig=orig):
            if "RotaryEmbedding" in type(mod).__name__ and hasattr(mod, "inv_freq"):
                _fill_rotary(mod, inv)
                return None
            return _orig(self, mod)

        cls._init_weights = _init_weights
    notes.append(f"{why}: _init_weights of {', '.join(c.__name__ for c in targets)} fills RoPE inv_freq from config.json")


def _patch_sdar(module, notes: list[str]) -> None:
    import torch
    norm = getattr(module, "SDARRMSNorm")

    def forward(self, hidden_states):
        # modeling_sdar.py:87-92, the eager RMSNorm left commented out under the flash-attn call
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    norm.forward = forward
    module.flash_attn_func = _flash_attn_func_sdpa
    module.flash_rms_norm = _unavailable("flash_rms_norm")
    module.liger_kernel_is_available = False
    notes.append("SDAR: SDARRMSNorm.forward -> eager (modeling_sdar.py:87-92); flash_attn_func -> SDPA "
                 "(:276-286); liger SwiGLU off (:58-62, :114)")


def _import_remote_class(ref: str, hf: Path, notes: list[str]):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    try:
        return get_class_from_dynamic_module(ref, str(hf), local_files_only=True)
    except Exception as first:
        import transformers.utils as tu
        orig = getattr(tu, "auto_docstring", None)
        if orig is None:
            raise

        def _noop(obj=None, **kwargs):
            return obj if obj is not None else (lambda o: o)

        notes.append(f"importing {ref} failed ({type(first).__name__}: {first}); retried with transformers' "
                     "auto_docstring disabled")
        tu.auto_docstring = _noop
        try:
            return get_class_from_dynamic_module(ref, str(hf), local_files_only=True)
        except Exception:
            raise first
        finally:
            tu.auto_docstring = orig


def _verify_rotary(model, cfg: dict, notes: list[str]) -> None:
    import torch
    inv = expected_inv_freq(cfg)
    found = 0
    for name, mod in model.named_modules():
        if "RotaryEmbedding" not in type(mod).__name__ or not hasattr(mod, "inv_freq"):
            continue
        found += 1
        cur = mod.inv_freq
        good = (isinstance(cur, torch.Tensor) and not cur.is_meta and tuple(cur.shape) == tuple(inv.shape)
                and bool(torch.isfinite(cur).all()) and bool(torch.allclose(cur.float(), inv, rtol=1e-5, atol=0.0)))
        if not good:
            mod.inv_freq = inv.clone()
            if "original_inv_freq" in mod._buffers:
                mod.original_inv_freq = inv.clone()
            notes.append(f"{name}.inv_freq did not hold RoPE(theta={cfg['rope_theta']}) after loading: set from config.json")
        if "original_inv_freq" not in mod._buffers and hasattr(mod, "original_inv_freq"):
            mod.original_inv_freq = mod.inv_freq
        if getattr(mod, "rope_type", "default") != "default" or getattr(mod, "attention_scaling", 1.0) != 1.0:
            raise SystemExit(f"{name}: rope_type {getattr(mod, 'rope_type', None)!r}, attention_scaling "
                             f"{getattr(mod, 'attention_scaling', None)!r}; expected plain RoPE")
    if not found:
        raise SystemExit("no *RotaryEmbedding module with an inv_freq buffer found in the model")


def load_model(kind: str, hf: Path, cfg: dict, dtype_name: str, attn: str, notes: list[str]):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import transformers

    if cfg.get("model_type") != MODEL_TYPE[kind]:
        raise SystemExit(f"{hf}: model_type {cfg.get('model_type')!r}, expected {MODEL_TYPE[kind]!r} for --model {kind}")
    ref = (cfg.get("auto_map") or {}).get("AutoModelForCausalLM")
    if not ref:
        raise SystemExit(f"{hf}/config.json has no auto_map.AutoModelForCausalLM")

    stubs: list[str] = []
    if kind == "sdar":
        _shim_loss_kwargs(notes)
        _shim_default_rope(notes)
        stubs = _install_flash_attn_stub(notes)
    try:
        cls = _import_remote_class(ref, hf, notes)
    finally:
        for n in stubs:
            sys.modules.pop(n, None)
    module = sys.modules[cls.__module__]
    if kind == "sdar":
        _patch_sdar(module, notes)
        _patch_rotary_init(module, cfg, notes, "SDAR (modeling_sdar.py:384-395 has no rotary branch)")
    elif kind == "wedlm":
        _patch_rotary_init(module, cfg, notes, "WeDLM (modeling_wedlm.py:122-145 vs modeling_utils.py:2562-2570)")

    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    print(f"loading {cls.__name__} from {hf} ({dtype_name}, attn {attn}; transformers {transformers.__version__}, "
          f"torch {torch.__version__}, {torch.get_num_threads()} threads) ...", flush=True)
    t0 = time.time()
    # transformers 5.x no longer gives every config a pad_token_id; SDAR's model reads it
    # (modeling_sdar.py:442) and it only sets the embedding's padding_idx, which inference never
    # touches. Passing it keeps that attribute present for every model.
    extra = {}
    if "pad_token_id" not in cfg:
        config = transformers.AutoConfig.from_pretrained(str(hf), trust_remote_code=True, local_files_only=True)
        if not hasattr(config, "pad_token_id"):
            config.pad_token_id = None
            notes.append("config.json has no pad_token_id and 5.x gives no default: set it to None")
        extra["config"] = config
    model, info = cls.from_pretrained(str(hf), dtype=dtype, attn_implementation=attn, trust_remote_code=True,
                                      output_loading_info=True, local_files_only=True, **extra)
    model.eval()
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)

    missing = sorted(info.get("missing_keys") or [])
    mismatched = list(info.get("mismatched_keys") or [])
    unexpected = sorted(info.get("unexpected_keys") or [])
    if missing or mismatched:
        raise SystemExit(f"checkpoint tensors missing {missing[:8]} or mismatched {mismatched[:4]}: they would be random")
    if unexpected:
        notes.append(f"{len(unexpected)} checkpoint tensor(s) unused: {unexpected[:5]}")
    on_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if on_meta:
        raise SystemExit(f"tensors left on the meta device after loading: {on_meta[:8]}")
    pdt = next(model.parameters()).dtype
    if pdt != dtype:
        raise SystemExit(f"parameters loaded as {pdt}, asked for {dtype}")
    _verify_rotary(model, cfg, notes)

    impl = getattr(model.config, "_attn_implementation", None)
    if kind in ("wedlm", "llada2") and impl not in ("sdpa", "eager"):
        raise SystemExit(f"attention implementation is {impl!r}; only sdpa and eager take the explicit 4D mask as is")
    if kind == "wedlm" and impl == "sdpa" and "sdpa" not in getattr(module, "ALL_ATTENTION_FUNCTIONS", {}):
        notes.append("WeDLM: sdpa missing from the module's ALL_ATTENTION_FUNCTIONS, so it runs eager "
                     "(modeling_wedlm.py:314-316); the additive mask is right for both")
    if kind == "sdar":
        if module.SDARRMSNorm.forward.__module__ != __name__ or module.flash_attn_func is not _flash_attn_func_sdpa:
            raise SystemExit("SDAR CPU patches did not take")
    if kind == "llada2":
        gates = [m for m in model.modules() if type(m).__name__ == "LLaDA2MoeGate"]
        sizes = {(int(g.block_size), int(g.expert_capacity), int(g.top_k)) for g in gates}
        notes.append(f"LLaDA2: {len(gates)} MoE gates, (block_size, expert_capacity, top_k) = {sorted(sizes)}")
        if len(sizes) != 1:
            raise SystemExit("LLaDA2 gates disagree on block_size / expert_capacity")
    try:
        import psutil
        notes.append(f"resident set after loading: {psutil.Process().memory_info().rss / 2**30:.1f} GB")
    except Exception:
        pass
    return model, module


class HFRunner:
    def __init__(self, kind: str, model, dtype_name: str, native_head: bool):
        import torch
        self.torch = torch
        self.kind = kind
        self.model = model
        self.dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
        self.native_head = native_head
        self.vocab = int(model.lm_head.weight.shape[0])
        self._w32 = None
        self._b32 = None

    def _hidden(self, ids, pos, mask4):
        base = self.model.model
        quiet = dict(use_cache=False, output_attentions=False, output_hidden_states=False)
        if self.kind == "wedlm":
            # the dict skips create_causal_mask (modeling_wedlm.py:498-509)
            out = base(input_ids=ids, attention_mask={"full_attention": mask4}, position_ids=pos,
                       return_dict=True, **quiet)
        elif self.kind == "sdar":
            out = base(input_ids=ids, attention_mask=mask4, position_ids=pos, store_kv=False, **quiet)
        else:
            out = base(input_ids=ids, attention_mask=mask4, position_ids=pos, output_router_logits=False,
                       return_dict=True, **quiet)
        return out[0] if isinstance(out, tuple) else out.last_hidden_state

    def logits(self, plan: Plan):
        torch = self.torch
        ids = torch.tensor([plan.tokens], dtype=torch.long)
        pos = torch.tensor([plan.positions], dtype=torch.long)
        allowed = torch.from_numpy(plan.mask)
        if self.kind == "sdar":
            mask4 = allowed[None, None]                         # bool, True = attend
        else:
            mask4 = torch.zeros(allowed.shape, dtype=self.dtype)
            mask4.masked_fill_(~allowed, torch.finfo(self.dtype).min)
            mask4 = mask4[None, None]                           # additive
        with torch.no_grad():
            h = self._hidden(ids, pos, mask4)
            sel = h[0, torch.tensor(plan.logit_index, dtype=torch.long)]
            head = self.model.lm_head
            if self.native_head:
                return head(sel).float()
            if self._w32 is None:
                self._w32 = head.weight.detach().float()
                self._b32 = None if head.bias is None else head.bias.detach().float()
            return torch.nn.functional.linear(sel.float(), self._w32, self._b32)


# =============================================================================================
# COMPARISON AND REPORT
# =============================================================================================

@dataclass
class RowResult:
    trace: str
    pass_idx: int
    row: int
    pos: int
    k: int
    drv_top: list
    hf_top: list
    drv_top1: int
    hf_top1: int
    hf_logit_of_drv_top1: float
    agree: bool
    overlap: int
    pearson: float
    maxabs: float
    swing: float | None = None      # how far the two candidates' difference moved between runs
    swing_is_bound: bool = False    # HF's top-1 is not in the driver's top-k: swing is a lower bound
    hard: bool = False


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xs, ys = x - x.mean(), y - y.mean()
    d = math.sqrt(float((xs * xs).sum()) * float((ys * ys).sum()))
    return float((xs * ys).sum()) / d if d > 0 else float("nan")


def compare_row(hf_row, top: list[tuple[int, float]], tie_eps: float, trace: str, pass_idx: int, row: int,
                pos: int) -> RowResult:
    import torch
    k = len(top)
    ids = [t for t, _ in top]
    tl = np.array([v for _, v in top], dtype=np.float64)
    hv = hf_row[torch.tensor(ids, dtype=torch.long)].double().numpy()
    vals, idx = torch.topk(hf_row, k)
    hf_ids, hf_vals = idx.tolist(), vals.tolist()
    A, B = hf_ids[0], ids[0]
    res = RowResult(trace=trace, pass_idx=pass_idx, row=row, pos=pos, k=k,
                    drv_top=[[i, v] for i, v in top], hf_top=[[i, v] for i, v in zip(hf_ids, hf_vals)],
                    drv_top1=B, hf_top1=A, hf_logit_of_drv_top1=float(hf_row[B]), agree=A == B,
                    overlap=len(set(hf_ids) & set(ids)), pearson=pearson(tl, hv),
                    maxabs=float(np.max(np.abs(hv - tl))))
    if not res.agree:
        hf_gap = float(hf_row[A]) - float(hf_row[B])          # >= 0: how much HF prefers A
        if A in ids:
            drv_gap = tl[0] - tl[ids.index(A)]                 # >= 0: how much the driver prefers B
        else:
            drv_gap = tl[0] - tl[-1]                           # the driver's A is below its k-th
            res.swing_is_bound = True
        res.swing = hf_gap + float(drv_gap)
        res.hard = res.swing_is_bound or res.swing > tie_eps
    return res


def stats_of(results: list[RowResult]) -> dict:
    n = len(results)
    if not n:
        return {"rows": 0}
    rs = sorted(r.pearson for r in results if not math.isnan(r.pearson))
    agree = sum(r.agree for r in results)
    hard = sum((not r.agree) and r.hard for r in results)
    return {
        "rows": n,
        "agree": agree,
        "near_tie": n - agree - hard,
        "hard": hard,
        "overlap_mean": float(np.mean([r.overlap / r.k for r in results])),
        "k": max(r.k for r in results),
        "r_median": float(np.median(rs)) if rs else None,
        "r_p05": float(np.percentile(rs, 5)) if rs else None,
        "r_min": rs[0] if rs else None,
        "r_nan": n - len(rs),
        "maxabs": max(r.maxabs for r in results),
        "maxabs_median": float(np.median([r.maxabs for r in results])),
    }


def verdict(st: dict, results: list[RowResult]) -> tuple[str, str]:
    """Heuristic thresholds, stated so a reader can overrule them."""
    n = st.get("rows", 0)
    if not n:
        return "NO DATA", "no logit rows were compared"
    med = st["r_median"]
    first_bad = min((r.pass_idx for r in results if (not r.agree and r.hard) or r.pearson < 0.9), default=None)
    where = f"pass {first_bad}" if first_bad is not None else "the first pass with a low r_mean in the table"
    if st["hard"] == 0 and (med is None or med >= 0.98):
        return "CONSISTENT", ("the driver reproduces the HF original on every compared row up to rounding and "
                              "quantization noise; any top-1 flips are near-ties")
    if st["hard"] <= max(1, round(0.05 * n)) and (med is None or med >= 0.95):
        return "MOSTLY CONSISTENT", (f"isolated hard top-1 flips (first in {where}); expected now and then at "
                                     "Q4_1 and at LLaDA2 routing near-ties, suspicious at Q8_0 if they cluster "
                                     "in one pass")
    return "INCONSISTENT", (f"systematic disagreement from {where}: suspect the driver's positions, slots, mask, "
                            "cache trimming or (LLaDA2) block routing there, or a GGUF/model mismatch")


def fmt(x, spec: str) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def print_summary(title: str, st: dict, results: list[RowResult], tie_eps: float) -> tuple[str, str]:
    v, why = verdict(st, results)
    print(f"\n== {title}")
    if not st.get("rows"):
        print(f"verdict            {v}: {why}")
        return v, why
    n = st["rows"]
    print(f"logit rows         {n}")
    print(f"top-1 agreement    {st['agree']}/{n} ({100.0 * st['agree'] / n:.1f}%); disagreements: "
          f"{st['near_tie']} near-tie (swing <= {tie_eps}), {st['hard']} hard")
    print(f"top-{st['k']} overlap      mean {st['overlap_mean'] * st['k']:.2f} / {st['k']}")
    print(f"pearson r (top-k)  median {fmt(st['r_median'], '.4f')}, 5th pct {fmt(st['r_p05'], '.4f')}, "
          f"min {fmt(st['r_min'], '.4f')}" + (f", {st['r_nan']} undefined (flat)" if st["r_nan"] else ""))
    print(f"max |d logit|      {st['maxabs']:.3f} (median per row {st['maxabs_median']:.3f})")
    print(f"verdict            {v}: {why}")
    return v, why


class Tok:
    def __init__(self, hf: Path | None):
        self.tk = None
        if hf is None:
            return
        try:
            from tokenizers import Tokenizer
            self.tk = Tokenizer.from_file(str(hf / "tokenizer.json"))
        except Exception:
            self.tk = None

    def __call__(self, i: int) -> str:
        if self.tk is None:
            return ""
        try:
            return " " + repr(self.tk.decode([i], skip_special_tokens=False))
        except Exception:
            return ""


def check_trace(tr: Trace, kind: str, args, runner: HFRunner | None, tok: Tok, routing_block: int | None,
                results: list[RowResult]) -> None:
    """Replays one trace, appending to `results` as it goes, so an error or Ctrl-C mid-trace keeps
    what ran."""
    passes = tr.passes
    n_pre, n_default = default_pass_count(passes)
    n = min(len(passes), args.passes if args.passes is not None else n_default)
    print(f"\n#### trace {tr.label}  [{tr.path}]")
    if tr.meta:
        print("     " + ", ".join(f"{k}={v}" for k, v in tr.meta.items()))
    model_hint = str(tr.meta.get("model", "")) + " " + str(tr.meta.get("gguf", ""))
    if model_hint.strip() and kind not in model_hint.lower().replace("-", "").replace(".", ""):
        print(f"     [warn] the trace names {model_hint.strip()!r}; checking it as --model {kind}")
    print(f"     {len(passes)} passes ({n_pre} prefill); replaying the first {n}"
          + (f"; output {len(tr.output)} tokens" if tr.output is not None else ""))
    odd = sorted({p.attn for p in passes[:n]} - {EXPECTED_ATTN[kind]})
    if odd:
        print(f"     [warn] {kind} passes are {EXPECTED_ATTN[kind]}; this trace also has {odd} (replayed as written)")

    print(f"\n{'pass':>4} {'kind':<8} {'attn':<6} {'rows':>4} {'keep':>5} {'ctx':>5} {'L':>5} {'ghost':>5} "
          f"{'lrows':>5} {'top1':>7} {'near/hard':>9} {'ovl':>5} {'r_mean':>7} {'r_min':>7} {'max|d|':>7} "
          f"{'sec':>6}  mask")
    replay = Replay(routing_block)
    try:
        _replay_passes(passes[:n], n_pre, args, runner, replay, tr.label, results)
    finally:
        for w in replay.warnings:
            print(f"     [warn] {w}")
        _print_rows(results, tok, args.show)


def _replay_passes(passes: list[TPass], n_pre: int, args, runner: HFRunner | None, replay: Replay, label: str,
                   results: list[RowResult]) -> None:
    for p in passes:
        q = replay.add(p)
        pk = p.phase or ("prefill" if p.index < n_pre else ("decode" if p.logit_rows else "commit"))
        head = f"{p.index:>4} {pk[:8]:<8} {p.attn:<6} {len(p.rows):>4} {p.keep:>5}"
        if not p.logit_rows:
            print(f"{head} {'':>5} {'':>5} {'':>5} {0:>5}   (no logit rows: only builds the cache)")
        elif p.tops is None:
            print(f"{head}   [warn] logit rows but no top-k recorded: not compared")
        else:
            plan = replay.plan(q, p)
            pre = f"{head} {plan.n_ctx:>5} {len(plan.tokens):>5} {plan.ghosts:>5} {len(p.logit_rows):>5}"
            if runner is None:
                print(f"{pre} {'(dry run)':>58}  {plan.structure}")
            else:
                print(f"{pre} ...", end="\r", flush=True)
                bad_ids = sorted({i for t in p.tops for i, _ in t if not 0 <= i < runner.vocab}
                                 | {t for t in plan.tokens if not 0 <= t < runner.vocab})
                if bad_ids:
                    raise TraceError(f"pass {p.index}: token ids outside the HF vocab ({runner.vocab}): {bad_ids[:5]}")
                t0 = time.time()
                lg = runner.logits(plan)
                dt = time.time() - t0
                rr = [compare_row(lg[k], p.tops[k], args.tie_eps, label, p.index, r, p.rows[r].pos)
                      for k, r in enumerate(p.logit_rows)]
                results += rr
                agree = sum(x.agree for x in rr)
                hard = sum((not x.agree) and x.hard for x in rr)
                near = len(rr) - agree - hard
                rs = [x.pearson for x in rr if not math.isnan(x.pearson)]
                ovl = float(np.mean([x.overlap for x in rr]))
                print(f"{pre} {agree:>3}/{len(rr):<3} {near:>4}/{hard:<4} {ovl:>5.2f} "
                      f"{fmt(float(np.mean(rs)) if rs else None, '7.4f')} {fmt(min(rs) if rs else None, '7.4f')} "
                      f"{max(x.maxabs for x in rr):>7.3f} {dt:>6.1f}  {plan.structure}", flush=True)
        replay.commit(q, p.keep)


def _print_rows(results: list[RowResult], tok: Tok, show: int) -> None:
    flips = [r for r in results if not r.agree]
    if flips:
        print(f"\ntop-1 disagreements ({len(flips)}; showing {min(len(flips), show)}):")
        for r in flips[:show]:
            sw = ("-" if r.swing is None else (">=" if r.swing_is_bound else "") + f"{r.swing:.3f}")
            print(f"  pass {r.pass_idx} row {r.row} pos {r.pos}: driver {r.drv_top1}{tok(r.drv_top1)} "
                  f"{r.drv_top[0][1]:.3f} | HF {r.hf_top1}{tok(r.hf_top1)} {r.hf_top[0][1]:.3f}, "
                  f"HF on driver's pick {r.hf_logit_of_drv_top1:.3f}; swing {sw} -> {'HARD' if r.hard else 'near-tie'}")
    low = sorted((r for r in results if not math.isnan(r.pearson)), key=lambda r: r.pearson)[:5]
    if low and low[0].pearson < 0.99:
        print("\nlowest r rows:")
        for r in low:
            print(f"  pass {r.pass_idx} row {r.row} pos {r.pos}: r {r.pearson:.4f}, max|d| {r.maxabs:.3f}, "
                  f"overlap {r.overlap}/{r.k}; driver top-3 {[(i, round(v, 2)) for i, v in r.drv_top[:3]]}, "
                  f"HF top-3 {[(i, round(v, 2)) for i, v in r.hf_top[:3]]}")


# =============================================================================================
# MAIN
# =============================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="See the module docstring for the method, the CPU swaps and the costs.")
    ap.add_argument("--model", required=True, choices=sorted(DEFAULT_HF))
    ap.add_argument("--hf", type=Path, default=None, help="HF model directory (default: C:\\dev\\dllm\\hf\\<model>)")
    ap.add_argument("--trace", type=Path, nargs="+", required=True,
                    help="trace JSON file(s), or directories of *.json; the model loads once for all of them")
    ap.add_argument("--passes", type=int, default=None,
                    help="replay the first N passes of each trace (default: the prefill passes plus 4)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                    help="HF weights/activations dtype (fp32 is the cleaner reference; LLaDA2 fp32 needs ~65 GB)")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa",
                    help="transformers attention for WeDLM and LLaDA2 (SDAR's attention ignores it)")
    ap.add_argument("--native-head", action="store_true",
                    help="apply lm_head in the model dtype, as the HF code does (default: fp32 head)")
    ap.add_argument("--tie-eps", type=float, default=0.5,
                    help="a top-1 disagreement is a near-tie when the two candidates' logit difference moved by "
                         "at most this much between the runs (default 0.5)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    ap.add_argument("--show", type=int, default=20, help="list at most this many top-1 disagreements per trace")
    ap.add_argument("--json-out", type=Path, default=None, help="write every compared row here")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse the traces and print each pass's replay layout; import no torch, load nothing")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except Exception:
        pass

    hf = args.hf or DEFAULT_HF[args.model]
    cfg = None
    if (hf / "config.json").is_file():
        cfg = read_config(hf)
    elif not args.dry_run:
        raise SystemExit(f"{hf}: no config.json")
    if cfg is not None and cfg.get("model_type") != MODEL_TYPE[args.model]:
        raise SystemExit(f"{hf}: model_type {cfg.get('model_type')!r}, expected {MODEL_TYPE[args.model]!r} "
                         f"for --model {args.model}")
    routing_block = None
    if args.model == "llada2":
        routing_block = int((cfg or {}).get("block_size", 32))

    traces = load_traces(args.trace)
    if not traces:
        raise SystemExit("no readable traces")

    runner = None
    notes: list[str] = []
    if not args.dry_run:
        import torch
        if args.threads:
            torch.set_num_threads(args.threads)
        model, _ = load_model(args.model, hf, cfg, args.dtype, args.attn, notes)
        runner = HFRunner(args.model, model, args.dtype, args.native_head)
        for n_ in notes:
            print(f"  note: {n_}")
        if args.model == "llada2" and routing_block != int(getattr(model.config, "block_size", routing_block)):
            raise SystemExit("config.json block_size and the loaded config disagree")
    tok = Tok(hf if not args.dry_run and (hf / "tokenizer.json").is_file() else None)

    all_results: list[RowResult] = []
    per_trace = []
    interrupted = False
    for tr in traces:
        res: list[RowResult] = []
        entry: dict = {"trace": tr.label, "path": str(tr.path)}
        try:
            check_trace(tr, args.model, args, runner, tok, routing_block, res)
        except TraceError as e:
            print(f"\n[trace {tr.label}] stopped: {e}")
            entry["error"] = str(e)
        except KeyboardInterrupt:
            print("\ninterrupted; summarising what ran")
            interrupted = True
        all_results += res
        if runner is not None:
            st = stats_of(res)
            v, why = print_summary(tr.label + (" (partial)" if "error" in entry or interrupted else ""),
                                   st, res, args.tie_eps)
            entry.update(stats=st, verdict=v, why=why)
        per_trace.append(entry)
        if interrupted:
            break

    if runner is None:
        print("\n(dry run: nothing loaded, nothing compared)")
        return 0
    if len(traces) > 1 or interrupted:
        st = stats_of(all_results)
        print_summary(f"ALL TRACES ({args.model}, HF {args.dtype}, {len(per_trace)} traces)",
                      st, all_results, args.tie_eps)
    print(f"\n(HF in {args.dtype}{', native-dtype LM head' if args.native_head else ', LM head in fp32'}; "
          "thresholds are heuristics -- read the numbers. Expect corr near 0.999 and top-1 flips only at "
          "near-ties at Q8_0 with --dtype fp32; bf16 and Q4_1 widen both.)")

    if args.json_out:
        doc = {"model": args.model, "hf": str(hf), "dtype": args.dtype, "attn": args.attn,
               "native_head": args.native_head, "tie_eps": args.tie_eps, "notes": notes,
               "traces": per_trace, "rows": [r.__dict__ for r in all_results]}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, default=float)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
