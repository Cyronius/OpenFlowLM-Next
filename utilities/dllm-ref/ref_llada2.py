#!/usr/bin/env python3
"""ref_llada2.py: LLaDA2.2-mini's own decode loop, driven by a scripted fake model.

Writes specs/open-engine/tests/fixtures/dllm_llada2.json, which src/open_dllm/llada2_test.cpp
replays against make_llada2(). Traces: OPEN-DLLM-LLADA2 (plan:
specs/open-engine/plans/diffusion-lm.md).

What runs is the reference's code, not a re-implementation of it. The eight methods of
LLaDA2MoeModelLM the decode uses (generate, _joint_decode_block, _resample_to_escape_loop,
_apply_edit_operations_with_tracking, _diffusion_sample, _get_num_transfer_tokens,
_top_k_logits, _top_p_logits) are cut out of modeling_llada2_moe.py by source (ast) and
exec'd unchanged, so transformers is never imported. One thing is swapped: forward().
The scripted model reads the block-diffusion attention mask generate() builds, gives each
row a two-layer hash "K/V" of exactly the rows it can see, and gives the current block's
rows logits that depend on the row's position, its token, its left neighbour and a hash of
that visible context. The other overrides only count things and call through.

The reference keeps no KV cache: every step re-forwards x[:, :block_end]. The pass list a
cached backend runs is derived from that trace: one block pass per full prompt block, one
pass per reference forward, and a commit pass when a block's last forward did not see the
block's final tokens. That list is then replayed through an emulated KV cache (rows dropped
at keep() after every pass) with the same scripted model, and every pass's logits must
equal, bit for bit, the logits the reference's full re-forward produced. A commit pass in
the wrong place, or a wrong keep(), changes a later row's hash and fails that check.

Randomness: the reference draws from torch's RNG (multinomial when temperature > 0,
randint to pick the position to resample on a repeated state); the C++ policy draws from
mt19937_64. The fixture only contains draws with exactly one possible outcome (sampling
with top_k=1 or a tiny top_p; a repeat with one changed position), and the harness asserts
that for every draw. Logits are float64 multiples of 1/32, and every threshold, top-k and
argmax decision is checked to clear its boundary by a margin, so the C++ side (double)
makes the same call.

Usage:
    python utilities/dllm-ref/ref_llada2.py --ref <modeling_llada2_moe.py or its directory>
        [--out specs/open-engine/tests/fixtures/dllm_llada2.json]
  (the file comes from hf.co/inclusionAI/LLaDA2.2-mini; checked at revision 038922ee.)
"""

import argparse
import ast
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "specs" / "open-engine" / "tests" / "fixtures" / "dllm_llada2.json"

# ---- the scripted vocabulary ---------------------------------------------------------------
V = 48
STOP = 44     # eos_id / Common::stop_ids (the real 156892)
MASK = 45     # mask_id (156895)
DELETE = 46   # delete_token_id (156930)
INSERT = 47   # split_token_id (156931)
J = 43        # junk: a filled J asks to be deleted
X, Y = 42, 41 # X wants Y in front of it: a filled X after anything else asks for an INSERT
O1, O2 = 39, 40  # an oscillating pair: a filled O1 wants O2 and back
F = 38        # O1/O2's second choice; stable once written
# 0..15: prompt tokens and wrong guesses; 16..37: answer tokens.

# Logit levels. Rows predict one token with a bump over background noise in [-0.5, 0.5];
# p(top) is ~0.13 at 2.0, ~0.5 at 3.87, ~0.89 at 6.0 and ~0.96 at 7.0 (V = 48).
C_FIRST, C_CHAIN, C_EDIT, C_STABLE = 6.0, 6.0, 6.0, 7.0
C_POS, GAIN = 2.0, 0.75   # a position-only guess, +GAIN per filled neighbour within 2
C_EASY_STOP = 2.5         # added when that guess is STOP

M64 = (1 << 64) - 1
TUNING = False   # --tuning: report a scenario that misses its features instead of failing


def _splitmix(z):
    z = (z + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def mix(*vals):
    h = 0x243F6A8885A308D3
    for v in vals:
        h = _splitmix(h ^ (v & M64))
    return h


def kv0(tok, pos):
    """Layer-0 "K/V": a function of the token and its position only."""
    return mix(1, tok, pos)


def kv1(tok, pos, visible_kv0):
    """Layer-1 "K/V": depends on every row the row attends to (position order)."""
    return mix(2, tok, pos, *visible_kv0)


# ---- scenarios ------------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    what: str
    prompt_len: int
    answer: List[int]
    gen: int
    expect: List[str]                       # features this scenario must exercise
    block: int = 8
    steps: int = 8
    threshold: float = 0.5
    editing_threshold: float = 0.0
    max_post_steps: int = 16
    max_steps_per_block: int = 1000
    eos_early_stop: bool = True
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 0
    wrong: dict = field(default_factory=dict)   # pos -> wrong position-only guess
    first: dict = field(default_factory=dict)   # pos -> confident guess while the block is all masks
    fixed: dict = field(default_factory=dict)   # pos -> [(token, logit), ...] for a mask row there, always
    bias: dict = field(default_factory=dict)    # pos -> added to the position-only guess's logit

    def __post_init__(self):
        L = self.prompt_len
        self.prompt = [(5 * i + 3) % 16 for i in range(L)]
        self.T = self.prompt + list(self.answer) + [STOP]
        assert len(set(self.answer)) == len(self.answer), "answer tokens must be unique"
        assert self.prompt[-1] not in self.answer
        self.nxt = {self.T[i]: self.T[i + 1] for i in range(L - 1, len(self.T) - 1)}
        self.nxt[STOP] = STOP
        if O1 in self.nxt:
            self.nxt[O2] = self.nxt[F] = self.nxt[O1]

    def truth(self, pos):
        return self.T[pos] if pos < len(self.T) else STOP

    def rule(self, tok, pos, vis):
        """[(token, logit)] the row predicts. vis: {pos: (token, kv1)} the row can see."""
        left = vis[pos - 1][0] if (pos - 1) in vis else None
        B = self.block
        b0 = pos // B * B
        if tok == MASK:
            if pos in self.fixed:
                return self.fixed[pos]
            if pos in self.first and all(
                vis[p][0] == MASK for p in range(max(b0, self.prompt_len), b0 + B) if p in vis
            ):
                return [(self.first[pos], C_FIRST)]
            if left in self.nxt:
                return [(self.nxt[left], C_CHAIN)]
            n = sum(
                1 for p in range(pos - 2, pos + 3)
                if p != pos and b0 <= p < b0 + B and p in vis and vis[p][0] != MASK
            )
            g = self.wrong.get(pos, self.truth(pos))
            # trailing STOPs are an easy guess (a real model pads with EOS in bulk)
            easy = C_EASY_STOP if g == STOP else 0.0
            return [(g, C_POS + GAIN * n + easy + self.bias.get(pos, 0.0))]
        if tok == J:
            return [(DELETE, 7.0), (self.nxt.get(left, self.truth(pos)), 3.0)]
        if tok == X and left not in (None, MASK, Y):
            return [(INSERT, 7.0), (X, 3.0)]
        if tok in (O1, O2):
            return [(O2 if tok == O1 else O1, 7.0), (F, 4.0)]
        if tok == F:
            return [(F, C_STABLE)]
        if left in self.nxt and tok != self.nxt[left]:
            return [(self.nxt[left], C_EDIT)]
        return [(tok, C_STABLE)]

    def config(self):
        return {
            "block": self.block, "steps": self.steps, "threshold": self.threshold,
            "editing_threshold": self.editing_threshold, "max_post_steps": self.max_post_steps,
            "max_steps_per_block": self.max_steps_per_block, "delete_id": DELETE,
            "insert_id": INSERT, "eos_early_stop": self.eos_early_stop, "mask_id": MASK,
            "stop_ids": [STOP], "max_new_tokens": self.gen, "temperature": self.temperature,
            "top_k": self.top_k, "top_p": self.top_p, "seed": self.seed,
        }


def row_logits(sc, tok, pos, vis):
    ctx = mix(3, tok, pos, *(vis[p][1] for p in sorted(vis)))
    out = [((mix(ctx, v) % 9) - 4) / 8.0 for v in range(V)]
    out[MASK] = -8.0
    for t, val in sc.rule(tok, pos, vis):
        out[t] = val + ((mix(ctx, 1000 + t) % 9) - 4) / 32.0
    return out


A = list(range(16, 38))   # answer pool

SCENARIOS = [
    Scenario(
        "tail_threshold",
        "3-token prompt tail rides with the first block; several confident guesses fill "
        "by threshold in one step; STOP in block 2 stops the request before block 3",
        prompt_len=11, answer=A[:10], gen=20,
        bias={13: 2.5, 14: 2.5, 18: 2.5, 19: 2.5},
        expect=["tail", "threshold_fill", "stop", "early_stop"],
    ),
    Scenario(
        "aligned_wrong_guesses",
        "prompt is exactly one block (no tail); steps=4 (2 per step); two wrong guesses are "
        "rewritten by token-to-token edits",
        prompt_len=8, answer=A[:12], gen=16, steps=4,
        wrong={11: 2, 12: 4, 17: 6},
        expect=["no_tail", "t2t", "topk_fill", "stop"],
    ),
    Scenario(
        "short_prompt_edit_gate",
        "prompt shorter than a block (no prefill pass); steps=3; editing_threshold 0.95 "
        "blocks every rewrite, so wrong guesses stay; output cut at max_new_tokens mid-block",
        prompt_len=5, answer=A[:14], gen=9, steps=3, editing_threshold=0.95,
        wrong={7: 2, 9: 4}, bias={7: 2.5, 9: 2.5},
        expect=["short_prompt", "edit_gated", "truncated"],
    ),
    Scenario(
        "delete",
        "a junk token guessed on the first step is deleted (T2T DELETE): the block shifts "
        "left and a fresh, non-original mask is padded at the end",
        prompt_len=8, answer=A[:12], gen=16, first={10: J},
        expect=["delete", "new_masks", "stop"],
    ),
    Scenario(
        "insert",
        "X guessed where Y belongs; once X's left neighbour is filled with something else "
        "it becomes INSERT: a mask opens in front of X and the block's last row is truncated",
        prompt_len=11, answer=A[:2] + [Y, X] + A[2:10], gen=16, first={13: X},
        expect=["insert", "new_masks", "tail", "stop"],
    ),
    Scenario(
        "insert_on_mask",
        "a mask row predicts INSERT on the first step: M2T writes it, the edit opens "
        "[new MASK, original MASK] and truncates the block's last (original) mask; "
        "max_post_steps=1, so the first head with only that new mask open is the final round",
        prompt_len=8, answer=A[:12], gen=16, max_post_steps=1, first={11: INSERT},
        expect=["insert", "new_masks", "final_round", "stop"],
    ),
    Scenario(
        "pad_mask_final_zero",
        "max_post_steps=0 with a junk delete early: the mask padded at the block end is not "
        "original and stays unconfident, so the step whose M2T covers the last two original "
        "masks is final while it is still open, and the final round fills it anyway",
        prompt_len=8, answer=A[:12], gen=16, max_post_steps=0, first={11: J},
        fixed={15: [(A[7], 1.0)]}, bias={13: 2.5},
        expect=["delete", "new_masks", "final_open_new_mask", "exit_post_steps", "commit_pass"],
    ),
    Scenario(
        "pad_mask_post_one",
        "pad_mask_final_zero's script with max_post_steps=1: post-steps count from the first "
        "head with no ORIGINAL mask, while the padded mask is still open, and that step is "
        "the final round",
        prompt_len=8, answer=A[:12], gen=16, max_post_steps=1, first={11: J},
        fixed={15: [(A[7], 1.0)]}, bias={13: 2.5},
        expect=["delete", "post_with_new_mask", "final_round", "commit_pass"],
    ),
    Scenario(
        "post_steps_final_t2t",
        "max_post_steps=1: a junk guess is the last mask filled, so the next step is the "
        "final round and its DELETE is re-picked with DELETE/INSERT at -inf; the block exits "
        "on the post-step budget and needs a commit pass",
        prompt_len=8, answer=A[:14], gen=16, max_post_steps=1,
        fixed={12: [(J, 1.5), (A[4], 1.0)]},
        expect=["final_round", "ds_repick", "exit_post_steps", "commit_pass"],
    ),
    Scenario(
        "max_steps_exit",
        "max_steps_per_block=2 with unconfident guesses: each block stops with masks left, "
        "the next block decodes over them, and each needs a commit pass",
        prompt_len=8, answer=A[:14], gen=16, max_steps_per_block=2,
        bias={p: -0.5 for p in range(8, 24)},
        expect=["exit_max_steps", "commit_pass", "residual_mask"],
    ),
    Scenario(
        "max_post_zero",
        "max_post_steps=0: the step whose M2T covers the last original mask is final; that "
        "mask predicts DELETE and is re-picked; exit on the post-step check, commit pass",
        prompt_len=11, answer=A[:12], gen=16, max_post_steps=0,
        fixed={14: [(DELETE, 1.5), (A[3], 1.0)]},
        expect=["final_round", "ds_repick", "exit_post_steps", "commit_pass", "tail"],
    ),
    Scenario(
        "temperature_top_k1",
        "max_post_zero's script at temperature 0.7, top_k 1: the sampled path (x_s, p_s from "
        "the unfiltered softmax, the final-round M2T re-pick) with one candidate per draw",
        prompt_len=11, answer=A[:12], gen=16, max_post_steps=0, temperature=0.7, top_k=1,
        seed=7, fixed={14: [(DELETE, 1.5), (A[3], 1.0)]},
        expect=["final_round", "ds_repick", "sampled"],
    ),
    Scenario(
        "temperature_top_p",
        "tail_threshold's script at temperature 1.3, top_p 0.02: nucleus filtering leaves one "
        "candidate per draw",
        prompt_len=11, answer=A[:10], gen=20, temperature=1.3, top_p=0.02, seed=11,
        bias={13: 2.5, 14: 2.5, 18: 2.5, 19: 2.5},
        expect=["threshold_fill", "sampled", "stop"],
    ),
    Scenario(
        "resample_one_position",
        "O1 and O2 rewrite each other once the block is otherwise complete; the repeated "
        "pre-edit state triggers the anti-loop resample with exactly one changed position, "
        "which takes the second choice F",
        prompt_len=8, answer=A[:7] + [O1] + A[7:12], gen=16,
        expect=["resample", "stop"],
    ),
    Scenario(
        "no_early_stop",
        "eos_early_stop=false: STOP in block 1, blocks 2 and 3 still decode (STOP after "
        "STOP) and emit nothing",
        prompt_len=8, answer=A[:4], gen=12, eos_early_stop=False,
        expect=["stop", "decoded_after_stop"],
    ),
    Scenario(
        "block32_default_geometry",
        "the shipped geometry: block 32, steps 32, an 8-token prompt tail, one decode block",
        prompt_len=40, answer=A[:15], gen=20, block=32, steps=32,
        wrong={45: 2, 52: 4}, bias={p: 2.5 for p in range(40, 64) if p % 4 != 1},
        expect=["tail", "threshold_fill", "t2t", "stop"],
    ),
]


# ---- the reference, loaded by source ------------------------------------------------------
NEEDED = [
    "_top_k_logits", "_top_p_logits", "_get_num_transfer_tokens",
    "_apply_edit_operations_with_tracking", "_diffusion_sample", "_resample_to_escape_loop",
    "_joint_decode_block", "generate",
]


class RecordingLogger:
    """Stands in for the module's transformers logger; keeps the debug lines (the
    reference logs every step's mask counts and post-step count, and why a block ended)."""

    def __init__(self):
        self.sink = None

    def _rec(self, level, msg, *args):
        if self.sink is not None:
            self.sink.append((level, msg % args if args else msg))

    def debug(self, msg, *args):
        self._rec("debug", msg, *args)

    def info(self, msg, *args):
        self._rec("info", msg, *args)

    def warning(self, msg, *args):
        self._rec("warning", msg, *args)

    def warning_once(self, msg, *args):
        self._rec("warning", msg, *args)


LOG = RecordingLogger()


def load_reference(path: Path):
    if path.is_dir():
        path = path / "modeling_llada2_moe.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LLaDA2MoeModelLM")
    lines = src.splitlines(keepends=True)
    chunks = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in NEEDED:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            chunks[node.name] = (start, node.end_lineno, "".join(lines[start - 1:node.end_lineno]))
    missing = [n for n in NEEDED if n not in chunks]
    if missing:
        raise SystemExit(f"{path}: LLaDA2MoeModelLM has no {missing}")
    code = "class LLaDA2DecodeReference:\n" + "".join(chunks[n][2] for n in NEEDED)
    ns = {
        "torch": torch, "F": torch.nn.functional, "math": math, "logger": LOG,
        "Optional": Optional, "List": List, "Tuple": Tuple, "Union": Union, "Callable": Callable,
    }
    exec(compile(code, f"{path.name}:LLaDA2MoeModelLM", "exec"), ns)
    info = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "functions": {n: [chunks[n][0], chunks[n][1]] for n in NEEDED},
    }
    return ns["LLaDA2DecodeReference"], info


# ---- the scripted model + counters --------------------------------------------------------
@dataclass
class Forward:
    tokens: list
    logits: list   # the block's rows, each V floats


@dataclass
class BlockRec:
    bs: int
    be: int
    entry: list
    forwards: list = field(default_factory=list)
    final: list = None
    log: list = field(default_factory=list)
    edits_delete: int = 0
    edits_insert: int = 0
    ds_repicks: int = 0
    resamples: list = field(default_factory=list)   # changed positions at each repeat
    tracking: list = field(default_factory=list)    # is_original_mask at each step's start


def make_fake(Ref):
    class Scripted(Ref):
        def __init__(self, sc):
            self.sc = sc
            self.device = torch.device("cpu")
            self.blocks = []
            self.cur = None

        def forward(self, input_ids, attention_mask=None, position_ids=None, **kw):
            ids = input_ids[0].tolist()
            pos = position_ids[0].tolist()
            visible = (attention_mask[0, 0] == 0).tolist()
            n = len(ids)
            k0 = [kv0(ids[j], pos[j]) for j in range(n)]
            seen = [[j for j in range(n) if visible[i][j]] for i in range(n)]
            k1 = [kv1(ids[i], pos[i], [k0[j] for j in sorted(seen[i], key=lambda j: pos[j])])
                  for i in range(n)]
            B = self.sc.block
            logits = torch.zeros(1, n, V, dtype=torch.float64)
            blk_logits = []
            for i in range(n - B, n):
                vis = {pos[j]: (ids[j], k1[j]) for j in seen[i]}
                row = row_logits(self.sc, ids[i], pos[i], vis)
                logits[0, i] = torch.tensor(row, dtype=torch.float64)
                blk_logits.append(row)
            self.cur.forwards.append(Forward(ids[n - B:], blk_logits))
            return SimpleNamespace(logits=logits)

        def _joint_decode_block(self, x, block_start, block_end, *args, **kw):
            self.cur = BlockRec(block_start, block_end, x[0, block_start:block_end].tolist())
            self.blocks.append(self.cur)
            LOG.sink = self.cur.log
            super()._joint_decode_block(x, block_start, block_end, *args, **kw)
            self.cur.final = x[0, block_start:block_end].tolist()

        def _apply_edit_operations_with_tracking(self, block_tokens, old_block_tokens, *args, **kw):
            self.cur.edits_delete += block_tokens.count(DELETE)
            self.cur.edits_insert += block_tokens.count(INSERT)
            self.cur.tracking.append(list(args[0]))   # is_original_mask at this step's start
            return Ref._apply_edit_operations_with_tracking(block_tokens, old_block_tokens, *args, **kw)

        def _diffusion_sample(self, logits, temperature=1.0, top_k=None, top_p=None):
            l2 = logits.reshape(-1, logits.shape[-1])
            if l2.shape[0] and bool(torch.isinf(l2[:, DELETE]).all() and torch.isinf(l2[:, INSERT]).all()):
                self.cur.ds_repicks += l2.shape[0]
            return super()._diffusion_sample(logits, temperature, top_k, top_p)

        def _resample_to_escape_loop(self, block_ids, old_block_ids, *args, **kw):
            seen = args[3]   # (block_logits, mt2_index, t2t_index, seen_block_results, ...)
            if tuple(block_ids.tolist()) in seen:
                self.cur.resamples.append(int((block_ids != old_block_ids).sum()))
            return super()._resample_to_escape_loop(block_ids, old_block_ids, *args, **kw)

    return Scripted


class OneOutcomeDraws:
    """Every random draw in the fixture must have exactly one possible outcome."""

    def __enter__(self):
        self.mult, self.randint = torch.multinomial, torch.randint
        self.draws = 0

        def multinomial(probs, num_samples, *a, **k):
            nz = (probs > 0).sum(-1)
            if bool((nz != 1).any()):
                raise AssertionError(f"a sampled row has {nz.tolist()} candidates: RNG-dependent")
            self.draws += int(probs.reshape(-1, probs.shape[-1]).shape[0])
            return self.mult(probs, num_samples, *a, **k)

        def randint(*a, **k):
            high = a[0] if len(a) == 2 else a[1]
            if high != 1:
                raise AssertionError(f"resample picks among {high} positions: RNG-dependent")
            self.draws += 1
            return self.randint(*a, **k)

        torch.multinomial, torch.randint = multinomial, randint
        return self

    def __exit__(self, *exc):
        torch.multinomial, torch.randint = self.mult, self.randint


# ---- one scenario -------------------------------------------------------------------------
def run_reference(Ref, sc):
    fake = make_fake(Ref)(sc)
    torch.manual_seed(sc.seed)
    with OneOutcomeDraws() as draws:
        out = fake.generate(
            inputs=torch.tensor([sc.prompt], dtype=torch.long),
            temperature=sc.temperature, block_length=sc.block, steps=sc.steps, gen_length=sc.gen,
            top_p=sc.top_p if sc.top_p < 1.0 else None, top_k=sc.top_k or None,
            threshold=sc.threshold, editing_threshold=sc.editing_threshold,
            max_post_steps=sc.max_post_steps, eos_early_stop=sc.eos_early_stop, eos_id=STOP,
            mask_id=MASK, delete_token_id=DELETE, split_token_id=INSERT,
            max_steps_per_block=sc.max_steps_per_block,
        )
    LOG.sink = None
    return fake, out[0].tolist(), draws.draws


def derive_passes(sc, fake):
    """The pass list a cached backend runs for what the reference computed."""
    B, L = sc.block, sc.prompt_len
    passes = []
    for pb in range(L // B):
        passes.append({"kind": "prefill", "rows": [[sc.prompt[p], p, p] for p in range(pb * B, pb * B + B)],
                       "attn": "block", "logit_rows": [], "logits": [], "keep": pb * B + B, "take": []})
    stopped = False
    for bi, rec in enumerate(fake.blocks):
        assert rec.forwards, f"{sc.name}: block@{rec.bs} ended without a forward (not covered)"
        lrows = [i for i, t in enumerate(rec.entry) if t == MASK]   # ~prompt_mask_block
        for k, fw in enumerate(rec.forwards):
            exact = k == len(rec.forwards) - 1 and fw.tokens == rec.final
            passes.append({"kind": "denoise", "rows": [[t, rec.bs + i, rec.bs + i] for i, t in enumerate(fw.tokens)],
                           "attn": "block", "logit_rows": lrows, "logits": [fw.logits[i] for i in lrows],
                           "keep": rec.be if exact else rec.bs, "take": []})
        take = []
        if not stopped:
            for p in range(max(L, rec.bs), min(rec.be, L + sc.gen)):
                take.append(rec.final[p - rec.bs])
                if take[-1] == STOP:
                    stopped = True
                    break
        passes[-1]["take"] = take
        if bi + 1 < len(fake.blocks) and rec.forwards[-1].tokens != rec.final:
            passes.append({"kind": "commit", "rows": [[t, rec.bs + i, rec.bs + i] for i, t in enumerate(rec.final)],
                           "attn": "block", "logit_rows": [], "logits": [], "keep": rec.be, "take": []})
    return passes


def replay_cached(sc, passes):
    """Run the pass list against an emulated KV cache; logits must match the reference's."""
    cache = {}   # slot -> (pos, token, kv1)
    for n, p in enumerate(passes):
        rows = p["rows"]
        base = len(cache)
        assert sorted(cache) == list(range(base)), f"{sc.name} pass {n}: cache has holes"
        assert [r[2] for r in rows] == list(range(base, base + len(rows))), \
            f"{sc.name} pass {n}: rows are not written right after the kept cache"
        assert p["attn"] == "block"
        every = sorted([(pos, tok) for pos, tok, _ in cache.values()] + [(r[1], r[0]) for r in rows])
        k0 = [kv0(tok, pos) for pos, tok in every]
        new = [(r[1], r[0], kv1(r[0], r[1], k0)) for r in rows]
        vis = {pos: (tok, k) for pos, tok, k in list(cache.values()) + new}
        for j, r in enumerate(p["logit_rows"]):
            got = row_logits(sc, rows[r][0], rows[r][1], vis)
            if got != p["logits"][j]:
                raise AssertionError(f"{sc.name} pass {n} row {r}: cached logits differ from the reference's")
        for (pos, tok, k), r in zip(new, rows):
            cache[r[2]] = (pos, tok, k)
        for s in [s for s in cache if s >= p["keep"]]:
            del cache[s]


def sc_sched(sc, rec):
    n = sum(1 for t in rec.entry if t == MASK)
    return [n // sc.steps + (i < n % sc.steps) for i in range(sc.steps)] if sc.steps else []


def softmax64(row):
    a = np.asarray(row, dtype=np.float64)
    e = np.exp(a - a.max())
    return e / e.sum()


def analyse(sc, fake, passes, out, draws):
    """Coverage counters, and the margins that make the fixture exact for a double-precision
    replay. Re-derives only what it needs to classify a step; nothing here feeds the fixture."""
    f = {k: 0 for k in ["threshold_fill", "topk_fill", "t2t", "edit_gated", "delete", "insert",
                        "ds_repick", "resample", "final_round", "final_open_new_mask",
                        "post_with_new_mask", "new_masks", "exit_stable", "exit_post_steps",
                        "exit_max_steps", "commit_pass"]}
    margin = math.inf
    for rec in fake.blocks:
        heads = [dict(kv.split("=") for kv in ln.split()[1:])
                 for lv, ln in rec.log if lv == "debug" and " step=" in ln]
        sched = sc_sched(sc, rec)
        for k, fw in enumerate(rec.forwards):
            new_masks = int(heads[k]["new_masks"])
            post = int(heads[k]["post_steps"].split("/")[0])
            f["new_masks"] += new_masks > 0
            if sc.max_post_steps > 0 and post >= sc.max_post_steps:
                f["final_round"] += 1
            # post-steps count while only masks an edit opened are left (M:1469-1472)
            if sc.max_post_steps > 0 and int(heads[k]["original_masks"]) == 0 and new_masks > 0:
                f["post_with_new_mask"] += 1
            p_top = {}
            for i, t in enumerate(fw.tokens):
                if rec.entry[i] != MASK:
                    continue   # prompt row: no logits
                pr = softmax64(fw.logits[i])
                top = int(np.argmax(pr))
                srt = np.sort(np.asarray(fw.logits[i]))
                margin = min(margin, float(srt[-1] - srt[-2]))           # argmax is unique
                if t == MASK:
                    p_top[i] = float(pr[top])
                    margin = min(margin, abs(pr[top] - sc.threshold))    # M2T gate
                else:
                    margin = min(margin, abs(pr[top] - sc.editing_threshold))  # T2T gate
                    if top != t:
                        f["t2t" if pr[top] > sc.editing_threshold else "edit_gated"] += 1
            s_ = sorted(p_top.values())
            if len(s_) > 1:
                margin = min(margin, min(b - a for a, b in zip(s_, s_[1:])))  # top-k order
            sel = set(p_top)   # past the schedule M2T takes every mask
            if p_top and k < len(sched):
                need = sched[k] + new_masks
                high = [i for i in p_top if p_top[i] > sc.threshold]
                if len(high) < need:
                    f["topk_fill"] += 1
                    sel = set(sorted(p_top, key=lambda i: -p_top[i])[:min(need, len(p_top))])
                else:
                    f["threshold_fill"] += len(high) > 0
                    sel = set(high)
            # max_post_steps == 0: the step whose M2T covers every open original mask is
            # final (M:1563-1564); count the ones that leave a (non-original) mask open,
            # which the final round fills anyway (M:1577)
            orig = [i for i in p_top if rec.tracking[k][i]]
            if sc.max_post_steps == 0 and orig and set(orig) <= sel:
                f["final_round"] += 1
                f["final_open_new_mask"] += len(set(p_top) - sel) > 0
        f["delete"] += rec.edits_delete
        f["insert"] += rec.edits_insert
        f["ds_repick"] += rec.ds_repicks
        if any(n != 1 for n in rec.resamples):
            raise AssertionError(f"{sc.name}: a resample with {rec.resamples} changed positions")
        f["resample"] += len(rec.resamples)
        why = " ".join(ln for _, ln in rec.log if "terminating" in ln)
        if "max_post_steps" in why:
            f["exit_post_steps"] += 1
        elif "max_steps_per_block" in why:
            f["exit_max_steps"] += 1
        else:
            f["exit_stable"] += 1
    f["commit_pass"] = sum(p["kind"] == "commit" for p in passes)
    L, B = sc.prompt_len, sc.block
    f["tail"] = int(L % B != 0 and L >= B)
    f["no_tail"] = int(L % B == 0)
    f["short_prompt"] = int(L < B)
    f["stop"] = int(STOP in out)
    num_blocks = (L + sc.gen + B - 1) // B
    f["early_stop"] = int(len(fake.blocks) < num_blocks - L // B)
    stop_block = next((r.bs for r in fake.blocks if STOP in r.final[max(0, L - r.bs):]), None)
    f["decoded_after_stop"] = int(stop_block is not None and fake.blocks[-1].bs > stop_block)
    f["truncated"] = int(STOP not in out and len(out) == sc.gen and (L + sc.gen) % B != 0)
    f["residual_mask"] = int(MASK in out)
    f["sampled"] = int(sc.temperature > 0 and draws > 0)
    f["rng_draws"] = draws
    if margin < 1e-6:
        raise AssertionError(f"{sc.name}: a decision lies within {margin:g} of its boundary")
    missing = [e for e in sc.expect if not f.get(e)]
    if missing:
        msg = f"{sc.name}: does not exercise {missing}"
        if not TUNING:
            raise AssertionError(msg + f"; features {f}")
        print("  !! " + msg)
    return f, margin


def compact(v):
    return int(v) if float(v).is_integer() else v


def build(ref_path: Path):
    Ref, info = load_reference(ref_path)
    out_scen = []
    for sc in SCENARIOS:
        fake, ref_out, draws = run_reference(Ref, sc)
        passes = derive_passes(sc, fake)
        streamed = [t for p in passes for t in p["take"]]
        if streamed != ref_out:
            raise AssertionError(f"{sc.name}: streamed {streamed} != generate() {ref_out}")
        replay_cached(sc, passes)
        feats, margin = analyse(sc, fake, passes, ref_out, draws)
        for p in passes:
            p["logits"] = [[compact(v) for v in row] for row in p["logits"]]
        out_scen.append({
            "name": sc.name, "what": sc.what, "config": sc.config(), "prompt": sc.prompt,
            "passes": passes, "output": ref_out,
            "decode_passes": sum(p["kind"] != "prefill" for p in passes),
            "features": feats,
        })
        print(f"{sc.name:26s} blocks {len(fake.blocks)}  passes {len(passes):3d} "
              f"(commit {feats['commit_pass']})  out {len(ref_out):2d}  margin {margin:.2e}  "
              + " ".join(f"{k}={v}" for k, v in feats.items()
                         if v and k not in ("tail", "no_tail", "short_prompt", "rng_draws")))
    return {
        "traces": "OPEN-DLLM-LLADA2",
        "generator": "utilities/dllm-ref/ref_llada2.py",
        "reference": info,
        "vocab": V,
        "scenarios": out_scen,
    }


def dump(doc):
    c = lambda x: json.dumps(x, separators=(",", ":"))
    lines = ["{"]
    head = {k: v for k, v in doc.items() if k != "scenarios"}
    for k, v in head.items():
        lines.append(f"  {c(k)}: {c(v)},")
    lines.append('  "scenarios": [')
    for si, s in enumerate(doc["scenarios"]):
        lines.append("    {")
        for k in ["name", "what", "config", "prompt", "output", "decode_passes", "features"]:
            lines.append(f"      {c(k)}: {c(s[k])},")
        lines.append('      "passes": [')
        for pi, p in enumerate(s["passes"]):
            lines.append("        " + c(p) + ("," if pi + 1 < len(s["passes"]) else ""))
        lines.append("      ]")
        lines.append("    }" + ("," if si + 1 < len(doc["scenarios"]) else ""))
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, type=Path,
                    help="modeling_llada2_moe.py from hf.co/inclusionAI/LLaDA2.2-mini, or its directory")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tuning", action="store_true",
                    help="report scenarios that miss their expected features instead of failing")
    a = ap.parse_args()
    global TUNING
    TUNING = a.tuning
    text = dump(build(a.ref))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {a.out} ({len(text.encode()):,} bytes, sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]})")


if __name__ == "__main__":
    sys.exit(main())
