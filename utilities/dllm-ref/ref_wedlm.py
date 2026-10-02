#!/usr/bin/env python3
"""ref_wedlm.py: the WeDLM reference's own decode loop, driven by a scripted fake model.

Writes specs/open-engine/tests/fixtures/dllm_wedlm.json, which src/open_dllm/wedlm_test.cpp
replays against make_wedlm(). Traces: OPEN-DLLM-WEDLM (canonical spec:
specs/open-engine/spec.md; plan: .claude/plans/diffusion-lm-support.md).

What runs is tencent/WeDLM's code, not a re-implementation of it: LLMEngine.generate and
step, Scheduler, BlockManager, ModelRunner.run / prepare_prefill / run_model /
_wedlm_decode_one_step, WeDLMDecoder and Sampler, loaded from a clone of the repo. Three
things are swapped, and nothing else:

  * the model. ScriptedModel stands in for WeDLMForDiffusionLM. It reads the attention
    context the real Attention layer reads (wedlm/layers/attention.py), writes every row
    into an emulated KV cache at its slot first, and gives each row logits that depend on
    the row's logical position, its token and a seeded hash of exactly the (token,
    position) pairs that row can see. A wrong slot, position or mask changes the logits;
  * device placement: "cuda" becomes "cpu" and pinned memory is off (textual patches on
    the loaded source, each one asserted to apply);
  * ModelRunner.__init__, which joins NCCL, loads weights and captures CUDA graphs.

Between passes the harness also drops every emulated cache row at or above the
committed length, which is what policy.hpp asks a backend to do. If the reference ever
read such a row again without rewriting it first, the scripted model would fail.

Greedy (temperature 0) only: a sampled run draws from torch's RNG, which the C++ policy's
mt19937_64 cannot reproduce. Every threshold and argmin decision is re-made in float64
and must agree with the reference's float32 one by a margin, so the C++ side (double)
cannot land on the other side of a comparison.

Usage:
    python utilities/dllm-ref/ref_wedlm.py --ref <clone of github.com/tencent/WeDLM>
        [--out specs/open-engine/tests/fixtures/dllm_wedlm.json]
    python utilities/dllm-ref/ref_wedlm.py --ref <clone> --search   # find seeds that cover
"""

import argparse
import contextlib
import hashlib
import io
import json
import re
import struct
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "specs" / "open-engine" / "tests" / "fixtures" / "dllm_wedlm.json"

# ---- the scripted vocabulary ---------------------------------------------------------------
V = 40
MASK = 39            # set as the engine's mask_token_id, like WeDLM's 151665
STOPS = [38, 37]     # stop_token_ids; the model emits both
NORMAL = 37          # ordinary tokens are 0..36

# Each row's logits are one of five templates: a peak at the predicted token over a fixed
# multiset of background values, shuffled by the row's hash. Rows of one class therefore
# have the same entropy (to float rounding), which is what makes "a tie broken by distance"
# a real event rather than a coincidence. Values are quarter-integers so the JSON is short
# and exact. Entropies (nats, V = 40): 0.0438, 0.3110, 0.6918, 1.3825, 2.7312.
PEAK = {"sharp": 8.5, "medium": 6.25, "mid": 5.25, "flat_lo": 4.25, "flat_hi": 2.75}
CLASSES = ["sharp", "medium", "mid", "flat_lo", "flat_hi"]
BACKGROUND = [-(i % 4) * 0.5 + 0.0 for i in range(V - 1)]


@dataclass
class Scenario:
    name: str
    what: str
    window: int
    tau: float | None          # None is the reference's one-token-per-pass mode
    lam: float
    prompt_len: int
    max_new: int
    seed: int
    s0: float                  # P(sharp) at the leftmost mask ...
    slope: float               # ... falling this much per slot of distance ...
    floor: float               # ... to this
    p_medium: float
    p_mid: float
    p_flat_lo: float           # the rest is flat_hi
    stop_after: int = 10**9    # generated index from which the model may predict a stop
    p_stop: float = 0.0
    p_maskpred: float = 0.0    # the model predicts the MASK id itself (filled, not a mask)
    # Every row is one of two fixed, unshuffled vectors: flat_lo predicting DUP_LO (chance
    # s0) or flat_hi predicting DUP_HI. Identical rows have bit-identical entropies in
    # float32 and float64 alike, so with lambda 0 the argmin meets exact ties and the order
    # it breaks them in (torch takes the first) is tested rather than excluded.
    dup_rows: bool = False
    requires: list = field(default_factory=list)


DUP_LO, DUP_HI = 5, 6


SCENARIOS = [
    Scenario("defaults", "W 16, tau 0.4, lambda 0.02 (the reference's defaults); several fills "
             "per pass; ends on a stop token committed mid-window, with filled tokens after it",
             window=16, tau=0.4, lam=0.02, prompt_len=6, max_new=64, seed=0,
             s0=0.9, slope=0.08, floor=0.15, p_medium=0.3, p_mid=0.1, p_flat_lo=0.2,
             stop_after=18, p_stop=0.12,
             requires=["multi_fill", "stop_mid_window", "threshold_split_by_distance"]),
    Scenario("one_at_a_time", "W 4, tau None (one fill per pass, the paper's quality setting); "
             "a one-token prompt; argmin ties broken by distance; ends on max_new_tokens",
             window=4, tau=None, lam=0.02, prompt_len=1, max_new=13, seed=0,
             s0=0.3, slope=0.05, floor=0.1, p_medium=0.2, p_mid=0.1, p_flat_lo=0.3,
             requires=["fallback_not_first", "tie_by_distance", "end_max"]),
    Scenario("max_mid_commit", "W 8, tau 0.6, lambda 0.05; the max_new_tokens limit falls inside "
             "one pass's commit, so the commit is cut short; passes that fill exactly one",
             window=8, tau=0.6, lam=0.05, prompt_len=9, max_new=17, seed=3,
             s0=0.85, slope=0.1, floor=0.05, p_medium=0.25, p_mid=0.1, p_flat_lo=0.3,
             requires=["max_mid_commit", "single_fill_with_tau", "end_max"]),
    Scenario("aggressive", "W 8, tau 0.9, lambda 0.05; a fully filled window (2W rows in one "
             "pass); the model predicts the MASK id itself, which is committed as an ordinary token",
             window=8, tau=0.9, lam=0.05, prompt_len=12, max_new=40, seed=1,
             s0=0.95, slope=0.04, floor=0.3, p_medium=0.3, p_mid=0.2, p_flat_lo=0.1,
             stop_after=30, p_stop=0.1, p_maskpred=0.05,
             requires=["full_window", "mask_id_committed", "multi_fill"]),
    Scenario("exact_ties", "W 6, tau None, lambda 0; mask rows with bit-identical logits, so the "
             "lowest adjusted entropy is an exact tie and the first such mask must win",
             window=6, tau=None, lam=0.0, prompt_len=3, max_new=12, seed=0,
             s0=0.3, slope=0.0, floor=0.0, p_medium=0.0, p_mid=0.0, p_flat_lo=0.0, dup_rows=True,
             requires=["exact_tie_first", "fallback_not_first", "end_max"]),
]

# ---- loading the reference -----------------------------------------------------------------

# (regex, replacement) per file; each must match at least once.
_TO_HOST = (r"\.cuda\(\s*non_blocking=True\s*\)", "")
CPU_PATCHES = {
    "engine/wedlm_decoder.py": [(r'device="cuda"', 'device="cpu"'),
                                (r'torch\.device\("cuda"\)', 'torch.device("cpu")'),
                                (r"pin_memory=True", "pin_memory=False"), _TO_HOST],
    "engine/model_runner.py": [(r"pin_memory=True", "pin_memory=False"), _TO_HOST],
}


def load_reference(ref: Path) -> SimpleNamespace:
    """Imports the reference modules from `ref`, patched for CPU only (see CPU_PATCHES)."""
    root = ref / "wedlm"
    if not (root / "engine" / "wedlm_decoder.py").is_file():
        raise SystemExit(f"{ref} is not a clone of github.com/tencent/WeDLM")

    def package(name, path):
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m

    package("wedlm", root)  # not wedlm/__init__.py: it imports the whole engine eagerly
    for sub in ("engine", "utils", "models", "layers"):
        package(f"wedlm.{sub}", root / sub)
    # The real model and weight loader pull in flash_attn and triton. ModelRunner imports
    # these names but only uses them in the __init__ this harness bypasses.
    stub = types.ModuleType("wedlm.models.wedlm")
    stub.WeDLMForDiffusionLM = None
    sys.modules[stub.__name__] = stub
    stub = types.ModuleType("wedlm.utils.loader")
    stub.load_model = None
    sys.modules[stub.__name__] = stub

    def load(name, rel):
        path = root / rel
        src = path.read_text(encoding="utf-8")
        for pattern, new in CPU_PATCHES.get(rel, []):
            src, hits = re.subn(pattern, new, src)
            if not hits:
                raise SystemExit(f"{rel}: expected {pattern!r}; has the reference changed?")
        mod = types.ModuleType(name)
        mod.__file__ = str(path)
        sys.modules[name] = mod
        exec(compile(src, str(path), "exec"), mod.__dict__)
        return mod

    ns = SimpleNamespace()
    ns.sampling_params = load("wedlm.sampling_params", "sampling_params.py")
    load("wedlm.config", "config.py")
    ns.context = load("wedlm.utils.context", "utils/context.py")
    ns.sequence = load("wedlm.engine.sequence", "engine/sequence.py")
    ns.sampler = load("wedlm.engine.sampler", "engine/sampler.py")
    load("wedlm.engine.block_manager", "engine/block_manager.py")
    ns.decoder = load("wedlm.engine.wedlm_decoder", "engine/wedlm_decoder.py")
    ns.scheduler = load("wedlm.engine.scheduler", "engine/scheduler.py")
    ns.model_runner = load("wedlm.engine.model_runner", "engine/model_runner.py")
    ns.llm_engine = load("wedlm.engine.llm_engine", "engine/llm_engine.py")
    try:
        ns.commit = subprocess.run(["git", "-C", str(ref), "rev-parse", "HEAD"], capture_output=True,
                                   text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        ns.commit = "unknown"
    return ns


# ---- the scripted model --------------------------------------------------------------------

def uniforms(digest: bytes):
    return [int.from_bytes(digest[i:i + 4], "little") / 2**32 for i in range(0, len(digest), 4)]


class ScriptedModel:
    """Stands in for WeDLMForDiffusionLM: `model(input_ids, positions)` gives the 'hidden
    states' and compute_logits() is the identity, so the reference's run_model is unchanged."""

    def __init__(self, sc: Scenario, ref, block_table):
        self.sc = sc
        self.ref = ref
        self.block_table = block_table   # callable: the running sequence's block table
        self.cache = {}                  # physical slot -> (token, pos): the emulated K/V
        self.passes = []

    def logical(self, phys, bs):
        table = self.block_table()
        return table.index(phys // bs) * bs + phys % bs

    def __call__(self, input_ids, positions):
        c = self.ref.context.get_context()
        bs = self.ref.sequence.Sequence.block_size
        ids, pos = input_ids.tolist(), positions.tolist()
        slots = c.slot_mapping.tolist()
        n = len(ids)
        # Attention.forward stores every row's K/V at its slot before attending (attention.py:87-90).
        for t, p, s in zip(ids, pos, slots):
            assert s != -1
            self.cache[s] = (t, p)
        if c.is_prefill:
            # One fresh sequence: no prefix-cache hit, so flash-attn runs causal over the rows.
            assert c.block_tables is None and c.cu_seqlens_q.tolist() == [0, n]
            visible = [[(ids[i], pos[i]) for i in range(j + 1)] for j in range(n)]
        else:
            q = int(c.per_seq_wedlm_sizes[0])
            assert q == n and c.context_lens.numel() == 1
            k = int(c.context_lens[0]) + q
            table = c.block_tables[0].tolist()
            phys = lambda l: table[l // bs] * bs + l % bs
            # flash-attn aligns the causal mask bottom-right when k > q: query j sees keys 0..k-q+j.
            visible = [[self.cache[phys(l)] for l in range(k - q + j + 1)] for j in range(n)]
        logits = torch.tensor([self.row_logits(ids[j], pos[j], visible[j]) for j in range(n)],
                              dtype=torch.float32)
        self.passes.append({
            "rows": [[ids[j], pos[j], self.logical(slots[j], bs)] for j in range(n)],
            "attn": "causal",
            "logit_rows": [], "logits": [],
            "_all": logits, "_prefill": bool(c.is_prefill)})
        return logits

    def compute_logits(self, hidden):
        return hidden

    def row_logits(self, tok, pos, visible):
        sc = self.sc
        # Attention is a weighted sum over keys, so what a row sees is the SET of (token,
        # position) pairs, not their slot order.
        keys = sorted((p, t) for t, p in visible)
        h = hashlib.blake2b(digest_size=32, key=struct.pack("<q", sc.seed))
        h.update(struct.pack("<ii", tok, pos))
        for p, t in keys:
            h.update(struct.pack("<ii", p, t))
        d = h.digest()
        u = uniforms(d)
        if sc.dup_rows:
            lo = u[0] < sc.s0
            row = [0.0] * V
            row[DUP_LO if lo else DUP_HI] = PEAK["flat_lo" if lo else "flat_hi"]
            for v, b in zip([v for v in range(V) if v != (DUP_LO if lo else DUP_HI)], BACKGROUND):
                row[v] = b
            return row
        # Confidence falls with distance from the first position this row cannot see filled:
        # for a mask row that is its distance from the window's first mask.
        known = {p for p, t in keys if t != MASK}
        frontier = 0
        while frontier in known:
            frontier += 1
        gap = max(0, pos - frontier)
        ps = max(sc.floor, sc.s0 - sc.slope * gap)
        rest = 1.0 - ps
        cuts = [ps, ps + rest * sc.p_medium, ps + rest * (sc.p_medium + sc.p_mid),
                ps + rest * (sc.p_medium + sc.p_mid + sc.p_flat_lo)]
        cls = CLASSES[sum(u[0] >= x for x in cuts)]
        gen = pos - sc.prompt_len
        if gen >= sc.stop_after and u[1] < sc.p_stop:
            target = STOPS[int(u[2] * len(STOPS))]
        elif u[1] > 1.0 - sc.p_maskpred:
            target = MASK
        else:
            target = int(u[3] * NORMAL)
        others = [v for v in range(V) if v != target]
        others.sort(key=lambda v: hashlib.blake2b(d + struct.pack("<i", v), digest_size=8).digest())
        row = [0.0] * V
        row[target] = PEAK[cls]
        for v, b in zip(others, BACKGROUND):
            row[v] = b
        return row


# ---- one scenario through the reference ----------------------------------------------------

def entropy64(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(p), 0.0)
    return -t.sum(axis=1)


MARGIN = 1e-4


def run_scenario(ref, sc: Scenario):
    Sequence = ref.sequence.Sequence
    Sequence.block_size = 256  # kvcache_block_size must be a multiple of 256 (config.py:76)
    cfg = SimpleNamespace(kvcache_block_size=256, wedlm_window_size=sc.window, enforce_eager=True,
                          tensor_parallel_size=1, mask_token_id=MASK, max_num_seqs=1,
                          max_num_batched_tokens=16384, num_kvcache_blocks=8, max_model_len=4096)

    engine = ref.llm_engine.LLMEngine.__new__(ref.llm_engine.LLMEngine)
    engine.scheduler = ref.scheduler.Scheduler(cfg)
    engine.eos_token_id = None  # stop ids are exactly the scenario's
    engine.tokenizer = SimpleNamespace(decode=lambda ids, **kw: "", encode=lambda s, **kw: [])

    model = ScriptedModel(sc, ref, lambda: list(engine.scheduler.running[0].block_table))

    class ScriptedRunner(ref.model_runner.ModelRunner):
        def __init__(self):
            # ModelRunner.__init__ (model_runner.py:65-89) minus NCCL, weights and CUDA graphs.
            self.config = cfg
            self.block_size = cfg.kvcache_block_size
            self.wedlm_window_size = cfg.wedlm_window_size
            self.enforce_eager = True
            self.world_size = 1
            self.rank = 0
            self.event = []
            self.model = model
            self.sampler = ref.sampler.Sampler()
            self._init_mask_token(cfg, SimpleNamespace())
            self._init_wedlm_decoder()

    runner = ScriptedRunner()
    engine.model_runner = runner
    dec = runner.wedlm_decoder
    feats = set()

    # Spies: they record what the reference did and change nothing.
    orig_prune = dec._process_pruned_tokens

    def prune(seq, state, k):
        model.passes[-1]["_window"] = (list(state.window_tokens), list(state.window_mask_flags), k,
                                       state.current_seq_len - seq.num_prompt_tokens)
        return orig_prune(seq, state, k)

    dec._process_pruned_tokens = prune

    orig_select = dec.sampler.process_mask_positions

    def select(**kw):
        rec = model.passes[-1]
        ml, rmi = kw["mask_logits"], kw["remaining_mask_indices"]
        tau, lam = kw["entropy_threshold"], kw["pos_penalty_factor"]
        fill, toks = orig_select(**kw)
        n, m = rec["_all"].shape[0], ml.shape[0]
        assert ml.data_ptr() == rec["_all"][n - m].data_ptr(), "mask logits are not the last rows"
        rec["logit_rows"] = list(range(n - m, n))
        rec["logits"] = [[int(v) if v == int(v) else v for v in r] for r in ml.tolist()]
        # Re-make the decision in float64 and require a margin, so double-precision C++
        # cannot fall on the other side of any comparison the reference made in float32.
        h = entropy64(ml.numpy())
        dist = np.array(rmi, dtype=np.float64) - rmi[0]
        adj = h + lam * dist
        sel = list(np.nonzero(adj < tau)[0]) if tau is not None else []
        if tau is not None and np.min(np.abs(adj - tau)) < MARGIN:
            raise RuntimeError(f"{sc.name}: an adjusted entropy sits within {MARGIN} of tau")
        fallback = not sel
        if fallback:
            best = adj.min()
            tied = [i for i in range(m) if adj[i] == best]
            rest = adj[adj != best]
            if rest.size and rest.min() - best < MARGIN:
                raise RuntimeError(f"{sc.name}: argmin decided by less than {MARGIN}")
            if len(tied) > 1:
                # An exact tie is only safe between bit-identical rows: then the reference's
                # float32 values tie too (checked), and so will the C++ doubles.
                f32 = dec.sampler.compute_entropy(ml) + (torch.tensor(rmi, dtype=torch.float) - rmi[0]) * lam
                if not all(torch.equal(ml[i], ml[tied[0]]) and f32[i] == f32[tied[0]] for i in tied):
                    raise RuntimeError(f"{sc.name}: an exact argmin tie between different rows")
                feats.add("exact_tie_first")
            sel = [tied[0]]
        if [int(x) for x in sel] != list(fill):
            raise RuntimeError(f"{sc.name}: float64 picks {sel}, the reference picked {fill}")
        assert toks == [int(np.argmax(ml[k].numpy())) for k in fill]
        # Coverage.
        if len(fill) >= 3:
            feats.add("multi_fill")
        if len(fill) == 1 and tau is not None:
            feats.add("single_fill_with_tau")
        if fallback and fill[0] > 0:
            feats.add("fallback_not_first")
        same = lambda a, b: abs(h[a] - h[b]) < 1e-9
        if fallback:
            k = fill[0]
            if any(same(j, k) for j in range(k + 1, m)) and all(h[j] >= h[k] - 1e-9 for j in range(m)):
                feats.add("tie_by_distance")
        if tau is not None:
            for a in fill:
                if any(same(a, b) and b not in fill for b in range(m)):
                    feats.add("threshold_split_by_distance")
        return fill, toks

    dec.sampler.process_mask_positions = select

    taken_so_far = 0
    orig_post = engine.scheduler.postprocess

    def post(seqs, outs):
        nonlocal taken_so_far
        (seq,) = seqs
        table = list(seq.block_table)
        orig_post(seqs, outs)
        rec = model.passes[-1]
        st = seq.wedlm_state
        rec["keep"] = st.current_seq_len if st is not None else len(seq)
        comp = seq.completion_token_ids
        rec["taken"] = list(comp[taken_so_far:])
        taken_so_far = len(comp)
        if seq.is_finished:
            return
        # The backend contract: drop every cache row at or above keep().
        for s in [s for s in model.cache if table.index(s // 256) * 256 + s % 256 >= rec["keep"]]:
            del model.cache[s]

    engine.scheduler.postprocess = post

    prompt = [int(u * NORMAL) for u in uniforms(hashlib.blake2b(f"prompt-{sc.name}".encode(),
                                                                digest_size=64).digest())][:sc.prompt_len]
    sp = ref.sampling_params.SamplingParams(
        temperature=0.0, top_p=1.0, top_k=0, max_tokens=sc.max_new, stop_token_ids=list(STOPS),
        wedlm_entropy_threshold=sc.tau, wedlm_pos_penalty_factor=sc.lam)
    with contextlib.redirect_stdout(io.StringIO()):  # generate() prints a [Stats] line
        (result,) = engine.generate([prompt], sp, use_tqdm=False)
    output = list(result["token_ids"])

    # End-of-run coverage, from the recorded windows.
    passes = model.passes
    assert passes[0]["_prefill"] and not any(p["_prefill"] for p in passes[1:])
    last = passes[-1]
    win, flags, k, gen_before = last["_window"]
    if output and output[-1] in STOPS:
        feats.add("end_stop")
        i = win.index(output[-1])
        if any(not f for f in flags[i + 1:]):
            feats.add("stop_mid_window")
    if len(output) == sc.max_new and output[-1] not in STOPS:
        feats.add("end_max")
        if gen_before + k > sc.max_new:
            feats.add("max_mid_commit")
    if any(len(p["rows"]) == 2 * sc.window for p in passes[1:]):
        feats.add("full_window")
    if MASK in output:
        feats.add("mask_id_committed")
    assert last["logit_rows"] == [], "the finishing pass consumed logits"
    assert [t for p in passes for t in p["taken"]] == output
    # llm_engine.py:165 counts active sequences after each step, so the pass that finishes
    # the request is not in decode_forwards.
    assert result["stats"]["decode_forwards"] == len(passes) - 2

    return {
        "name": sc.name,
        "what": sc.what,
        "config": {"window": sc.window, "tau": sc.tau, "lambda": sc.lam, "mask_id": MASK,
                   "stop_ids": list(STOPS), "max_new_tokens": sc.max_new, "temperature": 0.0,
                   "top_k": 0, "top_p": 1.0, "seed": 0},
        "prompt": prompt,
        "passes": [{k: v for k, v in p.items() if not k.startswith("_")} for p in passes],
        "output": output,
        "decode_passes": len(passes) - 1,
        "reference_decode_forwards": result["stats"]["decode_forwards"],
        "covers": sorted(feats),
    }, feats


def dump(fixture) -> str:
    """One pass per line: small, and diffs read pass by pass."""
    c = lambda x: json.dumps(x, separators=(",", ":"))
    out = ["{"]
    for key in ("generator", "reference", "note", "vocab"):
        out.append(f' {c(key)}: {c(fixture[key])},')
    out.append(' "scenarios": [')
    for si, s in enumerate(fixture["scenarios"]):
        out.append("  {")
        for key in ("name", "what", "config", "prompt", "output", "decode_passes",
                    "reference_decode_forwards", "covers"):
            out.append(f'   {c(key)}: {c(s[key])},')
        out.append('   "passes": [')
        out.append(",\n".join("    " + c(p) for p in s["passes"]))
        out.append("   ]")
        out.append("  }" + ("," if si + 1 < len(fixture["scenarios"]) else ""))
    out.append(" ]")
    out.append("}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, type=Path, help="clone of github.com/tencent/WeDLM")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--search", action="store_true",
                    help="print, per scenario, the first seeds whose run covers its `requires`")
    args = ap.parse_args()
    torch.set_num_threads(1)
    ref = load_reference(args.ref)

    if args.search:
        for sc in SCENARIOS:
            found = []
            for seed in range(400):
                sc.seed = seed
                try:
                    _, feats = run_scenario(ref, sc)
                except RuntimeError:
                    continue
                if set(sc.requires) <= feats:
                    found.append(seed)
                    if len(found) == 5:
                        break
            print(f"{sc.name}: {found}")
        return

    scenarios = []
    for sc in SCENARIOS:
        s, feats = run_scenario(ref, sc)
        missing = set(sc.requires) - feats
        if missing:
            raise SystemExit(f"{sc.name} (seed {sc.seed}) no longer covers {sorted(missing)}")
        scenarios.append(s)
    fixture = {
        "generator": "utilities/dllm-ref/ref_wedlm.py",
        "reference": {"repo": "https://github.com/tencent/WeDLM", "commit": ref.commit},
        "note": "Each pass: rows [token, pos, slot]; logits are those of logit_rows, in order; keep "
                "and taken are read after feed. tau null is the reference's one-fill-per-pass mode. "
                "reference_decode_forwards is llm_engine's count, which leaves out the finishing pass.",
        "vocab": V,
        "scenarios": scenarios,
    }
    text = dump(fixture)
    json.loads(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(text.encode("utf-8"))
    print(f"wrote {args.out} ({len(text.encode('utf-8'))} bytes, sha256 "
          f"{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]})")
    for s in scenarios:
        print(f"  {s['name']}: {len(s['passes'])} passes, {len(s['output'])} tokens, covers {s['covers']}")


if __name__ == "__main__":
    main()
