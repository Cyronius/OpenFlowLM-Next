#!/usr/bin/env python3
"""Golden traces for the SDAR decode policy (OPEN-DLLM-SDAR).

Runs JetAstra/SDAR generate.py's own block_diffusion_generate -- vendored below, byte for
byte -- against a scripted fake model, and writes every forward pass it makes, in the Pass
vocabulary of src/open_dllm/policy.hpp, to specs/open-engine/tests/fixtures/dllm_sdar.json.
src/open_dllm/sdar_test.cpp replays that file against make_sdar() and requires every pass
to match exactly.

    python utilities/dllm-ref/ref_sdar.py [--out PATH]
    python utilities/dllm-ref/ref_sdar.py --check-sources DIR
        DIR holds the downloaded reference as gh_generate.py and gh_huggingface_bd3.py;
        this proves the vendored text below is identical to them.

What is swapped into generate.py's loop, and nothing else:

* The model: FakeSDAR. A row's logits depend on its position, its token, and a seeded hash
  of every (position, token) it can see -- read from the attention mask it is handed and
  from its own record of what the store_kv=True calls wrote. A policy that shows a row the
  wrong context gets different logits and diverges.
* For the greedy scenarios, the sampler. generate.py has no T=0 path (it divides by the
  temperature, generate.py:35-36). The published greedy numbers come from the OpenCompass
  wrapper at T=1, top_k=1 (eval_sdar_hf.py:115-117), whose sampler reads the confidence
  BEFORE the top-k filter (huggingface_bd3.py:75,90): softmax(logits)[argmax]. That
  sampler is vendored too and stands in for generate.py's; make_sdar's T=0 is the same
  thing. The gh_topk1 scenario keeps generate.py's own sampler at T=1, top_k=1, where the
  confidence is taken AFTER the filter and is therefore 1.0 everywhere.

And one guard. generate.py's top-k takes sched[step] rows even when fewer MASKs are left,
so it also picks -inf rows -- the prompt's tail, tokens already filled -- and overwrites
them with their x0 (generate.py:164-166,177-179,196). The wrapper replaces x0 with the
current token on non-MASK rows (huggingface_bd3.py:233); that is applied here by wrapping
the sampler, so generate.py's text stays untouched. make_sdar clamps k to the MASKs left,
which selects the same rows. Each scenario reports how often the guard mattered and whether
the unguarded generate.py would have answered differently.

Greedy only: every scenario's draws are forced (argmax, or a one-hot distribution), so the
torch RNG and make_sdar's mt19937_64 never have to agree. The generator refuses a fixture
whose confidences sit within 1e-6 of tau or tie at a top-k cut, so the C++ side's double
softmax cannot land on the other side of a decision.

Plan: specs/open-engine/plans/diffusion-lm.md.
"""
import argparse
import json
import hashlib
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
DEFAULT_OUT = os.path.join(REPO, 'specs', 'open-engine', 'tests', 'fixtures', 'dllm_sdar.json')

# ---- third-party notices for the vendored text below ------------------------------------
# generate.py is from https://github.com/JetAstra/SDAR, under its MIT License:
#
#   MIT License
#
#   Copyright (c) 2025 JetAstra
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#
# The sampler from huggingface_bd3.py comes from the same repository's OpenCompass tree
# (evaluation/opencompass/; OpenCompass is Apache-2.0,
# https://github.com/open-compass/opencompass). Both are copied unchanged, which
# --check-sources verifies.

# ---- the reference, verbatim ------------------------------------------------------------
# JetAstra/SDAR generate.py (git blob a24d2a824bf2b3e4469d6404acf0d5a0f256fb73), lines
# 1-202: everything but the argument parsing and __main__.
GH_GENERATE_FILE = 'gh_generate.py'
GH_GENERATE_LINES = (1, 202)
GH_GENERATE_SRC = r'''import argparse
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def top_k_logits(logits, k):
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)     # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


@torch.no_grad()
def block_diffusion_generate(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        eb_threshold=None,
        stopping_criteria_idx=None
    ):

    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:,
                                                       :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache
                model(cur_x,
                      attention_mask=cur_attn_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
                break

            # Denosing
            logits = model(cur_x,
                           attention_mask=cur_attn_mask,
                           position_ids=cur_position_ids,
                           past_key_values=past_key_values,
                           use_cache=True,
                           store_kv=False).logits

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Sampling strategy
            if remasking_strategy == 'sequential':
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        first_mask_index = mask_index[j].nonzero(as_tuple=True)[
                            0].min().item()
                        transfer_index[j, first_mask_index:first_mask_index +
                                       num_transfer_tokens[step]] = True
                    else:
                        raise ValueError(
                            "No mask tokens found in the current block.")

            elif remasking_strategy == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(
                        confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        _, idx = torch.topk(
                            confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
            elif remasking_strategy == "entropy_bounded":
                eps = 1e-12
                entropies = -(x0_p.clamp_min(eps) * (x0_p.clamp_min(eps)).log()).sum(dim=-1)
                entropies = torch.where(mask_index, entropies, torch.inf)
                ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
                cumsum = torch.cumsum(ent_sorted, dim=1)
                for j in range(x0_p.shape[0]):
                    k = torch.searchsorted(cumsum[j], torch.tensor(eb_threshold, device=x0_p.device), right=False).item()
                    k = max(1, min(k, int(mask_index[j].sum().item())))
                    selected_token_indices = order[j, :k]
                    transfer_index[j, selected_token_indices] = True
                
            else:
                raise ValueError(
                    f"Unknown remasking strategy: {remasking_strategy}")

            cur_x[transfer_index] = x0[transfer_index]

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return x
'''

# JetAstra/SDAR evaluation/opencompass/opencompass/models/huggingface_bd3.py (git blob
# 11e5c7cd379edaee5d518be62d797ccc0e8bd720), lines 66-92: the wrapper's sampler. Its
# top_k_logits / top_p_logits (lines 43-64) are generate.py's plus comments; generate.py's
# are passed in.
WRAPPER_FILE = 'gh_huggingface_bd3.py'
WRAPPER_SAMPLER_LINES = (66, 92)
WRAPPER_SAMPLER_SRC = r'''def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    # 1️⃣ 原始概率：先应用温度，然后 softmax
    assert temperature > 0, "Temperature must be positive"
    logits = logits / temperature if temperature != 1.0 else logits
    ori_probs = F.softmax(logits, dim=-1)  # 用于置信度排序

    # 2️⃣ 再做 top-k / top-p 筛选用于采样
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    assert probs.dim() == 2

    # 3️⃣ multinomial 采样
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]

    # 4️⃣ 获取 token 在原始概率分布里的概率
    token_prob = torch.gather(ori_probs, -1, token)

    return token.view(*orig_shape), token_prob.view(*orig_shape)
'''


def load_reference():
    """A fresh namespace holding generate.py's functions, plus the wrapper's sampler."""
    ns = {'__name__': 'sdar_generate'}
    exec(compile(GH_GENERATE_SRC, 'generate.py', 'exec'), ns)
    wrapper = {'torch': torch, 'F': ns['F'],
               'top_k_logits': ns['top_k_logits'], 'top_p_logits': ns['top_p_logits']}
    exec(compile(WRAPPER_SAMPLER_SRC, 'huggingface_bd3.py', 'exec'), wrapper)
    ns['gh_sampler'] = ns['sample_with_temperature_topk_topp']
    ns['wrapper_sampler'] = wrapper['sample_with_temperature_topk_topp']
    return ns


def check_sources(d):
    for fname, src, (a, b) in ((GH_GENERATE_FILE, GH_GENERATE_SRC, GH_GENERATE_LINES),
                               (WRAPPER_FILE, WRAPPER_SAMPLER_SRC, WRAPPER_SAMPLER_LINES)):
        with open(os.path.join(d, fname), encoding='utf-8', newline='') as f:
            lines = f.read().splitlines(keepends=True)
        if ''.join(lines[a - 1:b]) != src:
            sys.exit(f'FAIL vendored {fname} lines {a}-{b} differ from {d}')
        print(f'ok   vendored {fname} lines {a}-{b} are identical to {os.path.join(d, fname)}')


# ---- the fake model ---------------------------------------------------------------------
VOCAB = 40
MASK = 39
STOPS = [37, 38]      # two stop ids, as SDAR has <|im_end|> and <|endoftext|>
N_REGULAR = 37        # tokens 0..36 are ordinary text
LOGIT_SCALE = 0.125   # logits are integers in units of 1/8: exact in fp32, short in JSON


class FakeSDAR:
    """Stands in for SDARForCausalLM in generate.py's calls.

    Per row: 40 noise logits in [-1, 1], the MASK token at -8, and one preferred token
    lifted 1.0 + m above the noise ceiling -- 'sure' (m 6.5-9, greedy confidence ~0.97+)
    with probability p_sure, else 'unsure' (m 1-4, confidence ~0.14-0.8). Rows at
    position >= stop_from prefer a stop id with probability p_stop. All of it is drawn
    from a hash of (seed, position, token, everything the row sees), so it rerolls
    whenever the row's context changes.
    """
    device = torch.device('cpu')

    def __init__(self, sc):
        self.seed = sc['seed']
        self.p_sure = sc['p_sure']
        self.stop_from = sc.get('stop_from')
        self.p_stop = sc.get('p_stop', 0.0)
        self.kv = []          # (pos, token) of every cache row a store_kv=True call wrote
        self.calls = []
        self.last_input = None

    def eval(self):
        return self

    def row_logits(self, pos, tok, seen):
        digest = hashlib.blake2b(repr((self.seed, pos, tok, seen)).encode(), digest_size=16).digest()
        r = random.Random(int.from_bytes(digest, 'little'))
        v = [r.randint(-8, 8) for _ in range(VOCAB)]
        v[MASK] = -64
        if self.stop_from is not None and pos >= self.stop_from and r.random() < self.p_stop:
            pref = r.choice(STOPS)
        else:
            pref = r.randrange(N_REGULAR)
        m = r.randint(52, 72) if r.random() < self.p_sure else r.randint(8, 32)
        v[pref] = 8 + m
        return v

    def __call__(self, input_ids, attention_mask, position_ids, past_key_values, use_cache, store_kv):
        toks = input_ids[0].tolist()
        pos = position_ids[0].tolist()
        n, c = len(toks), len(self.kv)
        m = attention_mask[0]
        assert tuple(m.shape) == (n, c + n), (tuple(m.shape), n, c)
        keys = self.kv + list(zip(pos, toks))
        logits = []
        for i in range(n):
            seen = tuple(keys[j] for j in range(c + n) if m[i, j] > 0)
            logits.append(self.row_logits(pos[i], toks[i], seen))
        self.calls.append(dict(tokens=toks, pos=pos, store_kv=bool(store_kv), cache=c,
                               mask=m.clone(), logits=logits))
        self.last_input = input_ids.clone()
        if store_kv:
            self.kv += list(zip(pos, toks))
        out = torch.tensor(logits, dtype=torch.float64).unsqueeze(0) * LOGIT_SCALE
        return SimpleNamespace(logits=out)


# ---- the scenarios ----------------------------------------------------------------------
# sampler: 'greedy' = the wrapper's sampler at T=1, top_k=1 (make_sdar: temperature 0);
#          'gh_topk1' = generate.py's own sampler at T=1, top_k=1 (make_sdar: the same).
# expect:  features the run must show, or the generator refuses to write the fixture.
SCENARIOS = [
    dict(name='b16_dynamic_tail_budget',
         what='b16 defaults; P=21, so the 5-token tail (holding a stop id) rides in the first '
              'decode block; runs to max_new_tokens 40 and returns 43 (block end)',
         block=16, steps=16, dynamic=True, tau=0.9, sampler='greedy',
         prompt_len=21, prompt_stop_at=19, max_new=40, p_sure=0.4, seed=1,
         expect=['prompt_tail', 'prompt_stop_ignored', 'multi_over_tau', 'min_only', 'budget', 'overshoot']),
    dict(name='b16_dynamic_aligned_stop',
         what='b16 defaults; P=32 (two prefill passes); a stop id lands mid-block in the second '
              'decode block, the rest of that block is still decoded and committed',
         block=16, steps=16, dynamic=True, tau=0.9, sampler='greedy',
         prompt_len=32, max_new=64, p_sure=0.4, stop_from=70, p_stop=0.5, seed=2,
         expect=['prompt_aligned', 'multi_over_tau', 'min_only', 'stop_mid_block']),
    dict(name='b16_static_short_prompt_stop',
         what='static, one token per step; P=9 < block, so no prefill pass; stop in the first block',
         block=16, steps=16, dynamic=False, tau=0.9, sampler='greedy',
         prompt_len=9, max_new=100, p_sure=0.4, stop_from=12, p_stop=0.5, seed=3,
         expect=['no_prefill', 'static', 'stop_mid_block']),
    dict(name='b8_static_s4_clamp',
         what='block 8, steps 4 (2 per step), static; the 3-MASK tail block leaves 1 MASK for a '
              'step of 2, which the clamp handles',
         block=8, steps=4, dynamic=False, tau=0.9, sampler='greedy',
         prompt_len=13, max_new=20, p_sure=0.4, seed=4,
         expect=['prompt_tail', 'static', 'clamp', 'budget', 'overshoot']),
    dict(name='b8_dynamic_s4_clamp',
         what='block 8, steps 4, dynamic with few sure rows: the top-2 fallback runs with 1 MASK '
              'left; a stop ends it',
         block=8, steps=4, dynamic=True, tau=0.9, sampler='greedy',
         prompt_len=15, max_new=24, p_sure=0.3, stop_from=30, p_stop=0.4, seed=5,
         expect=['prompt_tail', 'clamp', 'min_only', 'stop_mid_block']),
    dict(name='b8_static_s3_remainder',
         what='block 8, steps 3: the schedule is [3, 3, 2] (generate.py front-loads the remainder)',
         block=8, steps=3, dynamic=False, tau=0.9, sampler='greedy',
         prompt_len=16, max_new=16, p_sure=0.4, seed=6,
         expect=['prompt_aligned', 'remainder', 'budget']),
    dict(name='b4_tiny_budget',
         what='block 4, steps 2; P=10, max_new_tokens 1 still decodes the whole tail block (2 tokens)',
         block=4, steps=2, dynamic=True, tau=0.9, sampler='greedy',
         prompt_len=10, max_new=1, p_sure=0.4, seed=7,
         expect=['prompt_tail', 'budget', 'overshoot']),
    dict(name='b4_zero_budget_aligned',
         what='block 4; P=8 and max_new_tokens 0: two prefill passes and nothing else',
         block=4, steps=4, dynamic=True, tau=0.9, sampler='greedy',
         prompt_len=8, max_new=0, p_sure=0.4, seed=8,
         expect=['prompt_aligned', 'no_decode']),
    dict(name='b4_empty_prompt',
         what='block 4, no prompt at all: no prefill, the first block is all MASK',
         block=4, steps=4, dynamic=True, tau=0.9, sampler='greedy',
         prompt_len=0, max_new=6, p_sure=0.4, seed=9,
         expect=['no_prefill', 'budget', 'overshoot']),
    dict(name='b16_gh_topk1',
         what="generate.py's own sampler at T=1, top_k=1: the confidence is taken after the "
              'filter, so it is 1.0 and every block fills in one denoise pass',
         block=16, steps=16, dynamic=True, tau=0.9, sampler='gh_topk1',
         prompt_len=20, max_new=30, p_sure=0.4, seed=10,
         expect=['prompt_tail', 'one_pass_blocks', 'budget', 'overshoot']),
]


def make_prompt(sc):
    r = random.Random(sc['seed'] * 1000003 + 17)
    p = [r.randrange(N_REGULAR) for _ in range(sc['prompt_len'])]
    if sc.get('prompt_stop_at') is not None:
        p[sc['prompt_stop_at']] = STOPS[0]
    return p


def run_reference(sc, guard=True):
    ns = load_reference()
    fake = FakeSDAR(sc)
    sampler = ns['wrapper_sampler'] if sc['sampler'] == 'greedy' else ns['gh_sampler']
    if guard:
        def guarded(logits, **kw):
            x0, p = sampler(logits, **kw)
            cur = fake.last_input   # the block as this forward saw it
            return torch.where(cur == MASK, x0, cur), p   # huggingface_bd3.py:233
        ns['sample_with_temperature_topk_topp'] = guarded
    else:
        ns['sample_with_temperature_topk_topp'] = sampler
    torch.manual_seed(0)
    prompt = make_prompt(sc)
    x = ns['block_diffusion_generate'](
        fake, {'input_ids': torch.tensor([prompt], dtype=torch.long).reshape(1, len(prompt))},
        mask_id=MASK, gen_length=sc['max_new'], block_length=sc['block'],
        denoising_steps=sc['steps'], temperature=1.0, top_k=1, top_p=1.0,
        remasking_strategy='low_confidence_dynamic' if sc['dynamic'] else 'low_confidence_static',
        confidence_threshold=sc['tau'], stopping_criteria_idx=STOPS)
    return prompt, fake.calls, x[0].tolist(), ns['get_num_transfer_tokens'](sc['block'], sc['steps']).tolist()


def confidences(rows, sampler):
    """What the reference's selection saw, recomputed in float64 from the logits it was fed."""
    lg = np.array(rows, dtype=np.int64)
    top2 = np.sort(lg, axis=1)[:, -2:]
    assert (top2[:, 1] > top2[:, 0]).all(), 'argmax tie: the draw would not be forced'
    if sampler == 'gh_topk1':
        return np.ones(len(lg))
    f = lg.astype(np.float64) * LOGIT_SCALE
    e = np.exp(f - f.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return p[np.arange(len(f)), f.argmax(axis=1)]


def to_passes(sc, prompt, calls, x, sched):
    """generate.py's forward calls -> Pass records, plus what the run exercised."""
    B, P = sc['block'], len(prompt)
    feats, stats = set(), dict(multi_over_tau=0, min_only=0, clamp=0)
    passes = []
    ci = 0
    npre = P // B
    if npre:
        c = calls[0]
        nb = npre
        want = torch.kron(torch.tril(torch.ones(nb, nb)), torch.ones(B, B))
        assert c['store_kv'] and c['cache'] == 0 and c['pos'] == list(range(nb * B))
        assert torch.equal(c['mask'], want.to(c['mask'].dtype)), 'prefill mask is not block-causal'
        for b in range(nb):
            rows = [[c['tokens'][q], q, q] for q in range(b * B, (b + 1) * B)]
            passes.append(dict(rows=rows, attn='block', logit_rows=[], logits=[], keep=(b + 1) * B, take=[]))
        ci = 1
    output, stopped, step, denoise_per_block = [], False, 0, []
    while ci < len(calls):
        c = calls[ci]
        s = c['pos'][0]
        assert not stopped or c['store_kv'], 'a denoise forward after the stop block'
        assert c['cache'] == s and c['pos'] == list(range(s, s + B)), 'decode block misplaced'
        assert c['mask'].shape == (B, s + B) and bool((c['mask'] > 0).all()), 'decode mask is not Block'
        rows = [[t, s + i, s + i] for i, t in enumerate(c['tokens'])]
        if not c['store_kv']:
            lr = [i for i, t in enumerate(c['tokens']) if t == MASK]
            lg = [c['logits'][i] for i in lr]
            conf = confidences(lg, sc['sampler'])
            n, ks = len(lr), sched[step]
            if sc['dynamic']:
                assert (np.abs(conf - sc['tau']) > 1e-6).all(), 'a confidence within 1e-6 of tau'
                high = int((conf > sc['tau']).sum())
                topk = high < ks
                if not topk and high >= 2:
                    stats['multi_over_tau'] += 1
                if topk:
                    stats['min_only'] += 1
            else:
                topk = True
            if topk:
                k = min(ks, n)
                srt = np.sort(conf)[::-1]
                assert k >= n or srt[k - 1] - srt[k] > 1e-9, 'a confidence tie at the top-k cut'
                if ks > n:
                    stats['clamp'] += 1
            take = []
            nxt = calls[ci + 1] if ci + 1 < len(calls) else None
            assert nxt is not None and nxt['pos'][0] == s, 'a block ended without its commit pass'
            if nxt['store_kv']:   # this pass filled the block's last MASK
                assert MASK not in nxt['tokens']
                for i, t in enumerate(nxt['tokens']):
                    if s + i < P:
                        continue
                    take.append(t)
                    if t in STOPS:
                        stopped = True
                        if i < B - 1:
                            feats.add('stop_mid_block')
                        break
            output += take
            passes.append(dict(rows=rows, attn='block', logit_rows=lr, logits=lg, keep=s, take=take))
            step += 1
        else:
            passes.append(dict(rows=rows, attn='block', logit_rows=[], logits=[], keep=s + B, take=[]))
            denoise_per_block.append(step)
            step = 0
        ci += 1
    decode_passes = len(calls) - (1 if npre else 0)

    # generate.py's own answer: x from P up to the first block it never reached, cut after
    # the first stop. It must be what the passes handed out.
    want = []
    for t in x[P:]:
        if t == MASK:
            break
        want.append(t)
        if t in STOPS:
            break
    assert want == output, f'passes disagree with generate.py: {want} vs {output}'

    feats.add('prompt_aligned' if P % B == 0 and P > 0 else 'prompt_tail' if P % B else 'empty_prompt')
    if any(t in STOPS for t in prompt[npre * B:]):
        feats.add('prompt_stop_ignored')
    if not npre:
        feats.add('no_prefill')
    if decode_passes == 0:
        feats.add('no_decode')
    if not stopped and decode_passes and len(output) == len(x) - P:
        feats.add('budget')
    if len(output) > sc['max_new']:
        feats.add('overshoot')
    if denoise_per_block and all(d == 1 for d in denoise_per_block):
        feats.add('one_pass_blocks')
    if B % sc['steps']:
        feats.add('remainder')
    feats.add('dynamic' if sc['dynamic'] else 'static')
    for k, v in stats.items():
        if v:
            feats.add(k)
    return passes, output, decode_passes, stats, feats


def build(scenarios):
    out = []
    for sc in scenarios:
        prompt, calls, x, sched = run_reference(sc)
        passes, output, ndec, stats, feats = to_passes(sc, prompt, calls, x, sched)
        missing = [f for f in sc['expect'] if f not in feats]
        if missing:
            sys.exit(f"FAIL {sc['name']}: the run does not show {missing} (it shows {sorted(feats)})")
        unguarded = None
        if stats['clamp']:
            _, _, x_raw, _ = run_reference(sc, guard=False)
            unguarded = x_raw != x
        print(f"ok   {sc['name']:30s} P={len(prompt):3d} passes={len(passes):3d} decode={ndec:3d} "
              f"out={len(output):3d} sched={sched} {stats}"
              + (f" unguarded generate.py differs={unguarded}" if unguarded is not None else ''))
        temp, top_k = (0.0, 0) if sc['sampler'] == 'greedy' else (1.0, 1)
        out.append({
            'name': sc['name'],
            'what': sc['what'],
            'ref_sampler': ('huggingface_bd3.py sampler at T=1, top_k=1 (confidence before the filter)'
                            if sc['sampler'] == 'greedy' else
                            'generate.py sampler at T=1, top_k=1 (confidence after the filter)'),
            'config': {'block': sc['block'], 'steps': sc['steps'], 'dynamic': sc['dynamic'], 'tau': sc['tau'],
                       'mask_id': MASK, 'stop_ids': STOPS, 'max_new_tokens': sc['max_new'],
                       'sampling': {'temperature': temp, 'top_k': top_k, 'top_p': 1.0, 'seed': 1234}},
            'prompt': prompt,
            'exercises': sorted(feats),
            'stats': stats,
            'output': output,
            'decode_passes': ndec,
            'passes': passes,
        })
    return {
        'requirement': 'OPEN-DLLM-SDAR',
        'generator': 'utilities/dllm-ref/ref_sdar.py',
        'reference': 'JetAstra/SDAR generate.py (blob a24d2a82) block_diffusion_generate, run on a scripted model',
        'vocab': VOCAB,
        'logit_scale': LOGIT_SCALE,
        'note': 'passes[].logits[i] is the vocab-wide logit row of rows[logit_rows[i]], in units of logit_scale',
        'scenarios': out,
    }


def dump(fx, path):
    """One pass per line, so a diff of the fixture reads pass by pass."""
    j = lambda v: json.dumps(v, separators=(',', ':'))
    lines = ['{']
    for k, v in fx.items():
        if k != 'scenarios':
            lines.append(f'  {json.dumps(k)}: {j(v)},')
    lines.append('  "scenarios": [')
    for si, sc in enumerate(fx['scenarios']):
        lines.append('    {')
        for k, v in sc.items():
            if k != 'passes':
                lines.append(f'      {json.dumps(k)}: {j(v)},')
        lines.append('      "passes": [')
        for pi, p in enumerate(sc['passes']):
            lines.append('        ' + j(p) + (',' if pi + 1 < len(sc['passes']) else ''))
        lines.append('      ]')
        lines.append('    }' + (',' if si + 1 < len(fx['scenarios']) else ''))
    lines += ['  ]', '}']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    with open(path, encoding='utf-8') as f:
        assert json.load(f) == json.loads(json.dumps(fx)), 'the writer lost something'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--check-sources', metavar='DIR',
                    help='verify the vendored reference text against the downloaded files first')
    a = ap.parse_args()
    if a.check_sources:
        check_sources(a.check_sources)
    fx = build(SCENARIOS)
    dump(fx, a.out)
    raw = open(a.out, 'rb').read()
    print(f'wrote {a.out}: {len(raw)} bytes, {len(fx["scenarios"])} scenarios, '
          f'{sum(len(s["passes"]) for s in fx["scenarios"])} passes, sha256 {hashlib.sha256(raw).hexdigest()[:16]}')


if __name__ == '__main__':
    main()
