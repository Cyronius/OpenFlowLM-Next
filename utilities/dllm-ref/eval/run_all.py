#!/usr/bin/env python3
"""Run the whole comparison: every model x quant x decode setting through dllm-ref, then score.

    python run_all.py --exe PATH\\TO\\dllm-ref.exe [--dry-run] [--models wedlm sdar ...]
                      [--quants Q8_0 Q4_1] [--trace] [--driver-arg=--threads --driver-arg=8]
                      [--gguf-dir C:\\dev\\dllm\\gguf] [--results-root C:\\dev\\dllm\\results]

Orchestration only: it runs nothing itself but tokenize_set.py, dllm-ref and score.py.
--dry-run prints every command and touches nothing.

Per model:
  1. <root>/<model>/prompts.jsonl  <- tokenize_set.py (once; the prompt does not depend on
     the quant or the decode setting).
  2. Per quant and setting, in <root>/<model>/<quant>/<setting>/:
       dllm-ref ... --out out.jsonl.part  (log in driver.log, traces in trace/ with
       --trace), renamed to out.jsonl only when the driver exits 0; then
       score.py -> summary.json, summary.md, scored.jsonl, chat_<model>.md.
     A run whose out.jsonl exists is not run again; it is re-scored only if its summary is
     missing or older than out.jsonl, or with --rescore.
  3. <root>/<model>/chat_<model>.md: that model's chat answers across all its runs, with
     Qwen3-8B Q8_0's answers alongside (score.py --chat-only --compare).
Finally <root>/summary.md and summary.json collect every run's numbers in one table.

Decode settings, all greedy (--temperature 0; WeDLM's own code defaults to T 0.2):
  wedlm   window 4 and 6, each at tau 0.4 and 0.6 (lambda at the driver's default)
  sdar    tau 0.9 and 0.95 (block 16, 16 steps: the driver's defaults)
  llada2  the driver's defaults (threshold 0.5, block 32)
  qwen3   --policy ar, the autoregressive yardstick
"""
import argparse
import json
import os
import subprocess
import sys
import time

from common import EVAL_DIR, EVAL_SET, GGUF_ROOT, MODELS, RESULTS_ROOT

QUANTS = ('Q8_0', 'Q4_1')
GREEDY = ['--temperature', '0']
# 2048 holds the longest prompt (~340) plus the largest budget (768); the driver's own default
# (8192) sizes llama.cpp's buffers for 8K and decodes markedly slower on the iGPU.
DRIVER_DEFAULTS = ['--ctx', '2048']

SETTINGS = {
    'wedlm': [(f'w{w}_tau{t}', ['--window', str(w), '--tau', str(t)])
              for w in (4, 6) for t in (0.4, 0.6)],
    'sdar': [(f'tau{t}', ['--tau', str(t)]) for t in (0.9, 0.95)],
    'llada2': [('default', [])],
    'qwen3': [('ar', [])],
}
BASELINE = ('qwen3', 'Q8_0', 'ar')   # shown next to every model's chat answers


def show(cmd):
    print('  $ ' + subprocess.list2cmdline(cmd), flush=True)


def run(cmd, dry, log=None):
    """Run a command (or just print it). Returns True on success."""
    show(cmd)
    if dry:
        return True
    t0 = time.time()
    if log:
        with open(log, 'w', encoding='utf-8') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    else:
        rc = subprocess.run(cmd).returncode
    print(f'    exit {rc} after {time.time() - t0:.0f} s', flush=True)
    return rc == 0


def newer(a, b):
    """True if file a exists and is at least as new as file b."""
    return os.path.exists(a) and os.path.getmtime(a) >= os.path.getmtime(b)


def driver_cmd(args, model, quant, knobs, prompts, part, run_dir):
    spec = MODELS[model]
    gguf = os.path.join(args.gguf_dir, f"{spec['gguf_stem']}-{quant}.gguf")
    cmd = [args.exe, '--model', gguf, '--policy', spec['policy'], '--in', prompts,
           '--out', part]
    if args.trace:
        cmd += ['--trace', os.path.join(run_dir, 'trace')]
    return cmd + knobs + GREEDY + DRIVER_DEFAULTS + args.driver_arg


def score_cmd(model, results, out_dir, extra=()):
    return [sys.executable, os.path.join(EVAL_DIR, 'score.py'), '--model', model,
            *[a for r in results for a in ('--results', r)], '--out-dir', out_dir, *extra]


def collect(root, runs):
    """One table over every scored run."""
    rows, table = [], [
        '| model | quant | setting | GSM8K acc | HumanEval pass@1 | IFEval prompt-strict '
        '| tok/pass math / code / ifeval / chat | ms/tok | no stop |',
        '|---|---|---|--:|--:|--:|---|--:|--:|']
    for model, quant, setting, run_dir in runs:
        path = os.path.join(run_dir, 'summary.json')
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            sm = json.load(f)['runs'][0]
        rows.append(dict(model=model, quant=quant, setting=setting, summary=sm))
        c = sm['categories']

        def g(cat, key, nd=3):
            v = c.get(cat, {}).get(key)
            return '-' if v is None else f'{v:.{nd}f}'
        ife = 'unscored' if 'unscored' in c.get('ifeval', {}) else g('ifeval', 'prompt_strict')
        tpp = ' / '.join(g(k, 'tok_per_pass', 2) for k in ('math', 'code', 'ifeval', 'chat'))
        table.append(f"| {model} | {quant} | {setting} | {g('math', 'acc')} "
                     f"| {g('code', 'pass_at_1')} | {ife} | {tpp} | {g('all', 'ms_per_tok', 1)} "
                     f"| {c.get('all', {}).get('no_stop', '-')} |")
    with open(os.path.join(root, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(root, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('# Diffusion LMs vs Qwen3-8B\n\n' + '\n'.join(table) + '\n')
    print('\n'.join(table))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exe', required=True, help='path to dllm-ref(.exe)')
    ap.add_argument('--models', nargs='+', default=list(SETTINGS), choices=list(SETTINGS))
    ap.add_argument('--quants', nargs='+', default=list(QUANTS), choices=list(QUANTS))
    ap.add_argument('--gguf-dir', default=GGUF_ROOT)
    ap.add_argument('--results-root', default=RESULTS_ROOT)
    ap.add_argument('--trace', action='store_true', help='have dllm-ref write traces')
    ap.add_argument('--driver-arg', action='append', default=[],
                    help='extra argument for every dllm-ref call (repeatable; use '
                         '--driver-arg=--flag for values starting with -)')
    ap.add_argument('--rescore', action='store_true',
                    help='re-score every existing output (e.g. after installing lm_eval)')
    ap.add_argument('--dry-run', action='store_true', help='print the commands only')
    ap.add_argument('--settings', nargs='+', default=None,
                    help='run only these setting names (e.g. w6_tau0.4 tau0.9 default ar)')
    ap.add_argument('--set', default=EVAL_SET,
                    help='the eval set to tokenise (eval_set_core.jsonl: the 60-item first pass)')
    args = ap.parse_args()
    dry, root = args.dry_run, args.results_root
    if not dry and not os.path.exists(args.exe):
        ap.error(f'no driver at {args.exe}')

    done, failed = [], []
    for model in args.models:
        print(f'== {model}', flush=True)
        model_dir = os.path.join(root, model)
        prompts = os.path.join(model_dir, 'prompts.jsonl')
        if not dry:
            os.makedirs(model_dir, exist_ok=True)
        if os.path.exists(prompts):
            print(f'  prompts: {prompts} (exists)')
        else:
            run([sys.executable, os.path.join(EVAL_DIR, 'tokenize_set.py'), '--model', model,
                 '--out', prompts, '--set', args.set], dry)

        for quant in args.quants:
            for setting, knobs in SETTINGS[model]:
                if args.settings and setting not in args.settings:
                    continue
                run_dir = os.path.join(model_dir, quant, setting)
                out = os.path.join(run_dir, 'out.jsonl')
                print(f'-- {model} {quant} {setting}', flush=True)
                if os.path.exists(out):
                    print(f'  output exists: {out}')
                else:
                    if not dry:
                        os.makedirs(run_dir, exist_ok=True)
                    part = out + '.part'
                    ok = run(driver_cmd(args, model, quant, knobs, prompts, part, run_dir), dry,
                             log=os.path.join(run_dir, 'driver.log'))
                    if not ok:
                        failed.append((model, quant, setting))
                        continue
                    if not dry:
                        os.replace(part, out)
                if dry or args.rescore or not newer(os.path.join(run_dir, 'summary.json'), out):
                    if not run(score_cmd(model, [f'{quant}/{setting}={out}'], run_dir), dry):
                        failed.append((model, quant, setting, 'score'))
                done.append((model, quant, setting, run_dir))

    # Each model's chat answers across its runs, next to the baseline's.
    base_out = os.path.join(root, *BASELINE, 'out.jsonl')
    for model in args.models:
        outs = [(q, s, os.path.join(root, model, q, s, 'out.jsonl'))
                for m, q, s, _ in done if m == model]
        outs = [(q, s, p) for q, s, p in outs if dry or os.path.exists(p)]
        if not outs:
            continue
        extra = ['--chat-only']
        if model != BASELINE[0] and (dry or os.path.exists(base_out)):
            extra += ['--compare', f'{BASELINE[0]}={base_out}']
        print(f'== chat side by side: {model}')
        run(score_cmd(model, [f'{q}/{s}={p}' for q, s, p in outs],
                      os.path.join(root, model), extra), dry)

    if not dry:
        collect(root, done)
    if failed:
        print('FAILED: ' + ', '.join(' '.join(f) for f in failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
