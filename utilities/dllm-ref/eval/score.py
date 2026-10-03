#!/usr/bin/env python3
"""Score dllm-ref outputs on the evaluation set, and lay out the chat answers for reading.

    python score.py --model {wedlm,sdar,llada2,qwen3} --results [LABEL=]out.jsonl
                    [--results ...] [--compare MODEL=out.jsonl ...] [--out-dir DIR]
                    [--set eval_set.jsonl] [--timeout 10] [--chat-only]
    python score.py --selftest

Every --results file is detokenised with --model's tokenizer and scored on its own. A
--compare file is decoded with its own model's tokenizer and only appears in the chat
markdown (e.g. Qwen3-8B's answers next to a diffusion model's).

Detokenising: out_ids is cut at the model's first stop id (common.MODELS stop_ids; the
response is everything before it), mask and edit-control ids are dropped, the rest is
decoded with skip_special_tokens. A leading <think>...</think> block is removed before
scoring (an unclosed one leaves an empty answer).

Scoring rules:

  math (GSM8K)  The prediction is the number in the LAST \\boxed{...} (after its last '=',
                if any; commas, $ and % ignored). Failing that, the first number after the
                last '####', then the number after the last "answer is", then the last
                number in the response. Correct when it equals the gold answer within 1e-6.
                Reported as acc (that chain) and acc_boxed (the \\boxed{} step alone, i.e.
                the requested format was followed).
  code          pass@1, greedy: the item passes when prompt + completion + HumanEval's
  (HumanEval)   test + check(entry_point) exits 0 within --timeout seconds (default 10).
                The completion is the first fenced block that defines the entry point (else
                the longest fenced block, else the raw text). If it defines the entry point,
                the program is the HumanEval prompt (for its imports and helpers) followed
                by the block with top-level demo code removed (prints, asserts,
                `if __name__ == "__main__"`); otherwise the block is taken as the body that
                continues the prompt.
  ifeval        IFEval's own programmatic checks, through lm_eval (lm_eval.tasks.ifeval) or
                Google's instruction_following_eval, whichever imports. Reported as
                prompt-level and instruction-level accuracy, strict and loose; headline is
                prompt-level strict. If neither imports, IFEval is reported as unscored:
                `pip install "lm_eval[ifeval]"` (it brings langdetect, immutabledict and
                nltk; nltk fetches its punkt_tab data on first use) and re-run this script
                on the same outputs.
  chat          not scored; written to chat_<model>.md, every run's answer to each prompt
                one after another, for a person to read.

Speed, per category, from the driver's counters: tok/pass = sum(committed) /
sum(decode_passes) (also the per-item mean), mean passes, ms/tok = sum(ms_decode) /
sum(committed), mean ms_prefill, mean output tokens, and how many items ran out of budget
without a stop token.

SAFETY: HumanEval scoring EXECUTES MODEL-WRITTEN CODE on this machine. Each program runs in
its own subprocess (`python -I -X utf8`, minimal environment, stdin closed) in a throwaway
temp directory, is killed after --timeout seconds, and starts with a guard that disables
the destructive os/shutil/subprocess calls and blocks importing socket, urllib,
multiprocessing and ctypes (after human-eval's reliability_guard). That is a seatbelt, not
a sandbox: only score outputs of models you are willing to run code from.

Outputs, in --out-dir (default: the first results file's folder): summary.json,
summary.md, scored.jsonl (scored_<label>.jsonl with several runs: per-item responses and
verdicts) and chat_<model>.md. With --chat-only, chat_<model>.md alone.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile

from common import CATEGORIES, EVAL_SET, MODELS, be_quiet, load_tokenizer, read_jsonl, \
    write_jsonl

# ---------------------------------------------------------------- results and detokenising


def parse_labelled(arg):
    """'LABEL=PATH' or 'PATH' -> (label or None, path). An existing path wins."""
    if '=' in arg and not os.path.exists(arg):
        label, path = arg.split('=', 1)
        if label:
            return label, path
    return None, arg


def default_label(path):
    """results/<model>/<quant>/<setting>/out.jsonl -> '<quant>/<setting>'."""
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    return '/'.join(parts[-3:-1])


def strip_think(text):
    """Remove a reasoning block. Returns (answer, had_think)."""
    if '</think>' in text:
        return text.rsplit('</think>', 1)[1].strip(), True
    if text.lstrip().startswith('<think>'):
        return '', True
    return text.strip(), False


def decode_rows(model, tok, rows):
    """Driver output rows -> {id: decoded record}."""
    spec = MODELS[model]
    stops, strip = set(spec['stop_ids']), set(spec['strip_ids'])
    vocab = len(tok)
    out = {}
    for r in rows:
        ids = list(r.get('out_ids') or [])
        cut = next((i for i, t in enumerate(ids) if t in stops), None)
        stopped = cut is not None
        if stopped:
            ids = ids[:cut]
        kept = [t for t in ids if t not in strip and 0 <= t < vocab]
        raw = tok.decode(kept, skip_special_tokens=True)
        response, had_think = strip_think(raw)
        out[r['id']] = dict(response=response, raw=raw, think=had_think, stopped=stopped,
                            n_out=len(ids), n_stripped=len(ids) - len(kept), driver=r)
    return out


# ---------------------------------------------------------------- GSM8K

_NUM = re.compile(r'-?(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?')  # 42, -3.5, .5, 3/4


def _clean_numeric(s):
    s = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', s)
    s = re.sub(r'\\[dt]?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}', r'\1/\2', s)
    return s.replace('\\$', '').replace('$', '').replace('\\%', '').replace('%', '')


def _as_float(s):
    try:
        if '/' in s:
            a, b = s.split('/')
            return float(a) / float(b)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def first_number(s):
    m = _NUM.search(_clean_numeric(s))
    return _as_float(m.group(0)) if m else None


def last_number(s):
    nums = _NUM.findall(_clean_numeric(s))
    return _as_float(nums[-1]) if nums else None


def last_boxed(text):
    """Content of the last \\boxed{...} / \\fbox{...}, braces balanced; None if absent."""
    start = max(text.rfind('\\boxed{'), text.rfind('\\fbox{'))
    if start < 0:
        return None
    i = text.index('{', start) + 1
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        j += 1
    return text[i:j - 1] if depth == 0 else text[i:]


def gsm8k_extract(response):
    """(prediction from the whole chain, prediction from \\boxed{} alone)."""
    boxed = last_boxed(response)
    from_box = None
    if boxed is not None:
        from_box = first_number(boxed.rsplit('=', 1)[-1])
    if from_box is not None:
        return from_box, from_box
    if '####' in response:
        v = first_number(response.rsplit('####', 1)[1])
        if v is not None:
            return v, None
    m = list(re.finditer(r'answer is', response, re.I))
    if m:
        v = first_number(response[m[-1].end():][:80])
        if v is not None:
            return v, None
    return last_number(response), None


def score_math(meta, response):
    gold = float(meta['answer'])
    pred, pred_boxed = gsm8k_extract(response)
    ok = pred is not None and abs(pred - gold) < 1e-6
    ok_boxed = pred_boxed is not None and abs(pred_boxed - gold) < 1e-6
    return dict(gold=meta['answer'], pred=pred, correct=ok, correct_boxed=ok_boxed)


# ---------------------------------------------------------------- HumanEval

_GUARD = '''\
import builtins as _b, os as _os, shutil as _sh, subprocess as _sp, sys as _sys
for _n in ("kill", "killpg", "system", "popen", "putenv", "remove", "removedirs", "rmdir",
           "fchdir", "setuid", "fork", "forkpty", "rename", "renames", "truncate", "replace",
           "unlink", "fchmod", "fchown", "chmod", "chown", "chroot", "lchflags", "lchmod",
           "lchown", "startfile", "execv", "execve", "execl", "execle", "execlp", "execlpe",
           "execvp", "execvpe", "spawnl", "spawnle", "spawnv", "spawnve"):
    if hasattr(_os, _n):
        setattr(_os, _n, None)
_sh.rmtree = _sh.move = _sh.chown = None
_sp.Popen = None
_b.exit = _b.quit = _b.help = None
for _m in ("socket", "urllib", "http", "multiprocessing", "ctypes", "ipdb", "joblib",
           "resource", "psutil", "tkinter"):
    _sys.modules[_m] = None
del _b, _os, _sh, _sp, _sys, _n, _m
'''

_FENCE_LINE = re.compile(r'^\s*```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*$')


def _fenced_blocks(response):
    """(lang, body) of each fenced block, read line by line. A fence line that directly follows
    another fence line (blank lines aside) is the same fence repeated and neither opens nor
    closes anything: models double their fences -- Qwen3-8B writes "```python" twice and
    "```" twice when the prompt itself is fenced -- and pairing them naively yields an empty
    block, then code mixed with fences and the prose after it. An unclosed last block runs to
    the end of the response."""
    blocks, cur, lang, after_fence = [], None, '', False
    for line in response.split('\n'):
        m = _FENCE_LINE.match(line)
        if m:
            if after_fence:
                continue
            after_fence = True
            if cur is None:
                cur, lang = [], m.group(1).lower()
            else:
                blocks.append((lang, '\n'.join(cur)))
                cur = None
            continue
        if line.strip():
            after_fence = False
        if cur is not None:
            cur.append(line)
    if cur is not None:
        blocks.append((lang, '\n'.join(cur)))
    return blocks


def _defines(code, name):
    return re.search(rf'^\s*(?:async\s+)?def\s+{re.escape(name)}\s*\(', code, re.M) is not None


def extract_code(response, entry_point):
    """The code the model meant as its answer, or None if there is none."""
    blocks = _fenced_blocks(response)
    py = [b for lang, b in blocks if lang in ('', 'python', 'py', 'python3')]
    cands = [b for b in (py or [b for _, b in blocks]) if b.strip()]
    if cands:
        with_def = [b for b in cands if _defines(b, entry_point)]
        return with_def[0] if with_def else max(cands, key=len)
    if not response.strip():
        return None
    # Unfenced: skip any prose before the first top-level import/def/class.
    m = re.search(r'^(?:from |import |def |async def |class |@)', response, re.M)
    return response[m.start():] if m else response


def _is_main_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__')


def strip_demo_code(code):
    """Drop top-level statements that only demonstrate the function: bare expressions
    (prints, calls), asserts, loops and the __main__ guard. Unparseable code is returned
    as is -- it will fail on its own."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    drop = (ast.Expr, ast.Assert, ast.For, ast.While, ast.AsyncFor)
    lines = code.split('\n')
    keep = []
    for node in tree.body:
        if isinstance(node, drop) or _is_main_guard(node):
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        keep.append('\n'.join(lines[start - 1:node.end_lineno]))
    return '\n\n'.join(keep)


def build_program(meta, response):
    """(program text without guard/test, how) or (None, 'no_code')."""
    prompt, entry = meta['prompt'], meta['entry_point']
    code = extract_code(response, entry)
    if code is None:
        return None, 'no_code'
    if _defines(code, entry):
        return prompt.rstrip('\n') + '\n\n\n' + strip_demo_code(code) + '\n', 'function'
    lines = code.split('\n')
    first = next((ln for ln in lines if ln.strip()), '')
    if not first.startswith((' ', '\t')):
        code = '\n'.join(('    ' + ln) if ln.strip() else ln for ln in lines)
    sep = '' if prompt.endswith('\n') else '\n'
    return prompt + sep + code.rstrip() + '\n', 'body'


def run_program(source, timeout):
    """Run a program in a throwaway directory. Returns (status, stderr tail)."""
    with tempfile.TemporaryDirectory(prefix='he_', ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, 'prog.py')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(source)
        env = {'TEMP': tmp, 'TMP': tmp}
        if os.name == 'nt':
            env['SYSTEMROOT'] = os.environ.get('SYSTEMROOT', r'C:\Windows')
        try:
            p = subprocess.run([sys.executable, '-I', '-X', 'utf8', path], cwd=tmp, env=env,
                               stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', timeout=timeout)
        except subprocess.TimeoutExpired:
            return 'timeout', ''
        return ('passed' if p.returncode == 0 else 'failed'), p.stderr[-600:]


def score_code(meta, response, timeout):
    program, how = build_program(meta, response)
    if program is None:
        return dict(passed=False, status='no_code', how=how)
    source = (_GUARD + '\n' + program + '\n\n' + meta['test'] + '\n\n'
              + f"check({meta['entry_point']})\n")
    status, err = run_program(source, timeout)
    return dict(passed=status == 'passed', status=status, how=how, stderr=err)


# ---------------------------------------------------------------- IFEval

IFEVAL_HINT = ('pip install "lm_eval[ifeval]"  (brings langdetect, immutabledict, nltk), '
               'then re-run score.py on the same outputs')


def load_ifeval_checker():
    """(name, check(meta, response) -> dict) for the first IFEval implementation that
    imports, or (None, reason)."""
    errors = []
    try:
        from lm_eval.tasks.ifeval import utils as ife

        def check(meta, response):
            doc = dict(key=meta['key'], prompt=meta['prompt'],
                       instruction_id_list=meta['instruction_id_list'], kwargs=meta['kwargs'])
            r = ife.process_results(doc, [response])
            return dict(prompt_strict=bool(r['prompt_level_strict_acc']),
                        inst_strict=[bool(x) for x in r['inst_level_strict_acc']],
                        prompt_loose=bool(r['prompt_level_loose_acc']),
                        inst_loose=[bool(x) for x in r['inst_level_loose_acc']])
        return 'lm_eval', check
    except Exception as e:  # ImportError, or a missing dependency of the module
        errors.append(f'lm_eval: {e}')
    try:
        from instruction_following_eval import evaluation_lib as ife

        def check(meta, response):
            kwargs = [{k: v for k, v in kw.items() if v is not None} for kw in meta['kwargs']]
            inp = ife.InputExample(key=meta['key'], instruction_id_list=meta['instruction_id_list'],
                                   prompt=meta['prompt'], kwargs=kwargs)
            p2r = {meta['prompt']: response}
            s = ife.test_instruction_following_strict(inp, p2r)
            lo = ife.test_instruction_following_loose(inp, p2r)
            return dict(prompt_strict=bool(s.follow_all_instructions),
                        inst_strict=[bool(x) for x in s.follow_instruction_list],
                        prompt_loose=bool(lo.follow_all_instructions),
                        inst_loose=[bool(x) for x in lo.follow_instruction_list])
        return 'instruction_following_eval', check
    except Exception as e:
        errors.append(f'instruction_following_eval: {e}')
    return None, '; '.join(errors)


def score_ifeval(check, meta, response):
    try:
        return check(meta, response)
    except Exception as e:  # a checker crash counts as not followed, and is recorded
        n = len(meta['instruction_id_list'])
        return dict(prompt_strict=False, inst_strict=[False] * n, prompt_loose=False,
                    inst_loose=[False] * n, error=repr(e))


# ---------------------------------------------------------------- aggregation


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def speed_stats(recs):
    d = [r['driver'] for r in recs]
    committed = [x.get('committed') or 0 for x in d]
    dpasses = [x.get('decode_passes') or 0 for x in d]
    ms_dec = [x.get('ms_decode') or 0.0 for x in d]
    return dict(
        tok_per_pass=sum(committed) / sum(dpasses) if sum(dpasses) else None,
        tok_per_pass_item_mean=_mean([c / p for c, p in zip(committed, dpasses) if p]),
        passes_mean=_mean([x.get('passes') for x in d]),
        decode_passes_mean=_mean([x.get('decode_passes') for x in d]),
        ms_per_tok=sum(ms_dec) / sum(committed) if sum(committed) else None,
        ms_prefill_mean=_mean([x.get('ms_prefill') for x in d]),
        out_tokens_mean=_mean([r['n_out'] for r in recs]),
        no_stop=sum(1 for r in recs if not r['stopped']),
        think=sum(1 for r in recs if r['think']),
    )


def score_run(model, label, path, tok, items, ifeval, timeout, chat_only):
    """Score one results file. Returns (summary dict, per-item rows, decoded records)."""
    rows = read_jsonl(path)
    dec = decode_rows(model, tok, rows)
    by_id = {it['id']: it for it in items}
    unknown = sorted(set(dec) - set(by_id))
    missing = sorted(set(by_id) - set(dec))
    if unknown:
        print(f'  warning: {len(unknown)} ids in {path} are not in the set: {unknown[:5]}')
    ife_name, ife_check = ifeval
    if ife_name is None:
        ife_check = None  # the second element is then the reason, not a checker

    scored, cats = [], {c: [] for c in CATEGORIES}
    for it in items:
        rec = dec.get(it['id'])
        if rec is None:
            continue
        cat, meta = it['category'], it['meta']
        verdict = {}
        if not chat_only:
            if cat == 'math':
                verdict = score_math(meta, rec['response'])
            elif cat == 'code':
                verdict = score_code(meta, rec['response'], timeout)
            elif cat == 'ifeval' and ife_check:
                verdict = score_ifeval(ife_check, meta, rec['response'])
        rec['verdict'] = verdict
        cats[cat].append(rec)
        scored.append(dict(id=it['id'], category=cat, response=rec['response'],
                           n_out=rec['n_out'], stopped=rec['stopped'], think=rec['think'],
                           n_stripped=rec['n_stripped'], **verdict))

    summary = dict(label=label, path=os.path.abspath(path), model=model, n=len(scored),
                   missing=missing, categories={})
    for cat, recs in cats.items():
        if not recs:
            continue
        s = dict(n=len(recs), **speed_stats(recs))
        v = [r['verdict'] for r in recs]
        if chat_only or cat == 'chat':
            pass
        elif cat == 'math':
            s['acc'] = _mean([x['correct'] for x in v])
            s['acc_boxed'] = _mean([x['correct_boxed'] for x in v])
        elif cat == 'code':
            s['pass_at_1'] = _mean([x['passed'] for x in v])
            s['status'] = {k: sum(1 for x in v if x['status'] == k)
                           for k in ('passed', 'failed', 'timeout', 'no_code')}
        elif cat == 'ifeval':
            if ife_check:
                s['checker'] = ife_name
                s['prompt_strict'] = _mean([x['prompt_strict'] for x in v])
                s['prompt_loose'] = _mean([x['prompt_loose'] for x in v])
                inst_s = [b for x in v for b in x['inst_strict']]
                inst_l = [b for x in v for b in x['inst_loose']]
                s['inst_strict'] = _mean(inst_s)
                s['inst_loose'] = _mean(inst_l)
                s['checker_errors'] = sum(1 for x in v if 'error' in x)
            else:
                s['unscored'] = f'no IFEval checker importable: {IFEVAL_HINT}'
        summary['categories'][cat] = s
    all_recs = [r for recs in cats.values() for r in recs]
    if all_recs:
        summary['categories']['all'] = dict(n=len(all_recs), **speed_stats(all_recs))
    return summary, scored, dec


# ---------------------------------------------------------------- reports


def _f(x, nd=3):
    return '-' if x is None else f'{x:.{nd}f}'


def score_cell(cat, s):
    if cat == 'math' and 'acc' in s:
        return f"acc {_f(s['acc'])} (boxed {_f(s['acc_boxed'])})"
    if cat == 'code' and 'pass_at_1' in s:
        return f"pass@1 {_f(s['pass_at_1'])}"
    if cat == 'ifeval':
        if 'prompt_strict' in s:
            return f"prompt-strict {_f(s['prompt_strict'])}, inst-strict {_f(s['inst_strict'])}"
        if 'unscored' in s:
            return 'unscored'
    return ''


def markdown_table(summaries):
    out = ['| run | category | n | score | tok/pass | passes | ms/tok | prefill ms '
           '| out tok | no stop |',
           '|---|---|--:|---|--:|--:|--:|--:|--:|--:|']
    for sm in summaries:
        for cat in list(CATEGORIES) + ['all']:
            s = sm['categories'].get(cat)
            if not s:
                continue
            out.append(f"| {sm['label']} | {cat} | {s['n']} | {score_cell(cat, s)} "
                       f"| {_f(s['tok_per_pass'], 2)} | {_f(s['passes_mean'], 1)} "
                       f"| {_f(s['ms_per_tok'], 1)} | {_f(s['ms_prefill_mean'], 0)} "
                       f"| {_f(s['out_tokens_mean'], 0)} | {s['no_stop']} |")
    return '\n'.join(out) + '\n'


def _demote_headings(text, by=3):
    """Push a response's own markdown headings below the report's, outside code fences."""
    out, fenced = [], False
    for ln in text.split('\n'):
        if ln.lstrip().startswith('```'):
            fenced = not fenced
        elif not fenced:
            m = re.match(r'^(#{1,6})(\s)', ln)
            if m:
                ln = '#' * min(6, len(m.group(1)) + by) + ln[len(m.group(1)):]
        out.append(ln)
    if fenced:
        out.append('```')
    return '\n'.join(out)


def chat_markdown(model, items, runs):
    """runs: [(label, {id: decoded record})]."""
    chats = [it for it in items if it['category'] == 'chat']
    out = [f'# Open chat: {model}', '',
           'Each prompt, then every run\'s answer in turn. Header: tokens generated, '
           'tokens per decode pass, and "no stop" when the budget ran out first.', '']
    for it in chats:
        meta = it['meta']
        out.append(f"## {it['id']}: {meta['topic']} ({meta['kind']})")
        out.append('')
        out.extend('> ' + ln for ln in it['messages'][-1]['content'].split('\n'))
        out.append('')
        if meta.get('reference'):
            out += [f"*Reference:* {meta['reference']}", '']
        for label, dec in runs:
            rec = dec.get(it['id'])
            if rec is None:
                out += [f'### {label}', '', '*(no output)*', '']
                continue
            d = rec['driver']
            tpp = (d.get('committed') or 0) / d['decode_passes'] if d.get('decode_passes') else None
            note = f"{rec['n_out']} tokens, {_f(tpp, 2)} tok/pass"
            if not rec['stopped']:
                note += ', no stop'
            out += [f'### {label}  ({note})', '', _demote_headings(rec['response']) or '*(empty)*',
                    '']
    return '\n'.join(out)


# ---------------------------------------------------------------- self-test


def selftest(items, timeout):
    """Check the harness on known answers: GSM8K golds written in \\boxed{} and in
    prose, and HumanEval's canonical solutions as a fenced function (also with its fences
    doubled and prose after it) and as a bare body."""
    bad = []
    for it in items:
        meta = it['meta']
        if it['category'] == 'math':
            for resp in (f"So the total is {meta['answer']} dollars.\n\\boxed{{{meta['answer']}}}",
                         f"Step 1 ... 3 + 4 = 7.\nThe answer is ${meta['answer']}."):
                if not score_math(meta, resp)['correct']:
                    bad.append((it['id'], resp))
        elif it['category'] == 'code':
            fn = f"Here you go:\n```python\n{meta['prompt']}{meta['canonical_solution']}```\nDone."
            # the fences doubled, then an explanation with a quote in it: what Qwen3-8B writes
            doubled = (f"Here you go:\n\n```python\n```python\n{meta['prompt']}{meta['canonical_solution']}"
                       f"```\n```\n\n### Explanation:\n- It's the obvious approach, and it works.")
            for how, resp in (('function', fn), ('function', doubled), ('body', meta['canonical_solution'])):
                v = score_code(meta, resp, timeout)
                if not v['passed'] or v['how'] != how:
                    bad.append((it['id'], how, v))
    n_math = sum(it['category'] == 'math' for it in items)
    n_code = sum(it['category'] == 'code' for it in items)
    print(f'selftest: {n_math} GSM8K x2, {n_code} HumanEval x3: '
          + ('all pass' if not bad else f'{len(bad)} FAILURES'))
    for b in bad:
        print('  ', b)
    return not bad


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=sorted(MODELS))
    ap.add_argument('--results', action='append', default=[], metavar='[LABEL=]PATH')
    ap.add_argument('--compare', action='append', default=[], metavar='MODEL=PATH',
                    help='another model\'s output, shown in the chat markdown only')
    ap.add_argument('--set', default=EVAL_SET)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--timeout', type=float, default=10.0, help='seconds per HumanEval program')
    ap.add_argument('--chat-only', action='store_true',
                    help='skip scoring; write only the chat markdown (speed table printed)')
    ap.add_argument('--selftest', action='store_true',
                    help='run the GSM8K and HumanEval scorers on known answers and exit')
    args = ap.parse_args()
    be_quiet()

    items = read_jsonl(args.set)
    if args.selftest:
        sys.exit(0 if selftest(items, args.timeout) else 1)
    if not args.model or not args.results:
        ap.error('--model and at least one --results are required')

    runs = []
    for arg in args.results:
        label, path = parse_labelled(arg)
        runs.append((label or default_label(path), path))
    labels = [lb for lb, _ in runs]
    runs = [(f'{lb} #{i}' if labels.count(lb) > 1 else lb, p) for i, (lb, p) in enumerate(runs)]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(runs[0][1]))
    os.makedirs(out_dir, exist_ok=True)

    ifeval = (None, None) if args.chat_only else load_ifeval_checker()
    if not args.chat_only and ifeval[0] is None:
        print(f'IFEval: unscored ({ifeval[1]}). To score it: {IFEVAL_HINT}')

    tok = load_tokenizer(args.model)
    summaries, chat_runs = [], []
    for label, path in runs:
        print(f'scoring {label}: {path}')
        sm, scored, dec = score_run(args.model, label, path, tok, items, ifeval, args.timeout,
                                    args.chat_only)
        summaries.append(sm)
        chat_runs.append((label, dec))
        if not args.chat_only:
            name = 'scored.jsonl' if len(runs) == 1 else \
                'scored_' + re.sub(r'[^A-Za-z0-9_.-]+', '_', label) + '.jsonl'
            write_jsonl(os.path.join(out_dir, name), scored)

    for arg in args.compare:
        cmodel, path = arg.split('=', 1)
        if cmodel not in MODELS:
            ap.error(f'--compare: unknown model {cmodel!r}')
        ctok = load_tokenizer(cmodel)
        chat_runs.append((f'{cmodel} {default_label(path)}',
                          decode_rows(cmodel, ctok, read_jsonl(path))))

    chat_path = os.path.join(out_dir, f'chat_{args.model}.md')
    with open(chat_path, 'w', encoding='utf-8') as f:
        f.write(chat_markdown(args.model, items, chat_runs))
    table = markdown_table(summaries)
    print(table)
    if args.chat_only:
        print(f'wrote {chat_path}')
        return

    summary = dict(model=args.model, set=os.path.abspath(args.set),
                   ifeval_checker=ifeval[0], runs=summaries)
    if ifeval[0] is None:
        summary['ifeval_unscored_reason'] = ifeval[1]
        summary['ifeval_install'] = IFEVAL_HINT
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(f'# {args.model}\n\n{table}')
    print(f'wrote summary.json, summary.md, scored*.jsonl, chat_{args.model}.md to {out_dir}')


if __name__ == '__main__':
    main()
