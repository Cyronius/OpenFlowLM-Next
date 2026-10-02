#!/usr/bin/env python3
"""Turn the evaluation set into dllm-ref's input JSONL for one model.

    python tokenize_set.py --model {wedlm,sdar,llada2,qwen3} --out prompts_<model>.jsonl
                           [--max-new N] [--set eval_set.jsonl] [--category C ...] [--limit N]

Each item's messages go through that model's own HF chat template with
add_generation_prompt=True (plus enable_thinking=False for Qwen3; see common.MODELS for what
each template does), and the rendered text is tokenised without adding special tokens --
the same thing apply_chat_template(tokenize=True) does. Output lines are exactly what the
driver reads:

    {"id": ..., "prompt_ids": [...], "max_new": N}

max_new defaults per category (common.DEFAULT_MAX_NEW: math 512, code 512, ifeval 768,
chat 768); --max-new overrides it for every item. --category and --limit cut the set down
for smoke tests.
"""
import argparse

from common import CATEGORIES, DEFAULT_MAX_NEW, EVAL_SET, MODELS, be_quiet, load_tokenizer, \
    read_jsonl, write_jsonl


def render(tok, model, messages):
    """The model's generation prompt for `messages`, as text."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   **MODELS[model]['template_kwargs'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--model', required=True, choices=sorted(MODELS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-new', type=int, default=None,
                    help='override the per-category generation budget')
    ap.add_argument('--set', default=EVAL_SET)
    ap.add_argument('--category', action='append', choices=CATEGORIES,
                    help='keep only these categories (repeatable)')
    ap.add_argument('--limit', type=int, default=None, help='keep the first N items')
    ap.add_argument('--show', action='store_true', help='print the first rendered prompt')
    args = ap.parse_args()
    be_quiet()

    items = read_jsonl(args.set)
    if args.category:
        items = [it for it in items if it['category'] in args.category]
    if args.limit is not None:
        items = items[:args.limit]

    tok = load_tokenizer(args.model)
    rows, lengths = [], []
    for n, it in enumerate(items):
        text = render(tok, args.model, it['messages'])
        if args.show and n == 0:
            print(f'--- {it["id"]} rendered ---\n{text}\n--- end ---')
        ids = tok(text, add_special_tokens=False)['input_ids']
        max_new = args.max_new if args.max_new is not None else DEFAULT_MAX_NEW[it['category']]
        rows.append({'id': it['id'], 'prompt_ids': ids, 'max_new': max_new})
        lengths.append(len(ids))
    write_jsonl(args.out, rows)
    if lengths:
        print(f'{args.model}: wrote {len(rows)} prompts to {args.out}; prompt tokens '
              f'min {min(lengths)} mean {sum(lengths) / len(lengths):.0f} max {max(lengths)}')


if __name__ == '__main__':
    main()
