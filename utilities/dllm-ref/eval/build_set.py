#!/usr/bin/env python3
"""Build the fixed evaluation set for the diffusion-LM comparison (eval_set.jsonl).

    python build_set.py [--seed 1234] [--out eval_set.jsonl]

Downloads three public datasets from the Hugging Face Hub at pinned revisions (well under
1 MB in all) and draws a fixed, seeded subset of each, then appends 20 open-ended chat
prompts written for this set:

    math    50  GSM8K test (openai/gsm8k, config "main")
    code    30  HumanEval (openai/openai_humaneval)
    ifeval  50  IFEval (google/IFEval)
    chat    20  hand-written requests of the kind people give a local assistant

Each output line is {"id", "category", "messages", "meta"}. `messages` is the chat to send:
a single user turn, no system prompt (each model's own template supplies whatever its
authors use by default). `meta` holds what scoring needs:

    math    answer (the gold final number), solution (GSM8K's worked answer), source
    code    task_id, prompt, test, entry_point, canonical_solution, source
    ifeval  key, prompt, instruction_id_list, kwargs, source
    chat    topic, kind, and for the reasoning item a reference answer for the reader

Prompt formats, all zero-shot:

    GSM8K     "{question}\\nPlease reason step by step, and put your final answer within
              \\boxed{}." -- Qwen3's recommended math format, also SDAR's OpenCompass
              gsm8k_0shot_v2 config.
    HumanEval EvalPlus's chat instruction: "Please provide a self-contained Python script
              that solves the following problem in a markdown code block:" followed by the
              HumanEval prompt in a fence. No assistant prefill, so every model gets the
              same add_generation_prompt=True turn.
    IFEval    the IFEval prompt verbatim.

The subset is a pure function of the seed and the pinned revisions; re-running rewrites the
same file.
"""
import argparse
import json
import os
import random
import re

from common import EVAL_SET, write_jsonl

SOURCES = {
    'gsm8k': dict(repo='openai/gsm8k', file='main/test-00000-of-00001.parquet',
                  revision='740312add88f781978c0658806c59bc2815b9866', n=50),
    'humaneval': dict(repo='openai/openai_humaneval',
                      file='openai_humaneval/test-00000-of-00001.parquet',
                      revision='7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544', n=30),
    'ifeval': dict(repo='google/IFEval', file='ifeval_input_data.jsonl',
                   revision='966cd89545d6b6acfd7638bc708b98261ca58e84', n=50),
}

GSM8K_SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
HUMANEVAL_PREFIX = ('Please provide a self-contained Python script that solves the following '
                    'problem in a markdown code block:')

SUMMARY_PARAGRAPH = (
    "When the city of Ravensholm converted three of its downtown parking garages into mixed-use "
    "buildings in 2019, critics predicted that shoppers would abandon the area. Five years later "
    "the opposite has happened. Foot traffic on the main commercial street is up 22 percent, "
    "and the number of vacant storefronts has fallen from 31 to 9. City planners credit two "
    "changes made at the same time: a free shuttle that loops between the train station and the "
    "old town every ten minutes, and a rule that ground floors of the converted garages must "
    "hold shops or cafes rather than offices. Not everyone has benefited. Several longtime "
    "residents say rents in the converted buildings are out of reach for the people who used to "
    "work downtown, and a survey by the local business association found that delivery drivers "
    "now spend an average of eleven extra minutes per stop looking for a place to unload. The "
    "council is expected to vote next spring on whether to repeat the program in two more "
    "neighborhoods."
)

WORDY_PARAGRAPH = (
    "At this point in time, it is our considered opinion that, due to the fact that the current "
    "onboarding process for new employees is in many ways somewhat lengthy and also in some "
    "respects rather confusing, it would be in the best interests of all parties concerned if we "
    "were to take the necessary steps to undertake a thorough review of the process with a view "
    "to making it simpler, in order that new hires are able to become productive members of their "
    "respective teams in a shorter period of time than is currently the case."
)

# (topic, kind, prompt, reference-or-None)
CHAT_PROMPTS = [
    ('heat-pump', 'explanation',
     "How does a heat pump manage to heat a house even when it's freezing outside? Please explain "
     "it for someone without a physics background.", None),
    ('junior-dev-questions', 'advice',
     "I started my first job as a junior software developer three weeks ago and I feel like I'm "
     "asking my teammates too many questions. How should I handle this without slowing myself "
     "down or annoying them?", None),
    ('lighthouse-story', 'short story',
     "Write a short story, about 300 words, about a lighthouse keeper who finds a message in a "
     "bottle that is addressed to her by name.", None),
    ('summarise-garages', 'summarise',
     "Summarise the following paragraph in two sentences:\n\n" + SUMMARY_PARAGRAPH, None),
    ('sqlite-vs-postgres', 'compare',
     "I'm building a small web app for my hiking club, maybe 200 users. Should I use SQLite or "
     "PostgreSQL? Compare the two for this case and give me a recommendation.", None),
    ('leaky-faucet', 'how-to',
     "My single-handle kitchen faucet drips constantly even when it's fully off. How do I fix it "
     "myself? Tell me what tools and parts I need and the steps.", None),
    ('landlord-email', 'email',
     "Write a polite but firm email to my landlord asking them to repair the heating in my "
     "apartment. It has not worked for eight days, and I already reported it by phone last "
     "week.", None),
    ('train-meeting', 'reasoning',
     "My train leaves at 9:40 am and the trip takes 2 hours 35 minutes. I have to change trains "
     "once, which adds a 20-minute wait. From the arrival station it's a 15-minute walk to my "
     "meeting, which starts at 1:00 pm. Will I make it on time, and with how much margin? Show "
     "your reasoning.",
     "9:40 + 2:35 = 12:15; + 20 min wait = 12:35 arrival; + 15 min walk = 12:50. Yes, with "
     "10 minutes to spare."),
    ('nonetype-error', 'debugging',
     "What does this Python error mean, and how do I fix it?\n\n"
     "```python\n"
     "def find_user(users, name):\n"
     "    for u in users:\n"
     "        if u['name'] == name:\n"
     "            return u\n"
     "\n"
     "user = find_user(users, 'dana')\n"
     "print(user['email'])\n"
     "```\n\n"
     "TypeError: 'NoneType' object is not subscriptable", None),
    ('pantry-dinner', 'cooking',
     "I have chicken thighs, rice, one onion, garlic, a can of chopped tomatoes, and basic spices "
     "(salt, pepper, paprika, cumin). Suggest a simple dinner for two and give me the steps.",
     None),
    ('lisbon-itinerary', 'planning',
     "Plan a relaxed three-day itinerary for my first visit to Lisbon in October. I like food, "
     "views, and walking, but I don't want to rush.", None),
    ('sky-blue-kid', 'explanation',
     "Explain to a curious 10-year-old why the sky is blue but sunsets are orange.", None),
    ('rewrite-concise', 'rewrite',
     "Rewrite this paragraph so it's clear and concise, keeping the meaning:\n\n" + WORDY_PARAGRAPH,
     None),
    ('bakery-names', 'brainstorm',
     "Give me 10 name ideas for a small neighborhood bakery that specializes in sourdough bread. "
     "Add a one-line reason for each.", None),
    ('beginner-running', 'advice',
     "I've never run before and I'm a bit overweight. Give me a safe beginner plan for my first "
     "four weeks of running, and tell me what to watch out for.", None),
    ('crisis-2008', 'explanation',
     "What caused the 2008 financial crisis? Give me a short overview a non-economist can "
     "follow.", None),
    ('buy-vs-lease', 'decision',
     "I drive about 8,000 miles a year and usually keep cars for a long time. Should I buy or "
     "lease my next car? Walk me through the trade-offs.", None),
    ('translate-memo', 'translation',
     "Translate this into French and into Spanish: \"The meeting has been moved to Thursday at "
     "3 pm. Please bring the quarterly report and let me know if you can't attend.\"", None),
    ('large-files-linux', 'how-to',
     "How can I find all files larger than 100 MB on my Linux machine and list them sorted by "
     "size? Explain the command you give me.", None),
    ('cat-keyboard-poem', 'poem',
     "Write a short, funny poem about a cat who refuses to get off my laptop keyboard while I'm "
     "trying to work.", None),
]


def download(name):
    """Fetch one pinned dataset file from the Hub; return its rows as dicts."""
    from huggingface_hub import hf_hub_download
    src = SOURCES[name]
    path = hf_hub_download(src['repo'], src['file'], repo_type='dataset',
                           revision=src['revision'])
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def pick(name, rows, seed):
    """Sorted indices of a fixed random subset; the RNG is seeded per dataset."""
    rng = random.Random(f'{seed}-{name}')
    return sorted(rng.sample(range(len(rows)), SOURCES[name]['n']))


def source(name, index):
    src = SOURCES[name]
    return {'dataset': src['repo'], 'revision': src['revision'], 'index': index}


def user(text):
    return [{'role': 'user', 'content': text}]


def gsm8k_items(seed):
    rows = download('gsm8k')
    out = []
    for i in pick('gsm8k', rows, seed):
        r = rows[i]
        gold = r['answer'].split('####')[-1].strip().replace(',', '')
        if not re.fullmatch(r'-?\d+(\.\d+)?', gold):
            raise ValueError(f'GSM8K row {i}: unexpected gold answer {gold!r}')
        out.append(dict(id=f'gsm8k-{i:04d}', category='math',
                        messages=user(r['question'] + GSM8K_SUFFIX),
                        meta=dict(answer=gold, solution=r['answer'], source=source('gsm8k', i))))
    return out


def humaneval_items(seed):
    rows = download('humaneval')
    out = []
    for i in pick('humaneval', rows, seed):
        r = rows[i]
        content = f"{HUMANEVAL_PREFIX}\n```\n{r['prompt'].strip()}\n```\n"
        num = r['task_id'].split('/')[-1]
        out.append(dict(id=f'humaneval-{int(num):03d}', category='code', messages=user(content),
                        meta=dict(task_id=r['task_id'], prompt=r['prompt'], test=r['test'],
                                  entry_point=r['entry_point'],
                                  canonical_solution=r['canonical_solution'],
                                  source=source('humaneval', i))))
    return out


def ifeval_items(seed):
    rows = download('ifeval')
    out = []
    for i in pick('ifeval', rows, seed):
        r = rows[i]
        out.append(dict(id=f"ifeval-{r['key']}", category='ifeval', messages=user(r['prompt']),
                        meta=dict(key=r['key'], prompt=r['prompt'],
                                  instruction_id_list=r['instruction_id_list'],
                                  kwargs=r['kwargs'], source=source('ifeval', i))))
    return out


def chat_items():
    out = []
    for n, (topic, kind, prompt, reference) in enumerate(CHAT_PROMPTS, 1):
        meta = dict(topic=topic, kind=kind)
        if reference:
            meta['reference'] = reference
        out.append(dict(id=f'chat-{n:02d}', category='chat', messages=user(prompt), meta=meta))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', default=EVAL_SET)
    args = ap.parse_args()

    items = gsm8k_items(args.seed) + humaneval_items(args.seed) + ifeval_items(args.seed)
    items += chat_items()
    ids = [it['id'] for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate ids in the set')
    write_jsonl(args.out, items)

    counts = {}
    for it in items:
        counts[it['category']] = counts.get(it['category'], 0) + 1
    print(f'wrote {len(items)} items to {args.out}: '
          + ', '.join(f'{k} {v}' for k, v in counts.items()))


if __name__ == '__main__':
    main()
