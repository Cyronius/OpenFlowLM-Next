"""Shared pieces of the dllm-ref evaluation: the model registry, tokenizer loading, JSONL I/O.

Used by tokenize_set.py, score.py and run_all.py. Nothing here runs a model.

The registry records, per model, where its HF tokenizer and chat template live, which GGUF
stem it ships as, which dllm-ref policy drives it, how its authors prompt it, and which
token ids end a turn or must never reach the detokeniser (mask and edit-control ids).
"""
import json
import os

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_SET = os.path.join(EVAL_DIR, 'eval_set.jsonl')

# Models, GGUFs and results live under $DLLM_ROOT (default ~/dllm): hf/, gguf/, results/.
DLLM_ROOT = os.environ.get('DLLM_ROOT') or os.path.join(os.path.expanduser('~'), 'dllm')
HF_ROOT = os.path.join(DLLM_ROOT, 'hf')
GGUF_ROOT = os.path.join(DLLM_ROOT, 'gguf')
RESULTS_ROOT = os.path.join(DLLM_ROOT, 'results')

CATEGORIES = ('math', 'code', 'ifeval', 'chat')

# Generation budget per category. SDAR's driver output comes back in whole 16-token blocks,
# so it can run up to 15 tokens past these; nothing downstream depends on the exact cap.
DEFAULT_MAX_NEW = {'math': 512, 'code': 512, 'ifeval': 768, 'chat': 768}

# How each model is prompted, from its own chat template, generation_config and README:
#
# wedlm  - Qwen2-style ChatML template. With no system message the template itself inserts
#          "You are a helpful assistant."; the README's examples pass a lone user turn and
#          nothing else, so neither do we. No thinking mode.
# sdar   - Qwen3's template (no default system prompt). SDAR's generate.py and its README
#          pass a lone user turn with no enable_thinking flag, which renders a plain
#          "<|im_start|>assistant\n" -- reproduced here.
# llada2 - its own template, the NO-THINK variant: "<role>user</role>...<|role_end|>
#          <role>assistant</role>", no system block unless one is given. The README's
#          example passes a lone user turn. Its tokenizer class is custom code in the HF
#          folder (tokenization_llada2.py), hence trust_remote_code.
# qwen3  - Qwen3's template with enable_thinking=False (non-thinking mode, the yardstick),
#          which appends an empty "<think>\n\n</think>\n\n" to the generation prompt.
#
# stop_ids: any of these in out_ids ends the response (everything from it on is dropped).
# strip_ids: removed before detokenising (mask rows, LLaDA2.2's DELETE/INSERT controls).
MODELS = {
    'wedlm': dict(
        hf_dir=os.path.join(HF_ROOT, 'WeDLM-8B-Instruct'),
        gguf_stem='WeDLM-8B-Instruct',
        policy='wedlm',
        trust_remote_code=False,
        template_kwargs={},
        stop_ids=(151643, 151645),        # <|endoftext|>, <|im_end|>
        strip_ids=(151665,),              # mask: an unused embedding row
    ),
    'sdar': dict(
        hf_dir=os.path.join(HF_ROOT, 'SDAR-8B-Chat-b16'),
        gguf_stem='SDAR-8B-Chat-b16',
        policy='sdar',
        trust_remote_code=False,
        template_kwargs={},
        stop_ids=(151645, 151643),        # generation_config eos_token_id
        strip_ids=(151669,),              # <MASK>
    ),
    'llada2': dict(
        hf_dir=os.path.join(HF_ROOT, 'LLaDA2.2-mini'),
        gguf_stem='LLaDA2.2-mini',
        policy='llada2',
        trust_remote_code=True,
        template_kwargs={},
        stop_ids=(156892, 156900),        # <|endoftext|>, <|role_end|>
        strip_ids=(156895, 156930, 156931),  # <|mask|>, DELETE, INSERT
    ),
    'qwen3': dict(
        hf_dir=os.path.join(HF_ROOT, 'Qwen3-8B-tokenizer'),
        gguf_stem='Qwen3-8B',
        policy='ar',
        trust_remote_code=False,
        template_kwargs={'enable_thinking': False},
        stop_ids=(151645, 151643),        # generation_config eos_token_id
        strip_ids=(),
    ),
}


def load_tokenizer(model):
    """The model's HF tokenizer, with its chat template, from the local HF folder."""
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from transformers import AutoTokenizer
    spec = MODELS[model]
    return AutoTokenizer.from_pretrained(spec['hf_dir'],
                                         trust_remote_code=spec['trust_remote_code'])


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def be_quiet():
    """Drop this process (and the children it starts) to below-normal priority, so
    tokenising or scoring yields to a benchmark sharing the box. Not idle priority: a
    starved HumanEval program would hit its timeout and score as a failure."""
    if os.name == 'nt':
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
        except Exception:
            pass
    else:
        try:
            os.nice(10)
        except Exception:
            pass
