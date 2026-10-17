---
layout: docs
title: whisper
nav_order: 7
parent: Models
---

## 🧩 Model Card: [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)

- **Type:** Speech-to-Text (ASR: Automatic Speech Recognition)
- **Think:** No
- **Tool Calling Support:** No
- **Base Model:** [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- **Quantization:** Q4_1
- **Max Output Length:** 448 tokens (fixed)
- **Maximum Encoder Window:** 30 seconds of audio (1500 frames)

> ⚠️ **The shipped `whisper-v3:turbo` container runs the closed engine.** It ships
> only `model.q4nx`. The open Whisper engine reads OpenAI's own checkpoint or a GGUF
> of it instead, with no conversion step; install one with `oflm-add` (below). Its
> decoder runs entirely on the host, with int8 weights by default (`OW_DEC_W=bf16`
> restores the exact path) -- the open Whisper README makes no NPU performance claim
> for it.

### The open engine: OpenAI's checkpoint or a GGUF

```shell
oflm-add handy-computer/whisper-large-v3-turbo-gguf             # registers whisper-v3:turbo-q8_0
oflm-add handy-computer/whisper-large-v3-turbo-gguf --quant F16 # or Q6_K, Q5_K_M, Q4_K_M
oflm-add openai/whisper-large-v3-turbo                          # registers whisper-v3:turbo-safetensors
oflm serve --asr 1 --asrmodel whisper-v3:turbo-q8_0
```

The engine loads the file as it ships: a GGUF's tensors are widened to the engine's
bf16 at load, so a smaller quant saves download and disk, not memory. Each quant
installs into its own directory, so two of them can sit side by side. A GGUF holds
no tokenizer or config files; `oflm-add` fetches them from the checkpoint the GGUF
names (here `openai/whisper-large-v3-turbo`) and refuses the install if they do not
match the GGUF's own vocabulary and settings.

The open engine also needs its Whisper kernel set. Releases ship it with the other
kernel sets. In a clone, build it once with `python open_kernels/export_whisper_kernels.py`
(it needs the IRON toolchain and writes `src/xclbins/Whisper-V3-Turbo-NPU2/open_kernels`).
The engine finds the set by the model's geometry, so any install directory works, and
`oflm-add` says whether it found one.

▶️ Run with OpenFlowLM in PowerShell:  

> The ASR model must be used with an LLM (loaded concurrently) in CLI Mode.
> The ASR model can be used as an independent ASR model in Server Mode .

### CLI Mode   

Start with ASR enabled: 

Load the ASR model (whisper-v3:turbo) in the background, with concurrent LLM loading (gemma3:4b).
```shell
oflm run gemma3:4b --asr 1 
```
or
```shell
oflm run gemma3:4b -a 1 
```

Then, type (replace `filename.mp3` with your audio file path):
```shell
/input "path\to\audio_sample.mp3" summarize it
```

### Server Mode 

Start with ASR enabled: 

- Load the ASR model (whisper-v3:turbo) in the background, with concurrent LLM loading (gemma3:4b).
```shell
oflm serve gemma3:4b --asr 1 
```
or
```shell
oflm serve gemma3:4b -a 1 
```

- Load the ASR model (whisper-v3:turbo) as a standalone ASR model.
```shell
oflm serve --asr 1 
```
or
```shell
oflm serve -a 1 
```

Send audio to `POST /v1/audio/transcriptions` via any OpenAI Client or Open WebUI.

> see more API details here → [/v1/audio/](https://platform.openai.com/docs/api-reference/audio)

**Example 1**: OpenAI Client

```python
# Import the official OpenAI Python SDK (OpenFlowLM mirrors the OpenAI API schema)
from openai import OpenAI

# Initialize the client to point at your local OpenFlowLM server
# - base_url: OpenFlowLM's local OpenAI-compatible REST endpoint
# - api_key: Dummy token; OpenFlowLM typically doesn't enforce auth, but the client requires a string
client = OpenAI(
    base_url="http://127.0.0.1:52625/v1",  # OpenFlowLM local API endpoint
    api_key="oflm",                         # Placeholder key
)

# Open the audio file in binary mode and create a transcription request
# - model: name of the speech-to-text model exposed by OFLM (e.g., "whisper-v3:turbo")
# - file: file-like object pointing to your audio
with open("audio.mp3", "rb") as f:
    resp = client.audio.transcriptions.create(
        model="whisper-v3:turbo",
        file=f,
    )

# Print the transcribed text returned by the server
print(resp.text)
```

**Example 2**: Open WebUI

- Follow Open WebUI setup [guide](https://openflowlm.com/docs/instructions/server/webui/).
- In the bottom-left corner, click User icon, then select Settings.
- In the bottom panel, open Admin Settings.
- In the left sidebar, navigate to Audio.
- Set Speech-to-Text Engine to OpenAI.
- Enter:
> API Base URL: `http://127.0.0.1:52625/v1` (Open WebUI Desktop) or `http://host.docker.internal:52625/v1` (Open WebUI in Docker)   
> API KEY: oflm (any value works)    
> STT Model: whisper-v3:turbo (type the tag; can be a different ASR model)    
- Save the setting.
- You're ready to upload audio files! (Choose an LLM to load and use concurrently)

---
