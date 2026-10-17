# The OpenAI-compatible server

Everything a client can see on the wire: the HTTP status, the `model` field, `finish_reason`,
the shape of an error body, and the embeddings task prompt. The code is `src/server/` --
`server.cpp`'s responder lambda, `rest_handler.cpp`'s handlers, `openai_compat.hpp`'s wire
vocabulary and `streaming_ostream_openai.hpp`'s chunks. None of it is about the weights: every
requirement here holds for whichever model happens to be loaded, closed or open kernels alike.

Directory name gives the prefix: `SERVER`.

These landed as fixes in [#52](https://github.com/Cyronius/OpenFlowLM-Next/pull/52) and
[#59](https://github.com/Cyronius/OpenFlowLM-Next/pull/59) with no requirement IDs at all, which
is why this file exists. The background for each is in the code comment next to the fix, and the
defect-by-defect account is in `.claude/plans/oflm-test-server-conformance.md` (local, gitignored).

Every one of those defects has the same shape: a wrong answer that is **well formed**. HTTP 200
with an error body parses. A truncated answer reads like a complete one. A vector embedded under
the wrong task prompt is correctly shaped, correctly normed and deterministic. A smoke test that
asks "did the server answer?" passes through all of them, which is why the criteria below assert
the metadata rather than the content.

The decisions themselves are pure functions -- `status_for`, `finish_reason`, `preflight`,
`resolve_task`, `task_policy` -- and they already have unit tests in
`src/server/openai_compat_test.cpp`, which need no device, no weights and no network. The
requirements here are the other half: that the server actually puts those decisions on the wire,
in both response modes, on every endpoint.

The operator-facing version of these checks is `oflm-test --api` (checks A1 to A5) and
`oflm-test --embedding` (checks E10 and E11), which report a verdict per check and fail the run.
The tests under `tests/` assert the same behaviour a second time, deliberately: they import
nothing but the Python standard library, so a bare checkout can run them against a running server
without installing the `oflm-test` package first.

## Requirements

### SERVER-ERROR-STATUS: an error body never comes back with a 2xx status
**Applies to:** openflowlm-next (`src/server/server.cpp`, `src/server/openai_compat.hpp`)
**Test category:** integration (through `oflm serve`)
**Test:** `specs/server-api/tests/test_error_status.py`

The responder lambda used to read `error.code` as an int to pick the HTTP status. In the OpenAI
error shape that field is a string -- `"model_not_found"`, `"invalid_value"` -- so the read threw
`type must be number, but is string` out of the lambda, the outer handler turned the exception
into `{"error": "<exception text>"}`, and it went out with the 200 the lambda had already set.
Every OpenAI-shaped error the server built was invisible: clients saw a success carrying an
`error` key, and the SDKs, which look at the status first, saw nothing wrong at all.

The server shall give every response that carries a top-level `error` a status of 400 or above.
`openai_compat::status_for()` is the whole rule and it accepts both spellings of `code`: a numeric
one in the 400-599 range is taken as the status, and otherwise `type` decides
(`invalid_request_error` is 400, `authentication_error` 401, `permission_error` 403,
`not_found_error` 404, `rate_limit_error` 429). An error object it cannot classify is 500, and so
is the flat `{"error": "<text>"}` shape that the handlers' own catch blocks still emit, because
200 is the one answer that is certainly wrong.

**Acceptance criteria:**
- A chat request naming a model the server cannot load returns HTTP 400 and a body whose `error`
  is an object with `message`, `type`, `param` and `code`.
- That `code` is the JSON string `"model_not_found"`. A number there is the defect, not a variant
  spelling: the string is what the OpenAI shape says and what every client library expects.
- No response carrying a top-level `error` has a 2xx status, over at least these probes: an
  unknown model on `/v1/chat/completions`, the same on `/v1/completions`, an explicit
  `"model": ""`, and the `"model-faker"` sentinel.
- A request body that is not JSON is refused with a non-2xx status, and the next well-formed
  request is answered normally. Reporting the offending bytes back used to throw a second time
  inside the catch block, which left the NPU lock held and hung every later request, valid ones
  included; `safe_dump()` substitutes U+FFFD rather than throwing, so the error path cannot be
  broken by the input that made it run.

### SERVER-MODEL-IDENTITY: a model the server cannot load is refused, never substituted
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/server/openai_compat.hpp`)
**Test category:** integration (through `oflm serve`)
**Test:** `specs/server-api/tests/test_error_status.py`,
`specs/server-api/tests/test_embed_task_prompt.py` (the embeddings half)

`get_auto_model()` returned a Llama-3.2-1B engine for any tag it did not recognise, so a request
for a model this build has never heard of was answered fluently by a different model -- HTTP 200,
with the client's own requested tag echoed back in `model`, so a client comparing the response to
its request saw agreement. The embeddings endpoint had the same defect with the vectors: started
with `bge-base:en-v1.5` loaded, a request for `gte-multilingual:base` returned bge-base's vectors
byte for byte under the name it had been asked for.

The server shall refuse a request whose `model` it cannot serve, with 400 and `model_not_found`,
and shall resolve the tag before unloading anything, so a typo in a client's model field does not
evict the model that is serving everyone else. An accepted request reports the model it was asked
for. The three client mistakes -- unknown tag, not a chat model, no model loaded at all -- all
answer 400; a model that is known but fails to load is ours and answers 500.

**Acceptance criteria:**
- `"model": "oflm-test-no-such-model:0b"` on `/v1/chat/completions` returns 400 with
  `error.code == "model_not_found"`, and the body carries no `choices`.
- The same request with `stream: true` is refused the same way: a JSON error body with a 400
  status, not an SSE stream.
- An explicit `"model": ""` and the `"model-faker"` sentinel are refused with `model_not_found`
  rather than served by whatever is loaded; an *omitted* `model` field is served by the loaded
  model, which is how every client that names no model works.
- An accepted request's response `model` equals the tag the request asked for.
- After a refused request, a request for the served model is answered normally -- the refusal
  unloaded nothing.
- On `/v1/embeddings`, a request naming any model other than the loaded one returns 400
  `model_not_found` and a message naming what *is* loaded, rather than the loaded model's vectors
  under the requested name.

### SERVER-FINISH-REASON: finish_reason reports what actually stopped generation
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/server/openai_compat.hpp`, `src/server/streaming_ostream_openai.hpp`)
**Test category:** integration (through `oflm serve`)
**Test:** `specs/server-api/tests/test_finish_reason.py`

The engine computes `meta_info.stop_reason` and the handler dropped it, hardcoding `"stop"` in
the response. An answer cut off at `max_tokens` was therefore indistinguishable from one that had
finished -- the failure mode is an agent that reads a half-written tool call, or a summariser that
silently truncates, with nothing in the response to say so.

The server shall map the engine's stop reason into OpenAI's `finish_reason` vocabulary and report
it in both response modes: `"length"` when generation hit the token limit, `"tool_calls"` when it
stopped on a tool call, `"stop"` otherwise. The mapping is one function, `openai_compat::finish_reason()`,
because the engine's own vocabulary is wider than OpenAI's -- it also yields `"cancel"`, `"error"`
and `"UNKNOWN"`, none of which are values the OpenAI schema has, and a streamed cancellation used
to go out as `"cancel"`.

**Acceptance criteria:**
- A request with `max_tokens: 8` on a prompt that cannot be answered in 8 tokens reports
  `finish_reason: "length"`, non-streaming and streaming alike.
- A request that finishes on its own inside a generous `max_tokens` reports `"stop"`.
- `finish_reason` is always one of OpenAI's five values (`stop`, `length`, `tool_calls`,
  `content_filter`, `function_call`); the engine's `cancel`, `error` and `UNKNOWN` never reach a
  client.

### SERVER-PARAM-ISOLATION: a request gets the model's defaults for the fields it omits
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`)
**Test category:** integration (through `oflm serve`)
**Test:** `specs/tool-calling/tests/test_request_params_reset.py`

The engine-side requirement is `TOOLS-REQUEST-PARAMS-RESET` in `specs/tool-calling/spec.md`,
which states the rule and owns the list of affected settings; this one exists only so the
server's API surface is covered under the `SERVER` prefix too, and states the client-observable
half. Read the other one for the mechanism. The behaviour is one server serving many clients:
one client sending `reasoning_effort: "high"` turned thinking on for everybody until the next
request that happened to set it back.

**Acceptance criteria:**
- Two identical requests that omit `reasoning_effort` get the same answer regardless of what a
  request between them set it to.
- The same holds for `temperature`: a request that omits it samples at the model's load-time
  default, not at the `0` a previous request asked for.

### SERVER-EMBED-TASK-PROMPT: the request's task prompt reaches the model, or the request is refused
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/server/openai_compat.hpp`)
**Test category:** integration (needs an embedding server: `oflm serve -e 1`, or
`oflm serve <chat model> --embed 1 --embeddingmodel <tag>`)
**Test:** `specs/server-api/tests/test_embed_task_prompt.py`

Models like nomic-embed-text prepend a short per-task prefix to the text before embedding it, and
which prefix is chosen changes the vector materially -- measured on this server, the same text
under `search_query` against `search_document` is cosine 0.914, not 1. The handler passed
`task_query` unconditionally and ignored the request, so every *document* in a RAG index was
embedded as a *query*. Nothing downstream could tell: the vector is correctly shaped, correctly
normed and deterministic either way. Retrieval just gets quietly worse.

The server shall read the request's `prompt_name` (or its alias `task_type`), map it onto the
model's own prompt table, and pass the result to the engine. Where it cannot do that it shall
refuse, for the same reason: a substituted task prompt is indistinguishable from the right one.
Which refusal depends on what the model offers, and there are three distinct cases, because an
empty prompt table means two different things -- a model with no task concept at all (the BERT
family: bge, all-minilm, gte-multilingual) and one whose prefixes are hardcoded rather than
declared (OpenGemma), which is why `supports_task_prompts()` is asked separately from
`prompt_names()`.

The names the REST API accepts are its own vocabulary, not the container's: a container declares
names like `Retrieval-query`, and `openai_compat::task_names()` maps the REST spellings onto
them. An error message must quote the REST names, because quoting the container's sent clients to
values the validator then rejected.

**Acceptance criteria:**
- A model that declares prompt names, sent a request with neither `prompt_name` nor `task_type`:
  400 with `error.code == "missing_required_parameter"`, `param` `prompt_name`, and a message
  listing the accepted names. It does not pick one.
- A model with no task concept, sent a `prompt_name`: 400 with `error.code == "invalid_value"`.
  It does not embed the text unprefixed and answer 200.
- An unknown name such as `"not_a_task"`: 400 `invalid_value`, and on a model that has prompts
  the message quotes the offending value and lists the accepted ones (`search_query`,
  `search_document` among them). A non-string value is refused the same way.
- A model that honours prompts: the same text under `search_query` and under `search_document`
  comes back as two materially different vectors -- never identical, and cosine well under
  0.9999 (0.914 on nomic-embed-text) -- while each is bit-for-bit reproducible under its own
  prompt.
- `task_type` is an accepted alias: it alone produces the same vector as the same value sent as
  `prompt_name`. Sending both is fine when they name the same task; when they disagree the
  request is refused with 400 `invalid_value` naming both fields, rather than resolved by
  precedence.
- A model that has prompts but none serving the requested task refuses with 400 `invalid_value`
  carrying the engine's own message, rather than reaching the handler's catch block and going out
  as a 200.

### SERVER-STREAM-PARITY: both response modes report the same metadata
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/server/streaming_ostream_openai.hpp`)
**Test category:** integration (through `oflm serve`)
**Test:** `specs/server-api/tests/test_finish_reason.py`

Streaming and non-streaming are two code paths that build their responses independently, and
every metadata fix so far has had to be made twice. `finish_reason` is the example: the
non-streaming responder mapped it and the streaming one did not. A check that exercises one mode
proves nothing about the other, so this requirement says they agree, and the tests run the same
request both ways.

**Acceptance criteria:**
- The same request answered in both modes reports the same `model` and the same `finish_reason`.
- The stream ends with a chunk that carries a non-null `finish_reason` in `choices[0]`, followed
  by `data: [DONE]`. Content chunks before it carry `finish_reason: null`.
- A request refused before generation starts (an unknown model, say) is refused identically in
  both modes: a JSON error body with a non-2xx status, not an SSE stream that opens and then
  stops.

### SERVER-REQUEST-VALIDATION: a malformed request is refused with a 400, and the server keeps serving
**Applies to:** openflowlm-next (`src/server/server.cpp`, `src/server/rest_handler.cpp`, `src/server/openai_compat.hpp`)
**Test category:** integration (through `oflm serve`)
**Test:** `specs/server-api/tests/test_request_validation.py`

Handlers read required fields with `request["field"]` on a `const json&`. For a key that is not
there that is undefined behaviour, and on this build it segfaults: `POST {}` killed the server on
`/api/show`, `/api/generate`, `/v1/completions` and `/v1/embeddings` (#70). A `request_id` that
is not a string threw before the handler ran, while the request held the NPU lock -- the same
leaked lock SERVER-ERROR-STATUS describes for a body that is not JSON, reached by a different
input. The process stayed up and `/v1/models` kept answering, because it does not take the lock,
so checking that a server survives a bad request means sending it a request that needs the NPU.

This extends two existing checks rather than replacing them: SERVER-ERROR-STATUS already covers a
body that is not JSON, and `oflm-test --api` check A1 already sends `/v1/chat/completions` a
request with no `messages` and one with `"messages"` as a string. The tests below repeat those two
probes, as the tests in this directory do for every `oflm-test` check.

The server shall check a request's required fields, and the type of `request_id`, before reading
them, and refuse a missing or mistyped one with 400 and an OpenAI-shaped error naming the field.
The embeddings endpoint shall also refuse a malformed `input` or `model`, and a server with no
embedding model loaded shall refuse an embeddings request rather than answer 200 with no body.

**Acceptance criteria:**
- `POST {}` to `/api/show`, `/api/generate`, `/api/chat`, `/v1/completions`,
  `/v1/chat/completions` and `/v1/embeddings` returns 400 with `error.param` naming the required
  field (`model`, `prompt`, `messages` or `input`), and after each one a chat request is answered.
- A required field of the wrong type (`"prompt": 5`, `"messages": "hi"`) returns 400 with
  `error.code == "invalid_value"`.
- A non-string `request_id` returns 400 with `error.param == "request_id"`, and a chat request
  sent after it is answered rather than left queued. A string `request_id` is accepted.
- On `/v1/embeddings`: `input` that is `null`, a number or an object returns 400; an array with a
  non-string element returns 400 with `param` naming that element (`input[1]`); `"model": null`
  returns 400; `"model": ""` returns 400 `model_not_found`; an omitted `model` is served by the
  loaded model and the response names it. `"input": []` stays a 200 with an empty `data`.
- A server started without an embedding model answers `/v1/embeddings` with 400
  `model_not_found`, not 200.

## The Images API

`POST /v1/images/generations` and `POST /v1/images/edits` are OpenAI's Images API over the open
diffusion engine (`src/open_diffusion`, FLUX.2 [klein] 4B with every op on the NPU; its own
requirements are in `specs/open-diffusion/spec.md`). The handlers are `rest_handler.cpp`'s
`handle_openai_images_*`. The request rules are one pure function, `openai_compat::images_request()`,
unit-tested in `src/server/openai_compat_test.cpp`. The plan these came from is
`archive/images-api.md`.

There is no `/v1/images/variations` (OpenAI no longer documents it), no `url` response format,
no `partial_images` streaming (a preview costs a full VAE decode per step: +35% time), no
A1111 `/sdapi` shim and no ComfyUI `/prompt`.

The integration tests are `specs/server-api/tests/test_images_api.py`: `oflm serve <chat model>`
with `flux2-klein:4b` installed, and the NPU.

### SERVER-IMAGES-GENERATIONS: a request gets images of the size and format it asked for
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/server/server.cpp`, `src/include/model_list.hpp`)
**Verification:** test
**Test:** `specs/server-api/tests/test_images_api.py`

`/v1/images/generations` answers `{created, data: [{b64_json, seed}], model, output_format, size}`.
`data` holds `n` images (1-10), made one after another with seeds `seed`, `seed + 1`, ...; each
item's `seed` says which, so a random-seed image can be made again. `output_format` is `png`
(the default) or `jpeg` (quality from `output_compression`, default 90). `webp` is refused as not
implemented: it needs `libwebp`. The response's `model` is the resolved tag.

The `model` field follows SERVER-MODEL-IDENTITY: it is resolved before anything is unloaded, a tag
the build does not have, an explicit `""` and a tag that makes no images are each a 400
`model_not_found`, and the llama3.2:1b fallback never applies. An omitted `model` is `--imagemodel`'s
(default `flux2-klein:4b`). `/v1/models` lists the image model; `/api/tags`, which chat clients
read, does not.

**Acceptance criteria:**
- `{"model": "flux2-klein:4b", "prompt": ..., "size": "512x512", "seed": 1}` returns 200, one
  `data` item whose `b64_json` decodes to a 512x512 PNG, an integer `created`, and
  `model == "flux2-klein:4b"`.
- `output_format: "jpeg"` gives a 512x512 JPEG.
- `n: 2, seed: 41` gives two different images with `seed` 41 and 42, and the second is byte for
  byte what `seed: 42` alone gives.
- The same request twice gives the same `b64_json`.
- `output_format: "webp"` is a 400 with `param == "output_format"` and `code == "not_implemented"`.
- `"model": "oflm-test-no-such-model:0b"`, `"model": ""` and a chat model's tag are 400
  `model_not_found`.
- `/v1/models` lists `flux2-klein:4b` and `/api/tags` does not.

### SERVER-IMAGES-PARAMS: one field under two names means one thing, and a refused control names itself
**Applies to:** openflowlm-next (`src/server/openai_compat.hpp`, `src/server/rest_handler.cpp`)
**Verification:** test
**Test:** `src/server/openai_compat_test.cpp` (`test_images_request`, the `openai_compat` CTest);
`specs/server-api/tests/test_images_api.py` (on the wire)

Clients written for vLLM-Omni send diffusers' names and clients written for Lemonade or A1111 send
A1111's, for the same controls. Each pair is one field: `steps` = `num_inference_steps`,
`cfg_scale` = `guidance_scale`, `sampler` = `sampler_name`. Both spellings in one request are
accepted when they agree and refused (400, `param` naming the second) when they do not: neither
silently wins.

- `steps`: 1-50, default the model's own (4 for klein, which is distilled for it).
- `seed`: a non-negative integer up to 2^64-1; `-1` (A1111's) and omitted mean a random seed.
- `cfg_scale` / `guidance_scale` (a number) and `negative_prompt` (a string) are accepted and
  ignored, with one log line: klein is guidance-distilled and has no CFG, and A1111 clients always
  send `cfg_scale: 7, negative_prompt: ""`.
- `sampler` / `sampler_name`: `euler`, `Euler`, `Euler a`, `flowmatch_euler` and
  `FlowMatchEulerDiscreteScheduler` are the model's flow-match Euler; any other name is a 400 that
  lists those.
- `response_format` must be `b64_json` (`url` is a 400 naming it); `stream: true` and
  `partial_images` > 0 are 400s naming them.
- Types are checked: a wrong one is a 400 with `param` naming the field. A float is not an
  integer, even `2.0`. `null` is the same as leaving the field out.
- Fields this server does not know (`quality`, `style`, `background`, `user`, ...) are ignored.

**Acceptance criteria:**
- `steps: 8, num_inference_steps: 8` is accepted; `steps: 8, num_inference_steps: 4` is a 400
  with `param == "num_inference_steps"`. The same holds for `cfg_scale`/`guidance_scale` (4 and
  4.0 agree) and `sampler`/`sampler_name`.
- `cfg_scale: 7, negative_prompt: ""` is accepted, and on the wire an image made with
  `guidance_scale: 3.5, negative_prompt: "blurry", sampler: "euler"` is byte for byte the plain
  request's.
- `sampler_name: "DPM++ 2M Karras"` is a 400 whose message lists `Euler a`.
- `steps: 2` runs and gives a different image from the default's.
- `n` 0, 11 or 2.0; `steps` 0 or 51; `seed` -2 or 1.5; `stream: true`; `partial_images: 1`;
  `response_format: "url"`; `prompt: 5`: each is a 400 naming its field.
- `seed: 18446744073709551615` is accepted; `seed: -1` is a random seed.

### SERVER-IMAGES-SIZE: only the engine's resolutions run
**Applies to:** openflowlm-next (`src/server/openai_compat.hpp`)
**Verification:** test
**Test:** `src/server/openai_compat_test.cpp`, `specs/server-api/tests/test_images_api.py`

`size` is `"WxH"` or `"auto"`. The sizes that run are the model's `image_sizes` in
`model_list.json`, which for klein are 512x512 and 1024x1024 (OPEN-DIFFUSION-RESOLUTIONS); `auto`
and an omitted size are the largest, 1024x1024. Any other size is a 400 that names the supported
ones. Non-square sizes need their own stream sets and are a separate item.

**Acceptance criteria:**
- `"512x512"` gives a 512x512 image and `"auto"` a 1024x1024 one.
- `"768x768"` and `"1024x512"` are 400s with `param == "size"` whose message contains
  `512x512, 1024x1024`; `"big"`, `"1024"` and the number `1024` are 400s with `param == "size"`.

### SERVER-IMAGES-EDITS: one reference image is edited; a mask or a second image is named as not implemented
**Applies to:** openflowlm-next (`src/server/server.cpp`, `src/server/multipart.cpp`, `src/server/rest_handler.cpp`)
**Verification:** test
**Test:** `specs/server-api/tests/test_images_api.py`; the multipart parser: `src/server/multipart_test.cpp`
(CTest `multipart`, no device)

klein edits by appending the reference's VAE latents to the joint sequence as tokens; the NPU
encodes the reference first (OPEN-DIFFUSION-EDIT, `specs/open-diffusion/spec.md`).
`/v1/images/edits` parses its multipart form and checks it, before the NPU is touched:
- one or more non-empty `image` / `image[]` files (at most 16), at most one `mask`;
- then the generations' model and control rules, with the model's `image_edit_sizes` for
  `size`.

The request is refused, naming the field:
- a `mask` is a 400 naming inpainting as not implemented;
- a second image is a 400 naming multiple references as not implemented;
- a model without `image_edit_sizes` is a 400 on `model`;
- an image over 32 MB is a 400;
- a reference that can't be prepared is a 400 on `image` with OPEN-DIFFUSION-REFERENCE's
  reason (not PNG or JPEG, too small, too elongated, over 64 MP, undecodable).

`size: auto` (OpenAI's edit default) follows the reference: the largest of the edit sizes
not above its shorter side, else the smallest. The response is the generations' shape:
b64_json images and their seeds, `n` of them from consecutive seeds, with the same
cancellation. An HRX build answers 501.

The multipart parser accepts a quoted boundary (`boundary="..."`, RFC 2046), fills each part's
`content_type`, finds headers case-insensitively, and keeps repeated parts (`image[]`), which used
to overwrite each other. A part's `name` and `filename` come from its own `Content-Disposition`
line and nowhere else: a header is the line that starts with its name, and a quoted parameter
value is read whole (`\"` and `\\` are its escapes). So neither another header nor text inside a
quoted filename can name a part, which would otherwise let a part pass as `image` or `mask`.

**Acceptance criteria:**
- One 512x512 PNG with `size` omitted is edited: 200, `size` "512x512", one 512x512 PNG.
- The same edit request twice (same seed) returns the same bytes.
- Two `image[]` parts, a `model` and a `prompt`, sent with a quoted boundary: a 400 with
  `param == "image"`, its message saying "not implemented".
- An image and a `mask`: a 400 with `param == "mask"`, naming inpainting.
- A BMP as the image: a 400 with `param == "image"` naming "not PNG or JPEG".
- A form with no image is a 400 with `param == "image"`.
- A form with an image and `size: 768x768` is a 400 with `param == "size"`.
- Multipart (`multipart_test`): a part whose `Content-Disposition` has no `name` is not named by
  `name="mask"` on its `Content-Type` line; a header whose value mentions `content-disposition:`
  is not that header; `filename="x; name=mask"; name="image"` is `image` with filename
  `x; name=mask`; an escaped `\"` does not end a quoted filename; `C:\dir\fox.png` keeps its
  backslashes.

### SERVER-IMAGES-NPU: an image request holds the NPU like chat, and always lets it go
**Applies to:** openflowlm-next (`src/server/server.cpp`)
**Verification:** test
**Test:** `specs/server-api/tests/test_images_api.py`

Both image paths are in `requires_npu_access()`: one request at a time on the NPU, queued behind
chat, embeddings and transcription. Every path through the handlers calls `send_response` exactly
once, which is what releases the lock (SERVER-ERROR-STATUS describes what a missed release does).
A client that disconnects during an `n` > 1 request stops it after the current image.

**Acceptance criteria:**
- After each of a refused size, an unknown model, `n: 0` and a request with no prompt, an image
  request is answered 200.
- Chat, image, chat, image on one server are each answered 200.

### SERVER-IMAGES-FAILURE: a failed image run does not leave the server unable to make images
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/open_diffusion/engine.cpp`)
**Verification:** manual

A request that fails once it has used the image engine (a run that errors or times out on the NPU,
an allocation that fails in `select`) is answered 500 with the reason, and the engine is unloaded.
Before `Engine::run` rethrows, it waits out every stretch it still had on the NPU, so no run is
left executing against buffers that are about to be freed. The next image request loads the
engine fresh, with a new hardware context, as a swap does (SERVER-IMAGES-RESIDENCY), with or
without `--imagegen 1`. A refused request never gets this far and unloads nothing.

No client request can make a run fail, so the procedure injects the failure.

**Verification (manual):**
1. Build `oflm` with a local, uncommitted change that makes the first `Engine::run` throw after
   its 20th stretch is submitted, with stretches still in flight.
2. `oflm serve llama3.2:1b`; send `{"prompt": "a red fox in fresh snow", "size": "512x512",
   "seed": 1}` to `/v1/images/generations` three times.
3. The first is a 500 naming the injected failure, and the log shows `unloading the image engine
   after a failed request`. The second loads the engine again (`Loading image model`) and is a
   200. The third is a 200 with the same `b64_json` as the second, and the same as an unpatched
   server gives for that request.
   (2026-10-08: the 500 came with 15 stretches in flight; the second request reloaded the engine
   and took 17.0 s, the third 4.3 s; both images and the unpatched server's were the same bytes.
   Not reproduced: the server's behaviour before this change, which was inferred from the code.)

### SERVER-IMAGES-RESIDENCY: swap by default, both resident with --imagegen 1
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/include/utils/vm_args.hpp`)
**Verification:** manual

The image engine holds one hardware context per configuration, 7.5 GB of weights and 1.4 / 4.6 GiB
of activations per resolution. By default an image request swaps the chat model off the NPU and loads the image
engine (5.2 s warm, the weights in the OS file cache), and a chat request swaps back; the log says
so each time. The chat model's tag is kept, so a chat request that names it, or names nothing, is
served by it again.

`oflm serve <tag> --imagegen 1 [--imagemodel <tag>]` loads the image engine at startup beside the
chat model, allocates every configuration (each resolution and each edit size), and never
swaps. With klein's four (512, 1024, 512e512, 1024e1024) that is 12.8 GiB of activations
(1.33 + 4.22 + 1.70 + 5.58) beside the 7.5 GB of weights. It still shares the NPU lock: resident
saves the load, not the queue. A startup that cannot load it exits naming why. `--imagemodel`
alone sets the model a request naming none gets; a tag that is not an image model stops the
server at startup. Both flags are refused by every other command.

The engine opens its kernel sets on the server's device handle (`npu_device_inst`), as the chat,
embedding and Whisper engines do.

**Verification (manual):**
1. `oflm serve llama3.2:1b`; send chat, image, chat, image (`test_chat_and_images_alternate`).
   All four succeed, and the log shows `swapping the chat model ... off the NPU`, then
   `swapping the image engine ... off the NPU` and `reloading 'llama3.2:1b'`, at each switch.
   (2026-09-28: 3 swaps each way over the whole test file; image engine load 5.2 s.)
2. `oflm serve llama3.2:1b --imagegen 1`: the log shows one `Loading image model` line at startup
   and none after the same four requests. (2026-09-28: loaded in 8.7 s with both resolutions
   allocated; llama3.2:1b and the image engine fit the NPU together.)
3. `oflm serve llama3.2:1b --imagegen 1 --imagemodel llama3.2:3b` exits with `--imagemodel: model
   'llama3.2:3b' is not an image model`; `oflm run llama3.2:1b --imagegen 1` is refused.
4. Not measured: resident beside `--asr 1` and `--embed 1` as well.

## Embeddings and reranking from a decoder LLM

Qwen3-Embedding and Qwen3-Reranker are the Qwen3 network with a different job: the embedder pools
the hidden state at the last token, and the reranker reads two logits ("yes" and "no") at the end
of a fixed prompt. Both run on the open kernels through `OPEN-HIDDEN-STATE`
(`specs/open-engine/spec.md`), read straight from their GGUFs, behind one backend,
`OpenLlmEmbedding` (`src/include/AutoEmbeddingModel/open_llm_embedding.hpp`). The pure rules -- the
text a query becomes, the ids, the prompt, the score, the order and `dimensions` -- are
`llm_conventions.hpp` and `openai_compat.hpp` functions, unit-tested in
`src/server/openai_compat_test.cpp`.

### SERVER-EMBED-QWEN3: an LLM-backed embedder follows the model's own pooling convention
**Applies to:** openflowlm-next (`src/include/AutoEmbeddingModel/`, `src/common/AutoEmbeddingModel/open_llm_embedding.cpp`, `src/server/rest_handler.cpp`, `src/server/openai_compat.hpp`, `utilities/oflm-add`)
**Verification:** test (the text, ids and `dimensions` rules: `src/server/openai_compat_test.cpp`
`test_embed_qwen3`; the install: `utilities/oflm-add/tests/test_llm_embed_install.py`) and manual
(the vectors)

`oflm serve --embed 1 --embeddingmodel qwen3-embedding:0.6b` serves Qwen3-Embedding-0.6B. A query
(`prompt_name` / `task_type` `query` or `search_query`) is embedded as
`Instruct: <instruction>\nQuery:<text>` with the card's web-search instruction; a document
(`document`, `search_document`, `search_result`) is embedded as written; a request naming no task
gets the model's own default, the text as written, which is what its sentence-transformers config
says and what llama.cpp and other OpenAI-compatible servers do. Any other task is refused: the
model declares prompts for query and document only. The ids are the tokenizer's, followed by the
end-of-text id when the GGUF's `tokenizer.ggml.add_eos_token` is true (151643 for Qwen3-Embedding).
The vector is the final-norm hidden state at that last id, L2-normalized. `dimensions` (32 to 1024)
returns the first `dimensions` entries renormalized to unit length (Matryoshka). A model that
cannot shorten its vectors -- every other backend -- refuses `dimensions` with 400 unless it asks
for the model's own width; it used to be ignored silently.

`oflm-add` installs the embedder's GGUF under the `qwen3-embedding` family with label
`embeddings`, never as a chat model, and asks for `--open-kernels` when no official entry carries
the kernel set.

**Acceptance criteria:**
- A query under the web-search task is exactly `Instruct: Given a web search query, retrieve
  relevant passages that answer the query\nQuery:<query>`; a document is the text unchanged; a
  clustering task is refused with `TaskPromptUnavailable`, which the handler answers with 400.
- `"hello"` (id 14990) is pooled over `14990, 151643` when the GGUF asks for the end-of-text id,
  and over `14990` alone when it does not.
- `dimensions: 2` of `(3, 4, 12, 0)/13` is `(0.6, 0.8)`; 16, 2048, 0, -5, `"256"` and 256.5 are
  refused with 400 for a 1024-wide model whose floor is 32; a fixed-width model refuses 256 and
  accepts its own width.
- `oflm-add` derives the `qwen3-embedding` and `qwen3-reranker` families from the repo names (not
  `qwen3`), refuses with a message naming `--open-kernels` when it has none, and with it registers
  the entry with label `embeddings` / `rerank` and links the kernel set at `<model>/open_kernels`.
- Unit length for every input, and a request naming another model returns 400
  (SERVER-MODEL-IDENTITY), as for every embedding backend.

**Verification (manual):** with the embedder installed and served, the card's example (two queries
under `prompt_name: query`, two documents under `document`) gives a 2x2 similarity matrix within
0.02 of the CPU reference's (transformers, fp32, the official safetensors); `oflm-test --embedding
--model qwen3-embedding:0.6b` passes.

**Result 2026-10-17** (mradermacher's i1-Q4_1 file, served beside the reranker). The served matrix
is [[0.7766, 0.1675], [0.1430, 0.6158]] against fp32's [[0.7646, 0.1414], [0.1355, 0.6000]]: max
difference 0.026, so the 0.02 bar is not met. The miss is the 4-bit weights, not the server: the same
GGUF run on the CPU is itself 0.026 from fp32, and the served matrix is within 0.0013 of it. Every
other check held: 1024 dimensions, unit length, `dimensions: 256` is the renormalized prefix,
`dimensions: 8`, a clustering task and the reranker's tag are refused with 400. `oflm-test
--embedding --model qwen3-embedding:0.6b`: 10 PASS, E8 SKIP (its bundled vectors are
embeddinggemma's).

### SERVER-RERANK: `/v1/rerank` scores and orders documents against a query
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/server/server.cpp`, `src/server/openai_compat.hpp`, `src/include/AutoEmbeddingModel/`)
**Verification:** test (the body, prompt, score and order: `src/server/openai_compat_test.cpp`
`test_rerank`) and manual (the scores)

`oflm serve --rerankmodel qwen3-reranker:0.6b` loads a reranker beside any chat and embedding
model and serves `POST /v1/rerank` in the Cohere / Jina / llama.cpp shape:
`{model?, query, documents: [string | {text}], top_n?, return_documents?, instruction?}` ->
`{object: "list", model, results: [{index, relevance_score, document?}], usage}`, results sorted by
score, best first, ties in request order. Each document is scored on the card's prompt (system
"Judge whether the Document meets the requirements ...", a user turn of `<Instruct>`, `<Query>`
and `<Document>` lines, then the assistant turn with an empty think block), tokenized in the
card's three pieces, as the softmax over the (no, yes) logits at the last position, i.e.
`sigmoid(yes - no)`. The two logits are the hidden state against the "yes" and "no" rows of the
tied head (`token_embd`, read exactly from the GGUF on the host), so no lm_head runs. The
endpoint holds the NPU lock like `/v1/embeddings`.

**Acceptance criteria:**
- The prompt for (the card's instruction, "What is the capital of China?", "The capital of China
  is Beijing.") is byte-equal to the card's template; an empty instruction is the card's default.
- Logits (yes, no) = (2.0, 0.0) score 0.8808 within 1e-4; equal logits score 0.5.
- Results are sorted by score descending, a tie keeps request order, `top_n` keeps the best n and
  a `top_n` past the count keeps all.
- A body with no `query`, a non-string `query`, no `documents`, `documents` that is not an array,
  a document that is neither a string nor an object with a string `text` (`documents[i]`), a
  `top_n` that is not a positive integer, a non-boolean `return_documents`, and an empty or
  non-string `model` each return 400 with `error.param` naming the field (SERVER-REQUEST-VALIDATION).
  An empty `documents` array is a well-formed request.
- A server with no reranker loaded, or a request naming another model, returns 400
  `model_not_found` naming what is loaded (SERVER-MODEL-IDENTITY).

**Verification (manual):** the card's example -- "What is the capital of China?" against its two
documents -- ranks the Beijing document first, and every score is within 0.02 of the CPU
reference's (transformers, fp32, the official safetensors); `oflm-test --rerank --model
qwen3-reranker:0.6b` passes (its R1-R8 checks are unit-tested against a fake server in
`utilities/oflm-test/tests/test_rerank_checks.py`).

**Result 2026-10-17: PASS** (mradermacher's i1-Q4_1 file). The Beijing document ranks first at
0.99931 (fp32 0.99950), the gravity document scores 0.0000036 (fp32 0.0000050), and "Explain
gravity" with `top_n: 1` keeps the gravity document at 0.99850 (fp32 0.99936). A body with no
`query`, a numeric document and another model's tag are each refused with 400 naming the field or
the loaded model. `oflm-test --rerank`: 8 PASS; repeat draws and a document scored alone against
beside another were bit-identical.
