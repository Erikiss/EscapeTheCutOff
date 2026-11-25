CUTOFF-MIX: API MECHANICS FOR RELAYING ONE REASONING CHAIN ACROSS claude-fable-5.2 / gpt-6.2-sol / gemini-4-neon (checked 2026-11-24)

Main conclusion: none of the three vendors lets one model continue another model's native thinking. Raw chain of thought is never returned by any of the three: Claude returns summaries or empty thinking blocks, OpenAI returns summaries plus encrypted reasoning, Gemini returns thought summaries plus encrypted signatures. Native reasoning is also bound to its own vendor or model family. So cross-model mixing can only happen at the visible-text level: the shared scratchpad goes into the user message, and each model's visible output is the next segment. Reasoning is mandatory on all three (`reasoning.mandatory: true` in /api/v1/models). The lowest effort is `low`; `none` and `minimal` are rejected or mapped up. Every segment therefore pays for hidden thinking tokens, and those count against `max_tokens`.

Critical risk for Claude: Anthropic's `reasoning_extraction` refusal category on Fable 5.2 explicitly covers "A <thinking>, <reasoning>, or scratchpad section that the model fills in", "a running log of its reasoning", "Reasoning asked for verbatim or in full", and also "earlier assistant turns and examples in the prompt that show reasoning in tags". The docs say: "change the prompt rather than retry" (https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback#keep-reasoning-in-thinking-blocks).
- A refusal of this kind that arrives before any output is billed.
- Via OpenRouter Chat Completions it appears as `finish_reason: "content_filter"` with `message.refusal`.
- Asking Fable 5.2 to "continue the scratchpad token-for-token" falls inside that description.
- The documented allowed routes are:
  - summarized thinking (`display: "summarized"`, which is OpenRouter's default for Claude);
  - asking for "a short explanation, the evidence behind a result".
- Recommendation: in Claude's segments, ask for "the next intermediate result + evidence" (template B below). Optionally also append its summarized-thinking text to the notes. Measure the refusal rate early. Do not iterate on rewordings to get past the classifier; Anthropic's guidance is to stop asking for reasoning in text.
- GPT and Gemini have no equivalent documented category. Template A (literal continuation) is fine for them. Whether GPT-6.2 Sol has a similar policy is UNVERIFIED.

(0) Live model facts from GET https://openrouter.ai/api/v1/models and /models/{slug}/endpoints (no key needed)
- anthropic/claude-fable-5.2:
  - Context 1,000,000; max output 128,000.
  - $8 in / $40 out per 1M tokens; cache read $0.20; cache write $10 (5 min), $16 (1 h).
  - Supported params: include_reasoning, max_tokens, max_completion_tokens, reasoning, reasoning_effort, response_format, stop, structured_outputs, tools, tool_choice, verbosity. No temperature, top_p or seed.
  - Efforts: max, xhigh, high, medium, low. Default high.
  - Providers: anthropic, anthropic/fast (priority tier, 2x price), azure, google-vertex/global. All support `stop`.
- openai/gpt-6.2-sol:
  - Context 1,050,000; max prompt 922,000; max output 128,000.
  - $2 in / $10 out; cache read $0.10; cache write $2.50. Above 272k prompt tokens: $4 in / $15 out, read $0.20, write $5.
  - Supported params: include_reasoning, max_tokens, max_completion_tokens, reasoning, reasoning_effort, response_format, seed, structured_outputs, tools, tool_choice, verbosity. No stop, temperature or top_p.
  - Efforts: max, xhigh, high, medium, low. Default medium. OpenAI's model page says the API rejects effort `none`.
  - Providers: openai, openai/fast, openai/ultrafast, azure, amazon-bedrock/global, google-vertex/global.
- google/gemini-4-neon:
  - Context 1,048,576.
  - Max output: 65,536 on the google-ai-studio standard endpoint; 943,718 on vertex, AI Studio flex and AI Studio priority.
  - $3 in / $15 out; cache read $0.15. OpenRouter also lists a "cache write" of $0.0833/M, which is unexplained (UNVERIFIED).
  - Supported params: include_reasoning, max_tokens, reasoning, reasoning_effort, response_format, seed, stop, structured_outputs, temperature, top_p, tools, tool_choice.
  - `stop` is offered only on the google-vertex endpoints, not on google-ai-studio.
  - Efforts: xhigh, high, medium, low. OpenRouter default is medium; Google's native default is high. Always set effort explicitly.
  - Providers: google-ai-studio (standard, flex, priority), google-vertex/global (standard, priority).
- Ignore `supports_implicit_caching`. It is false on every endpoint, including OpenAI, which caches automatically. Check `usage.prompt_tokens_details.cached_tokens` instead.
- `latency_last_30m` and `throughput_last_30m` are null on the public endpoint.
- The models API has a `knowledge_cutoff` field, but it is null for all three. Vendor-stated cutoffs:
  - Fable 5.2: June 2026 (Anthropic model docs).
  - GPT-6.2 Sol: Jun 01, 2026 (https://developers.openai.com/api/docs/models/gpt-6.2-sol).
  - Gemini 4 Neon: not stated.

(a) SSE format: POST /api/v1/chat/completions with `"stream": true`
- Framing: `data: {json}` lines, a blank line between events, terminated by `data: [DONE]`. Comment lines `: OPENROUTER PROCESSING` must be skipped before JSON parsing.
- `X-Generation-Id` response header on all endpoints.
- Chunk fields: `id` ("gen-…"), `object` ("chat.completion.chunk"), `created`, `model`, `provider`, `service_tier`, `choices[0]`, plus optional `usage`, `error` and `openrouter_metadata`.
  - `choices[0].delta` holds `role`, `content` (visible text), `reasoning` (reasoning text or summary string), `reasoning_details[]`, `refusal`, `tool_calls`.
  - `choices[0]` also holds `finish_reason` and `native_finish_reason`.
- `reasoning_details[]` item fields:
  - `type` is one of `reasoning.text` {text, signature}, `reasoning.summary` {summary}, `reasoning.encrypted` {data}.
  - Also `id`, `format` ("anthropic-claude-v1", "openai-responses-v1", "azure-openai-responses-v1", "google-gemini-v1", …), `index`.
  - Concatenate chunks in order. Encrypted content may stream as "[REDACTED]".
- `finish_reason` is normalized to stop | length | tool_calls | content_filter | error. The raw provider value is in `native_finish_reason`; for example, Gemini returns "MAX_TOKENS" or "CONTINUATION".
- Usage chunk: exactly once, just before [DONE]. Unlike OpenAI's spec, it has one choice with an empty delta that repeats `finish_reason`. Treat it as accounting, not a second terminal event. `stream_options.include_usage` is deprecated and has no effect; usage is always sent.
  - Fields: `prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_tokens_details.{cached_tokens, cache_write_tokens}`, `completion_tokens_details.reasoning_tokens`, `cost`, `cost_details.*`.
  - Visible tokens = completion_tokens − reasoning_tokens.
- Errors:
  - Before the stream starts: a normal HTTP error with JSON `{"error":{"code","message","metadata"}}`.
  - After the 200 is committed: a chunk with a top-level `error` (`metadata.error_type`, `provider_code`) and `choices[0].finish_reason: "error"`. The HTTP status stays 200.
  - An error chunk can be the first and only event.
  - Fallback to another provider stops once any output has reached the client.
- Debugging: `"debug": {"echo_upstream_body": true}` (streaming only) returns the transformed upstream body in the first chunk. Use it to confirm block boundaries, cache markers and stop handling per provider.
- Sources: https://openrouter.ai/docs/api_reference/streaming.md, https://openrouter.ai/docs/api_reference/overview.md, https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion.md, https://openrouter.ai/docs/guides/best-practices/reasoning-tokens.md

(b) Thinking visibility and injecting another model's thinking
- claude-fable-5.2:
  - Raw chain of thought is never returned. Thinking blocks are "summarized" (OpenRouter's default for Claude) or "omitted" (empty string; Anthropic's native default).
  - Via OpenRouter you get `delta.reasoning` plus `reasoning_details` of type `reasoning.text` with a signature, format anthropic-claude-v1.
  - Blocks are bound to the model, to the exact preceding prefix (Fable 5.2 enforces this on every account; an edited history gives a 400 "block is bound to a different conversation") and to the account.
  - Fable 5.2 reads only Anthropic-model thinking blocks. Blocks it cannot read are silently dropped and not billed.
  - Foreign or plaintext text cannot be passed off as a Claude thinking block (signed). Not injectable.
- gpt-6.2-sol:
  - Raw reasoning is never exposed: "we don't expose the raw reasoning tokens… you can view a summary". Request it with `reasoning.summary` (auto | concise | detailed). Some organizations need verification for summaries; via OpenRouter it is OpenRouter's organization.
  - Encrypted reasoning: "Persisted reasoning can be reused only within the same model family"; incompatible reasoning is omitted on a family switch. Not injectable.
  - Exact OpenRouter chunk shape for GPT-6.2 summaries (reasoning.summary vs reasoning.text) is UNVERIFIED.
  - Source: https://developers.openai.com/api/docs/guides/reasoning
- gemini-4-neon:
  - Thought summaries only ("By default, only the final output is returned"; summaries are opt-in).
  - Thought signatures are "an encrypted representation of the model's internal reasoning state", required for continuity, and must be resent unmodified. On model switches Google says to "resend the previous model's thought blocks. The backend manages compatibility" (Gemini models only). Not injectable cross-vendor.
  - OpenRouter shape: summaries plus `reasoning.encrypted`, format google-gemini-v1. The summary item type is UNVERIFIED.
  - Gemini 4 "long decode": a length stop returns a `reasoning.encrypted` item with id "gemini-continuation", which is provider-bound. Do not send it back as a trailing assistant turn unless you mean to resume.
  - Sources: https://aistudio.google.com/docs/thinking, https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/gemini-4.md
- Same-model continuity (A→B→A) through native reasoning would require a strictly append-only multi-turn transcript with A's assistant turns and `reasoning_details` replayed verbatim. That is possible but complex. Not recommended for v1.

(c) Assistant prefill / continuation via OpenRouter
- claude-fable-5.2: rejected with a 400 ("This model does not support assistant message prefill. The conversation must end with a user message."). That error text is quoted in OpenRouter's Claude 5.5 guide. Anthropic lists prefill as a 400 on the Fable 5/5.1/5.2, Opus 5/5.5 and 4.6+ family.
- gpt-6.2-sol: OpenRouter's overview advertises generic prefill (its example uses openai/gpt-5.2). OpenAI has no native continuation, so the model most likely treats the message as a finished prior turn and starts a new message. UNVERIFIED; check with debug.echo_upstream_body.
- gemini-4-neon: behaviour of a text-only trailing assistant turn is UNVERIFIED. A trailing assistant turn that carries a continuation `reasoning_details` item triggers a long-decode resume, and OpenRouter drops that trailing turn before forwarding.
- OpenRouter's own replacement for removed prefill: "Continuations: move the partial text into the user message and ask the model to continue from it." Use that for all three.
- `/api/v1/completions` (raw `prompt`) is listed for all three models. Whether it gives true raw continuation on these chat-only upstreams is UNVERIFIED and unlikely.

(d) Recommended black-box relay technique
1. Fixed per-model request shape, append-only:
   - `messages[0]` = static system message (instructions, today's date, K), byte-identical for the whole run.
   - `messages[1]` = one user message whose content array is: [header block with PROBLEM + "WORKING NOTES…"], [one text block per segment, in order, never edited or merged], [constant tail block "[END OF NOTES - your output is appended directly after the last character above]"].
   - Put the cache marker on the last segment block, never on the tail. With that placement, everything up to the previous segment is a byte-identical prefix in the next call.
   - Never add trailing system messages. With Claude via Chat Completions, OpenRouter sends ordinary system text as the top-level `system` prompt. A changing system prompt would invalidate the whole prefix.
2. Native reasoning at `effort: "low"` with `reasoning.exclude: true`: reasoning is hidden from the stream but still billed. The visible output is the segment.
3. Segment length cannot be set exactly with `max_tokens`, because thinking shares the budget.
   - Instruct "one step of about K tokens, then output <<END_STEP>>".
   - Stop on the sentinel: `stop` for Claude on all providers and for Gemini on Vertex. For GPT and AI Studio, detect it client-side and abort.
   - Hard-cut client-side at about 4×K characters at the last whitespace.
   - Set `max_tokens` = K + thinking allowance (start at 2048 and tune from `usage.completion_tokens_details.reasoning_tokens`).
   - The sentinel is non-whitespace on purpose: whitespace-only stop sequences may be rejected by Anthropic (UNVERIFIED).
4. Post-process each segment before appending:
   - Strip "Continuation:" preambles and code fences.
   - Remove echoed overlap (longest suffix of the notes that equals a prefix of the new text, at least 12 characters).
   - Keep a mid-word tail if the model completed it.
5. Template A (GPT, Gemini), system prompt:
   "You are one of several writers taking turns on a single shared working-notes document that works through the problem in the user message. The notes may stop mid-sentence or mid-word. Write the text that comes next so the notes read as one continuous document.
   Rules:
   1. Output only the new text. No preamble, no quotation marks, no code fences, no restating or summarising earlier notes.
   2. Start exactly where the notes end; if they end mid-word or mid-sentence, finish it first.
   3. Keep the same language, notation, voice and level of detail.
   4. Write one step of about {K} tokens, end at a sentence boundary, then output <<END_STEP>> and nothing else.
   5. If the notes have already reached a conclusion, write 'FINAL ANSWER:' and the answer.
   Today's date: 2026-11-24."
6. Template B (Claude, to stay within the documented allowed route), system prompt:
   "Several analysts are jointly working through the problem in the user message; their shared notes so far are included. Contribute the next intermediate result: state it and the evidence or calculation that supports it, in about {K} tokens, written so it can be appended to the notes as the next paragraph. Output only that paragraph, then <<END_STEP>>. If the notes already reach a conclusion, write 'FINAL ANSWER:' and the answer. Today's date: 2026-11-24."
   - Optionally drop `exclude` and also capture Claude's `delta.reasoning` (summarized thinking) as extra material.
   - Do not wrap the notes in <thinking> or <reasoning> tags; Anthropic lists such tags as a trigger.
7. A reference implementation is in the scratchpad, with offline tests passing (4/4): SSE parser, chunk accumulator, splice/overlap trimming, request builder, and an httpx streaming function with sentinel/K abort and TTFT/first-content timing.

(e) Stopping early, billing, max_tokens, stop
- Abort = close the HTTP connection. "For supported providers, this immediately stops model processing and billing."
  - Supported: OpenAI, Azure, Anthropic.
  - Not supported: AWS Bedrock, Google, Google AI Studio. For those "the model will continue processing and you will be billed for the complete response" (also true for any non-streaming request).
  - So abort saves money on Claude (anthropic/azure) and GPT (openai/azure), and never on Gemini 4 Neon. Every Gemini endpoint is Google or AI Studio, so keep Gemini's `max_tokens` tight and rely on stop (Vertex only).
  - Exactly what is billed on abort (input plus tokens generated so far, including hidden thinking) is UNVERIFIED in detail.
- An aborted stream has no usage chunk. Fetch tokens and cost afterwards with GET /api/v1/generation?id=<X-Generation-Id> (needs the key).
- max_tokens:
  - Covers reasoning plus visible output on all three. If thinking eats it all: `finish_reason: "length"`, empty content, still billed.
  - Some providers enforce a minimum of 16.
  - Gemini 4: keep at or below 65,536 (the per-request limit), otherwise long-decode semantics apply; `max_tokens` becomes a cumulative budget and continuation tokens are returned.
  - OpenRouter's in-flight spending budget estimates cost from `max_tokens`. Oversized values cause 402 `in_flight_budget_exhausted` under concurrency.
- stop: string or array, at most 4 items.
  - Claude: all providers.
  - Gemini: google-vertex only. Use `provider.order: ["google-vertex", …]`. `require_parameters: true` forces this, but gives a 503 if Vertex is unavailable.
  - GPT-6.2-sol: none. Unsupported params are silently ignored unless `provider.require_parameters: true`.
- Do not send `verbosity` to Claude: OpenRouter maps it to `output_config.effort` there. For GPT, `verbosity: "low"` is fine.
- `temperature` and `top_p` are not supported on Fable 5.2 or GPT-6.2 Sol (ignored by OpenRouter); Gemini accepts them. `seed` works on GPT and Gemini only.

(f) Prompt caching. Critical: every switch re-sends the prefix, and caches are per model.
- Claude Fable 5.2:
  - Explicit `cache_control: {type: "ephemeral"[, ttl: "1h"]}` on content blocks, at most 4 breakpoints. Top-level automatic `cache_control` is supported on Anthropic, Vertex, Azure and Bedrock, but it would key on the moving tail. Use an explicit marker on the last segment block instead.
  - Minimum cacheable prefix: 512 tokens.
  - Write 1.25x ($10/M) for the 5-minute TTL, 2x ($16/M) for 1 hour; read $0.20/M (0.025x).
  - Each breakpoint looks back at most 20 block positions for a previous entry. If Claude sits out more than about 15–20 appended segments, add an intermediate marker. Never merge or edit old blocks.
  - Changing thinking or effort invalidates the messages cache. Pin effort per model; if needed use OpenRouter's content-less system message with `configuration_update`, which Fable 5.2 supports.
  - The TTL counts from the start of a request. For idle gaps of 5–60 minutes, Anthropic recommends a keep-alive over the 1-hour TTL (a `max_tokens: 0` re-send, non-streaming). Whether OpenRouter allows `max_tokens: 0` is UNVERIFIED.
- GPT-6.2 Sol:
  - Automatic caching for prompts of 1024 tokens or more. On GPT-5.6 and later, writes are billed automatically at 1.25x ($2.50/M); reads $0.10/M (0.05x).
  - Explicit `prompt_cache_breakpoint` on text blocks plus root `prompt_cache_options: {mode: "explicit", ttl: "30m"|"12h"}`; a 12-hour write costs 2.5x. Anthropic-style `cache_control` block markers are converted to `prompt_cache_breakpoint` automatically.
  - Use `prompt_cache_key` per run.
  - Old guidance says caching works in 128-token increments; UNVERIFIED for GPT-6.2.
- Gemini 4 Neon:
  - Implicit caching only. Google: "explicit caching is not supported … for Gemini 4 models". OpenRouter's `cache_control` for Gemini 4 is expected to be a no-op (UNVERIFIED).
  - Minimum 4,096 tokens. Cached input $0.15/M (0.05x).
  - Gemini 4 rule: "The full previous request is cached only if the new request adds at least 2,050 tokens… For smaller additions, the last 2,050 tokens of the previous request are reprocessed at the standard input rate." With small segments, each Gemini call pays full price for about 2,050 tokens plus the delta.
  - Implicit TTL is best effort (OpenRouter cites 3–5 minutes on average for 2.5-era models; Gemini 4 UNVERIFIED).
- Routing: `session_id` (body, or `x-session-id`, at most 256 characters) makes the provider sticky, but sticky routing is disabled when `provider.order` is set. Pinning an order is effectively sticky anyway.
- Per-switch input cost for model X: P_cached×read + D×(write or input) + tail, where D = tokens appended since X's last call.
  - Example at P = 20k, D = 300, computed from list prices: Fable ≈ $0.004 + $0.003 = $0.007; GPT ≈ $0.002 + $0.00075 = $0.0028; Gemini ≈ 17.65k×$0.15/M + 2.35k×$3/M ≈ $0.0097.
  - Uncached Fable would be $0.16 per call.
  - Hidden thinking output (for example 500 tokens × $40/M = $0.02 on Fable) likely dominates per-switch cost; thinking token counts at low effort are UNVERIFIED.
- Sources: https://openrouter.ai/docs/guides/best-practices/prompt-caching.md, https://aistudio.google.com/docs/caching, https://aistudio.google.com/docs/models/gemini-4-neon

(g) Latency
- No per-model TTFT is documented by OpenRouter or the vendors.
- OpenRouter: uncached 20k–90k-token prompts "typically add 5–10 seconds of prefill delay".
- Time to first visible token = TTFT + thinking time. That is the main cost of high-frequency switching.
- Artificial Analysis (fetched 2026-11-24, max/xhigh effort, 5-day average, TTFT not shown):

| Model (effort, provider) | Output speed | Time to first answer token |
| --- | --- | --- |
| Claude Fable 5.2 (Max, Anthropic) | 111 tok/s | 200.31 s |
| Gemini 4 Neon (xhigh, Google) | 88 tok/s | 58.85 s |
| GPT-6.2 Sol (Max, OpenAI) | 66 tok/s | 88.22 s |

- Low-effort figures are UNVERIFIED; measure them. The reference client records t_first_any and t_first_content.
- Speed levers (all at higher price):
  - Fable fast mode via `service_tier: "fast"` → anthropic/fast endpoint, $16/$80, up to 2.5x output speed per Anthropic.
  - GPT `fast` (2x price) or `ultrafast` (3x, rising to 6x from 2027-03-01).
  - Gemini `priority` ($5.40/$27).
  - OpenAI suggests asking for a short preamble to get faster first visible tokens.

(h) Limits and errors
- Status codes: 400 (invalid_request, context_length_exceeded, invalid_prompt, e.g. prefill), 401, 402 (credits, or in-flight budget with `Retry-After`; branch on `error.metadata.limit_source`), 403 (permission_denied, budget/guardrail, content_policy_violation, refusal-as-error; Fable and GPT `top_provider.is_moderated = true`), 408, 429 rate_limit_exceeded (honor `Retry-After`; `provider_code` is in metadata), 502 provider_unavailable, 503 provider_overloaded or no provider matches routing, 504 timeout, 500 server.
- Switch on `error.metadata.error_type`, not on the HTTP status.
- Successful responses carry no X-RateLimit-* headers. Those headers appear only on OpenRouter's own free-model or key-management 429s. Rate limits are governed globally, so extra keys don't help.
- A model's own refusal (for example Claude `stop_reason: "refusal"`) is not an error: `finish_reason: "content_filter"` + `message.refusal`.
- Thinking exhausted the budget: `finish_reason: "length"`, empty content.
- OpenAI direct limits for reference (Standard tier, RPM / TPM): Build 5,000 / 1M; Launch 10,000 / 4M; Grow 15,000 / 40M.
- Sources: https://openrouter.ai/docs/api_reference/errors-and-debugging.md, https://openrouter.ai/docs/api_reference/limits.md

REQUEST BODIES: Chat Completions, POST https://openrouter.ai/api/v1/chat/completions
SYS_A and SYS_B are the templates in (d); K = 128 here.

1) Plain continuation segment
Claude:
{"model":"anthropic/claude-fable-5.2","stream":true,"max_tokens":2176,"reasoning":{"effort":"low","exclude":true},"stop":["<<END_STEP>>"],"provider":{"order":["anthropic","azure"],"allow_fallbacks":true},"session_id":"run42-claude","messages":[{"role":"system","content":[{"type":"text","text":"SYS_B(K=128)","cache_control":{"type":"ephemeral"}}]},{"role":"user","content":[{"type":"text","text":"PROBLEM:\n<q>\n\nWORKING NOTES (append directly after the last character):\n"},{"type":"text","text":"<seg1>"},{"type":"text","text":"<segN>","cache_control":{"type":"ephemeral"}},{"type":"text","text":"\n[END OF NOTES - your output is appended directly after the last character above]"}]}]}

GPT (no stop or temperature; abort client-side on the sentinel or about 4K characters):
{"model":"openai/gpt-6.2-sol","stream":true,"max_tokens":2176,"reasoning":{"effort":"low","exclude":true},"verbosity":"low","prompt_cache_key":"run42-gpt","session_id":"run42-gpt","provider":{"order":["openai","azure"],"allow_fallbacks":true},"messages":[{"role":"system","content":[{"type":"text","text":"SYS_A(K=128)"}]},{"role":"user","content":[{"type":"text","text":"PROBLEM:…"},{"type":"text","text":"<seg1>"},{"type":"text","text":"<segN>"},{"type":"text","text":"\n[END OF NOTES - …]"}]}]}

Gemini (not cancellable, so keep max_tokens tight; stop works on Vertex only):
{"model":"google/gemini-4-neon","stream":true,"max_tokens":2176,"reasoning":{"effort":"low","exclude":true},"stop":["<<END_STEP>>"],"provider":{"order":["google-vertex","google-ai-studio"],"allow_fallbacks":true},"session_id":"run42-gemini","messages":[{"role":"system","content":"SYS_A(K=128)"},{"role":"user","content":[{"type":"text","text":"PROBLEM:…"},{"type":"text","text":"<seg1>"},{"type":"text","text":"<segN>"},{"type":"text","text":"\n[END OF NOTES - …]"}]}]}

2) Native-reasoning call that returns reasoning text (summaries; raw chain of thought is unavailable on all three)
Claude:
{"model":"anthropic/claude-fable-5.2","stream":true,"max_tokens":32000,"reasoning":{"effort":"medium"},"provider":{"order":["anthropic","azure"]},"messages":[{"role":"system","content":"…"},{"role":"user","content":"…"}]}
→ summarized thinking arrives in delta.reasoning / delta.reasoning_details[type=reasoning.text, signature, format=anthropic-claude-v1], then delta.content.

GPT:
{"model":"openai/gpt-6.2-sol","stream":true,"max_tokens":32000,"reasoning":{"effort":"medium","summary":"detailed"},"provider":{"order":["openai","azure"]},"messages":[…]}
→ summary plus encrypted items, format openai-responses-v1 (exact item types UNVERIFIED).

Gemini:
{"model":"google/gemini-4-neon","stream":true,"max_tokens":32000,"reasoning":{"effort":"medium"},"provider":{"order":["google-vertex","google-ai-studio"]},"messages":[…]}
→ thought summaries plus encrypted signatures, format google-gemini-v1 (exact item types UNVERIFIED).

Untested at runtime: no API key was available, so every request body above and every UNVERIFIED item still needs one live smoke test. Use `debug.echo_upstream_body` and log `usage` (cached_tokens, cache_write_tokens, reasoning_tokens).

Other notes:
- If the router model will be trained on model outputs, check OpenRouter's distillation policy and the `provider.enforce_distillable_text` flag.
- Anthropic also has a `frontier_llm` refusal category ("could assist the development of competing AI models").

Files are in (scratchpad)/api-mechanics/:
- relay_client.py
- test_relay_client.py
- models.json
- ep_anthropic_claude-fable-5.2-20261109.json
- ep_openai_gpt-6.2-sol-20261117.json
- ep_google_gemini-4-neon-20261118.json
- dl/ (downloaded OpenRouter doc pages; or-full.txt is the complete OpenRouter docs)

Sources:
- OpenRouter: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens.md, https://openrouter.ai/docs/api_reference/streaming.md, https://openrouter.ai/docs/guides/best-practices/prompt-caching.md, https://openrouter.ai/docs/guides/best-practices/latency-and-performance.md, https://openrouter.ai/docs/api_reference/parameters.md, https://openrouter.ai/docs/guides/routing/provider-selection.md, https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/fable-5-1.md, https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/claude-5-5.md, https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/gemini-4.md, https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/gpt-6.md
- Anthropic: https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback
- OpenAI: https://developers.openai.com/api/docs/guides/reasoning, https://developers.openai.com/api/docs/models/gpt-6.2-sol
- Google: https://aistudio.google.com/docs/thinking, https://aistudio.google.com/docs/caching, https://aistudio.google.com/docs/models/gemini-4-neon
- Artificial Analysis: https://artificialanalysis.ai/models/claude-fable-5-2, https://artificialanalysis.ai/models/gemini-4-neon, https://artificialanalysis.ai/models/gpt-6-2-sol