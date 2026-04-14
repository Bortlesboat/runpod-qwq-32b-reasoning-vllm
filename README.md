# QwQ 32B Reasoning Worker For RunPod Hub

This repo packages an OpenAI-compatible `vLLM` worker around `Qwen/QwQ-32B` for H100-class RunPod Serverless deployments.

It is meant for longer-form reasoning use cases:

- high-effort assistant backends,
- research-style prompts where longer outputs are expected,
- teams that want a reasoning-tuned 32B worker instead of a general instruct default.

## Positioning

This listing is the reasoning lane in the portfolio:

- H100-focused hardware targeting,
- `QwQ-32B` defaults instead of generic instruct defaults,
- presets biased toward longer outputs and lower concurrency than the coding and chat listings.

The handler still forwards plain completions, chat completions, and explicit OpenAI-style routes through the local `vLLM` server inside the worker container.

## Presets

The Hub metadata ships with presets tuned for H100-class reasoning deployment:

- `Balanced QwQ 32B`
- `Long Reasoning QwQ 32B`
- `High Throughput QwQ 32B`

The default model is:

- `Qwen/QwQ-32B`

## Important note about automated tests

The Hub submission smoke test intentionally uses a much smaller Qwen-family model than the default deployment model. That keeps automated validation fast and cheap while still proving the worker boots, serves requests, and routes payloads correctly.

## Local verification

```bash
python -m unittest discover -s tests -v
python -m json.tool .runpod/hub.json
python -m json.tool .runpod/tests.json
python -m py_compile handler.py tests/test_handler.py
```

## Publish flow

1. Cut a GitHub release.
2. Submit the repo to RunPod Hub.
3. Let RunPod build and test the release.
4. Iterate by publishing new releases rather than rewriting old tags.
