# QwQ 32B Reasoning vLLM Worker

Deploy `Qwen/QwQ-32B` on RunPod Hub with an OpenAI-compatible `vLLM` worker tuned for longer reasoning-style outputs on H100-class hardware.

## Best for

- higher-effort assistants and research-style prompts,
- users who expect longer outputs and lower concurrency than a general chat worker,
- users searching for `QwQ`, `reasoning model`, `32B`, or `H100` on RunPod Hub.

## Request shapes

- `prompt` for `/v1/completions`
- `messages` for `/v1/chat/completions`
- `route` + `body` for explicit OpenAI-compatible requests

## Main knobs

- `MODEL_NAME`
- `MAX_MODEL_LEN`
- `GPU_MEMORY_UTILIZATION`
- `MAX_NUM_SEQS`
- `DEFAULT_MAX_TOKENS`
- `MAX_CONCURRENCY`

The smoke test uses a smaller Qwen-family model on easier-to-find GPU capacity, but the deployment presets stay centered on H100-class reasoning use.
