# PoC: pinned-CPU state image tier (research prototype, not for upstream)

Question: on hybrid models, does keeping per-conversation state images (Mamba/conv
state, sliding-window blocks) plus content-hashed KV blocks in pinned CPU memory
cut median TTFT on capy-flash's agent replay by >=2x over vLLM main with its CPU
KV offload (`--kv-offloading-size`)?

Code: `vllm/v1/core/state_image_tier.py` (scheduler: write-back on GPU eviction,
restore before the prefix-cache lookup, superseded-image drop) and
`vllm/v1/worker/state_image_tier_worker.py` (pinned slots, copies before the
forward pass). Enable with `VLLM_SIT_CPU_GB=<GiB>`; `VLLM_SIT_POLICY=all` keeps
superseded Mamba images; `VLLM_SIT_DEBUG=1` logs every restore.

Harness (this directory):

- `rr_replay.py`: capy-flash `bench/capy_flash/sessions.json` (3 sessions, 53 turns,
  8.6k-token system prompt growing to ~81k, ~3.8k new tokens per turn, greedy, 256
  output tokens) replayed by N agents (agent a -> session a % 3, tagged first user
  message so agents share only the system prompt). `--mode rr`: one request at a
  time, round robin, so each agent's cache must survive N-1 other turns.
  `--mode conc --think S`: N agents at once with exponential tool time.
- `exp.sh NAME TREE MODEL "SERVE_ARGS" "REPLAY_ARGS"`: one server + one replay
  (paths are the gpu-dev pod's `/home/dev/nr`; worktrees under `wt/`).
- `table.py`, `cmp.py`: TTFT by turn depth, CPU bytes, restore bandwidth; turn-by-turn
  output comparison (text, cached tokens, logprob deltas).

Example (H100, vLLM main 43108ffc):

```bash
M=nvidia/Nemotron-H-4B-Instruct-128K
scripts/exp.sh off64 main $M "--kv-cache-memory-bytes 4G --kv-offloading-size 64" "--agents 12 --mode rr"
VLLM_SIT_CPU_GB=64 scripts/exp.sh sit64 sit $M "--kv-cache-memory-bytes 4G" "--agents 12 --mode rr"
```
