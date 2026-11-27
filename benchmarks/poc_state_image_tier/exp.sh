#!/bin/bash
# usage: exp.sh NAME TREE MODEL "SERVE_ARGS" "REPLAY_ARGS"
# starts vllm serve from worktree TREE, runs rr_replay.py, saves metrics, stops the server.
N=$1; T=$2; M=$3; SA=$4; RA=$5
R=/home/dev/nr
export HF_HOME=$R/hf PYTHONPATH=$R/wt/$T VLLM_CACHE_ROOT=$R/cache/$T TRITON_CACHE_DIR=$R/cache/$T/triton
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export VLLM_SIT_DEBUG=${VLLM_SIT_DEBUG:-1}
mkdir -p $R/results/$N
cd $R
pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null; sleep 3
echo "$(date -u +%T) START $N tree=$T $(git -C wt/$T rev-parse --short HEAD) model=$M serve=[$SA] replay=[$RA]" | tee -a logs/exp.log
venvs/v/bin/python -m vllm.entrypoints.cli.main serve $M --port 8000 --served-model-name m \
  --max-model-len 98304 --enable-prefix-caching --enable-prompt-tokens-details $SA > results/$N/serve.log 2>&1 &
SP=$!
for i in $(seq 1 600); do
  curl -sf localhost:8000/health > /dev/null && break
  kill -0 $SP 2>/dev/null || { echo "$(date -u +%T) SERVER DIED $N" | tee -a logs/exp.log; tail -30 results/$N/serve.log; exit 1; }
  sleep 2
done
curl -s localhost:8000/metrics > results/$N/metrics_before.txt
nvidia-smi --query-gpu=memory.used --format=csv > results/$N/gpumem.txt
venvs/v/bin/python scripts/rr_replay.py http://localhost:8000 m capy-flash/bench/capy_flash/sessions.json results/$N/out.jsonl $RA > results/$N/replay.log 2>&1
curl -s localhost:8000/metrics > results/$N/metrics_after.txt
ps -o rss= -p $(pgrep -f "vllm.entrypoints.cli.main serve" | head -1) > results/$N/rss.txt 2>/dev/null
grep -h "EngineCore" results/$N/serve.log | grep -i -E "KV cache|offload|cpu" | head -20 > results/$N/kvinfo.txt
kill $SP; sleep 5; pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null; sleep 3
echo "$(date -u +%T) END $N $(tail -1 results/$N/replay.log)" | tee -a logs/exp.log
