set -euo pipefail
source uv_agentgym_rl/bin/activate

TASK=$1                     # alfworld, sciworld, textworld, webshop
MODEL=$2                    # gpt-4o-mini, gpt-4-turbo, gpt-4.1, gpt-5
API_KEY=$3
API_BASE_URL=$4
WORLD_MODEL=$5              # world model checkpoint
MAX_CONCURRENCY=${6:-150}   # max concurrency
MAX_ROUND=${7:-50}          # max round
NUM_EXAMPLES=${8:--1}       # num examples, -1 means all samples
SPLIT=${9:-test}            # train, test(valid_train), valid_seen, valid_unseen
OUTPUT_ROOT=${10:-outputs}

WM_NAME="vllm_world_model"
AGENT_INSTRUCT_FILE="data/init_contexts/${TASK}/agent_instruct_${SPLIT}.json"
WM_INSTRUCT_FILE="data/init_contexts/${TASK}/wm_instruct_${SPLIT}.json"
OUTPUT_ROOT="$OUTPUT_ROOT/interaction/world_model/${SPLIT}/${TASK}/${MODEL}/${WORLD_MODEL}/${MODEL}"

echo "TASK: $TASK"
echo "MODEL: $MODEL"
echo "World Model Checkpoint: $WORLD_MODEL"
echo "Max Concurrency: $MAX_CONCURRENCY"
echo "Output Root: $OUTPUT_ROOT"
echo "Number of Samples: $NUM_EXAMPLES"
echo "Agent Instruct File: $AGENT_INSTRUCT_FILE"
echo "WM Instruct File: $WM_INSTRUCT_FILE"


# PIDs initialized to avoid -u issues in cleanup
VLLM_PID=""
ENV_PID=""

# Safe killers compatible with set -euo pipefail
kill_safe() {
    local pid=${1:-}
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
    fi
}

kill_group() {
    local pid=${1:-}
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
        # Kill the whole process group if possible
        local pgid
        pgid=$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || echo "")
        if [ -n "${pgid}" ]; then
            kill -TERM -"${pgid}" 2>/dev/null || true
            sleep 2
            kill -KILL -"${pgid}" 2>/dev/null || true
        else
            # Fallback to killing just the pid
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 2
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    fi
}

cleanup() {
    echo "[cleanup] stopping env and vllm if alive..."
    kill_safe "${ENV_PID}"
    kill_group "${VLLM_PID}"
}



## ====== Start vLLM Server for World MODEL ======
source uv_agentgym_rl/bin/activate
# ===== 1. Pick a free port =====
wm_port=$((30000 + RANDOM % (99999-30000+1)))
# NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
# NUM_GPUS=$(nvidia-smi -L | wc -l)
# echo "Number of GPUs: $NUM_GPUS"

# ===== 2. Start vLLM in background (new process group) =====
setsid vllm serve $WORLD_MODEL \
    --port $wm_port \
    --served-model-name $WM_NAME \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    </dev/null >/tmp/vllm_${wm_port}.log 2>&1 &
VLLM_PID=$!
echo "Launching vLLM server... (pid=$VLLM_PID, port=$wm_port)"

# ===== 3. Auto cleanup after exit =====
# This will be called when the script exits or is interrupted
trap cleanup EXIT INT TERM

# ===== 4. Wait for vLLM to be ready =====
wait_for_server() {
    local port=$1
    local log_file=$2
    local max_wait_time=600  # 10 minutes (600 seconds)
    local wait_time=0
    tail -f $log_file &   # This will print logs in real-time
    tail_pid=$!           # Save the process ID of the tail command
    while ! curl -sf "http://localhost:${port}/health" >/dev/null; do
        if [[ $wait_time -ge $max_wait_time ]]; then
            kill $tail_pid  # Stop tail command
            exit 1
        fi
        sleep 2
        ((wait_time+=2))
    done
    # Stop tail when server is ready
    kill $tail_pid
    echo "Server on port $port is ready!"
}
sleep 5
wait_for_server $wm_port /tmp/vllm_${wm_port}.log
echo "MODEL: $WORLD_MODEL | port: $wm_port | PID: $VLLM_PID | Log: /tmp/vllm_${wm_port}.log"
## ==== End of vLLM Server Setup =====


## Interact with World MODEL
for i in {1..10}; do
    echo "Output root: $OUTPUT_ROOT"
    python scripts/interact_with_world_model/run.py \
        --task $TASK \
        --model $MODEL \
        --wm-port $wm_port \
        --wm-name $WM_NAME \
        --max-concurrency $MAX_CONCURRENCY \
        --max-steps $MAX_ROUND \
        --agent-instruct-file $AGENT_INSTRUCT_FILE \
        --wm-instruct-file $WM_INSTRUCT_FILE \
        --output-root $OUTPUT_ROOT \
        --n_samples $NUM_EXAMPLES
    sleep 5
done
# Gracefully stop vLLM (whole group), then force if needed.
kill_group "$VLLM_PID"
echo "vLLM server (pgid of $VLLM_PID) stopped."


# TASK=alfworld
# MODEL=gpt-4o
# WORLD_MODEL=X1AOX1A/WorldModel-Alfworld-Qwen2.5-7B
# MAX_CONCURRENCY=150
# MAX_ROUND=50
# NUM_EXAMPLES=-1
# SPLIT=test
# OUTPUT_ROOT=outputs
# ts -G 1 bash scripts/interact_with_world_model/run.sh $TASK $MODEL $WORLD_MODEL $MAX_CONCURRENCY $MAX_ROUND $NUM_EXAMPLES $SPLIT $OUTPUT_ROOT