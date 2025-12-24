#!/usr/bin/env bash
set -euo pipefail

TASK=$1                     # alfworld, sciworld, textworld, webshop
RUN=${2:-0}                 # run id for multiple runs, just for separating output dirs
MODEL=$3                    # model dir
MAX_CONCURRENCY=${4:-150}   # max concurrency
MAX_ROUND=${5:-20}          # max round, reduce to 20 to prevent exceed context length
NUM_EXAMPLES=${6:--1}       # num examples
SPLIT=${7:-test}            # train, test(valid_train), valid_seen, valid_unseen
OUTPUT_ROOT=${8:-outputs}

OUTPUT_DIR=$OUTPUT_ROOT/interaction/real_env/$SPLIT/vllm/${TASK}/$MODEL/${TASK}_maxround${MAX_ROUND}_run${RUN}

# if global_step_ in MODEL, then NEED_MERGE=1
if [[ "$MODEL" == *"global_step_"* ]]; then
    NEED_MERGE=1
else
    NEED_MERGE=0
fi


if [ "$SPLIT" = "test" ]; then
    INFERENCE_FILE="data/eval/${TASK}_test.json"
elif [ "$SPLIT" = "train" ]; then
    INFERENCE_FILE="data/train/${TASK}_train.json"
elif [ "$SPLIT" = "valid_seen" ]; then
    INFERENCE_FILE="data/eval/${TASK}_valid_seen.json"
elif [ "$SPLIT" = "valid_unseen" ]; then
    INFERENCE_FILE="data/eval/${TASK}_valid_unseen.json"
fi

## ===== Start Environment Server =====
ENV_PORT=$((30000 + RANDOM % (99999-30000+1)))
if [ "$TASK" = "alfworld" ]; then
    source uv_alfworld/bin/activate
    alfworld --host 0.0.0.0 --port $ENV_PORT >/tmp/alfworld_server_${ENV_PORT}.log 2>&1 &
    SERVER_PID=$!
    trap "kill $SERVER_PID" EXIT INT TERM
elif [ "$TASK" = "sciworld" ]; then
    source uv_sciworld/bin/activate
    sciworld --host 0.0.0.0 --port $ENV_PORT >/tmp/sciworld_server_${ENV_PORT}.log 2>&1 &
    SERVER_PID=$!
    trap "kill $SERVER_PID" EXIT INT TERM
elif [ "$TASK" = "textworld" ]; then
    source uv_textworld/bin/activate
    textworld --host 0.0.0.0 --port $ENV_PORT >/tmp/textworld_server_${ENV_PORT}.log 2>&1 &
    SERVER_PID=$!
    trap "kill $SERVER_PID" EXIT INT TERM
elif [ "$TASK" = "webshop" ]; then
    source uv_webshop/bin/activate
    webshop --host 0.0.0.0 --port $ENV_PORT >/tmp/webshop_server_${ENV_PORT}.log 2>&1 &
    SERVER_PID=$!
    trap "kill $SERVER_PID" EXIT INT TERM
fi
echo "Launching Environment server... (pid=$SERVER_PID, port=$ENV_PORT)"
sleep 10  # wait for server to start
echo "Environment server is running on port $ENV_PORT"
echo "Logs: /tmp/${TASK}_server_${ENV_PORT}.log"
## ==== End of Environment Server Setup =====



## ====== Start vLLM Server ======
source uv_agentgym_rl/bin/activate
if [ $NEED_MERGE -eq 1 ]; then
    echo "Merging model checkpoints for vLLM serving..."
    python AgentGym-RL/scripts/multiple_model_merger.py --local_dir $MODEL --save_dir $MODEL
    MODEL=$MODEL/huggingface
    echo "Merged model saved to $MODEL"
fi

# ===== 1. Pick a free port =====
VLLM_PORT=$((30000 + RANDOM % (99999-30000+1)))
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $NUM_GPUS"

# ===== 2. Start vLLM in background =====
vllm serve $MODEL \
    --port $VLLM_PORT \
    --served-model-name vllm_model \
    --tensor-parallel-size $NUM_GPUS \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    </dev/null >/tmp/vllm_${VLLM_PORT}.log 2>&1 &
SERVER_PID=$!
echo "Launching vLLM server... (pid=$SERVER_PID, port=$VLLM_PORT)"

# ===== 3. Auto cleanup after exit =====
# This will be called when the script exits or is interrupted
cleanup() { echo "Stopping vLLM..."; kill "$SERVER_PID" 2>/dev/null || true; }
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
wait_for_server $VLLM_PORT /tmp/vllm_${VLLM_PORT}.log
echo "Model: $MODEL | port: $VLLM_PORT | PID: $SERVER_PID | Log: /tmp/vllm_${VLLM_PORT}.log"
## ==== End of vLLM Server Setup =====


echo "TASK: $TASK"
echo "Split: $SPLIT"
echo "Model: $MODEL"
echo "Max Round: $MAX_ROUND"
echo "Num Examples: $NUM_EXAMPLES"
echo "Max Concurrency: $MAX_CONCURRENCY"
echo "Env Port: $ENV_PORT"
echo "Inference File: $INFERENCE_FILE"
echo "Output Dir: $OUTPUT_DIR"

source uv_agentgym_rl/bin/activate
python scripts/interact_with_real_env/run.py \
    --api_key "EMPTY" \
    --base_url "http://localhost:$VLLM_PORT/v1" \
    --model vllm_model \
    --inference_file $INFERENCE_FILE \
    --output_dir $OUTPUT_DIR \
    --task_name $TASK \
    --max_round $MAX_ROUND \
    --num_examples $NUM_EXAMPLES \
    --max_concurrency $MAX_CONCURRENCY \
    --env_server_base http://127.0.0.1:$ENV_PORT