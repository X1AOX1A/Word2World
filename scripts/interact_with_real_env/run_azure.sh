#!/usr/bin/env bash
set -euo pipefail

TASK=$1                     # alfworld, sciworld, textworld, webshop
RUN=${2:-0}                 # run id for multiple runs, just for separating output dirs
MODEL=${3:-gpt-4o}          # model name
MAX_CONCURRENCY=${4:-150}   # max concurrency
MAX_ROUND=${5:-50}
NUM_EXAMPLES=${6:--1}
SPLIT=${7:-test}            # train or test(valid_train), valid_seen, valid_unseen
OUTPUT_ROOT=${8:-outputs}

OUTPUT_DIR=$OUTPUT_ROOT/interaction/real_env/$SPLIT/${TASK}/$MODEL/${TASK}_maxround${MAX_ROUND}_run${RUN}


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


echo "TASK: $TASK"
echo "Split: $SPLIT"
echo "Model: $MODEL"
echo "Max Concurrency: $MAX_CONCURRENCY"
echo "Max Round: $MAX_ROUND"
echo "Num Examples: $NUM_EXAMPLES"
echo "Port: $ENV_PORT"
echo "Inference File: $INFERENCE_FILE"
echo "Output Dir: $OUTPUT_DIR"

source uv_agentgym_rl/bin/activate
python scripts/interact_with_real_env/run.py \
    --api_key "azure" \
    --base_url "" \
    --model $MODEL \
    --inference_file $INFERENCE_FILE \
    --output_dir $OUTPUT_DIR \
    --task_name $TASK \
    --max_round $MAX_ROUND \
    --num_examples $NUM_EXAMPLES \
    --max_concurrency $MAX_CONCURRENCY \
    --env_server_base http://127.0.0.1:$ENV_PORT