set -euo pipefail

TASK=$1  # alfworld, alfworld_valid_seen, alfworld_valid_unseen, sciworld, textworld, webshop
test_file_root=$2
max_workers=${3:-50}

## ===== Start Environment Server =====
ENV_PORT=$((30000 + RANDOM % (99999-30000+1)))
if [ "$TASK" = "alfworld" ]; then
    source uv_alfworld/bin/activate
    alfworld --host 0.0.0.0 --port $ENV_PORT >/tmp/alfworld_server_${ENV_PORT}.log 2>&1 &
    SERVER_PID=$!
    trap "kill $SERVER_PID" EXIT INT TERM
elif [ "$TASK" = "alfworld_valid_seen" ]; then
    source uv_alfworld/bin/activate
    alfworld --host 0.0.0.0 --port $ENV_PORT >/tmp/alfworld_server_${ENV_PORT}.log 2>&1 &
    SERVER_PID=$!
    trap "kill $SERVER_PID" EXIT INT TERM
elif [ "$TASK" = "alfworld_valid_unseen" ]; then
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

source uv_agentgym_rl/bin/activate
for i in {1..10}; do
    python scripts/interact_with_world_model/cal_wm2real.py \
        --task $TASK \
        --port $ENV_PORT \
        --test_file_root $test_file_root \
        --max_workers $max_workers
    sleep 5
done