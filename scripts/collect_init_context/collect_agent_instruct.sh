#!/usr/bin/env bash
set -euo pipefail


OUT_ROOT="${OUT_ROOT:-outputs/init_contexts}"
NUM_EXAMPLES="${NUM_EXAMPLES:--1}"
ACTION_FORMAT="${ACTION_FORMAT:-react}"
SPLITS_STR="${SPLITS:-train test}"
TASKS_STR="${TASKS:-alfworld sciworld webshop textworld}"

mkdir -p "$OUT_ROOT"

pick_port() {
  echo $((30000 + RANDOM % (99999-30000+1)))
}

inference_file_for() {
  local task="$1"
  local split="$2"
  if [[ "$split" == "train" ]]; then
    echo "data/train/${task}_train.json"
  elif [[ "$split" == "test" ]]; then
    echo "data/eval/${task}_test.json"
  elif [[ "$split" == "valid_seen" ]]; then
    echo "data/eval/${task}_valid_seen.json"
  elif [[ "$split" == "valid_unseen" ]]; then
    echo "data/eval/${task}_valid_unseen.json"
  else
    echo "Unsupported split: $split" >&2
    exit 2
  fi
}

start_env_server() {
  local task="$1"
  ENV_PORT="$(pick_port)"

  local venv=""
  local cmd=""
  case "$task" in
    alfworld)  venv="uv_alfworld";  cmd="alfworld" ;;
    sciworld)  venv="uv_sciworld";  cmd="sciworld" ;;
    textworld) venv="uv_textworld"; cmd="textworld" ;;
    webshop)   venv="uv_webshop";   cmd="webshop" ;;
    *)
      echo "Unsupported task: $task" >&2
      exit 2
      ;;
  esac

  (
    source "${venv}/bin/activate"
    "$cmd" --host 0.0.0.0 --port "$ENV_PORT" >/tmp/${task}_server_${ENV_PORT}.log 2>&1
  ) &
  SERVER_PID=$!

  echo "Launching env server: task=$task pid=$SERVER_PID port=$ENV_PORT log=/tmp/${task}_server_${ENV_PORT}.log"
  sleep 10
}

stop_env_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

extract_split() {
  local task="$1"
  local split="$2"
  local inference_file
  inference_file="$(inference_file_for "$task" "$split")"

  if [[ ! -f "$inference_file" ]]; then
    echo "Missing inference_file: $inference_file" >&2
    exit 1
  fi

  local out_json="$OUT_ROOT/${task}/agent_instruct_${split}.json"
  mkdir -p "$(dirname "$out_json")"

  source "uv_agentgym_rl/bin/activate"
  python "scripts/collect_init_context/collect_agent_instruct.py" \
    --task_name "$task" \
    --action_format "$ACTION_FORMAT" \
    --env_server_base "http://127.0.0.1:$ENV_PORT" \
    --inference_file "$inference_file" \
    --num_examples "$NUM_EXAMPLES" \
    --output_json "$out_json"
}

for task in $TASKS_STR; do
  echo "==== TASK: $task ===="
  start_env_server "$task"
  trap stop_env_server EXIT INT TERM

  for split in $SPLITS_STR; do
    echo "-- split=$split"
    extract_split "$task" "$split"
  done

  stop_env_server
  trap - EXIT INT TERM
done

echo "Done. Outputs under: $OUT_ROOT"