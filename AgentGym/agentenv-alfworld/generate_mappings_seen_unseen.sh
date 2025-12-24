#!/usr/bin/env bash
set -euo pipefail

# Generate mappings for valid_seen and valid_unseen and corresponding eval lists.
# This scans $ALFWORLD_DATA/json_2.1.1/{valid_seen,valid_unseen} for game.tw-pddl.

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)
CFG_DIR="$REPO_ROOT/AgentGym/agentenv-alfworld/configs"
EVAL_DIR="$REPO_ROOT/data/eval"

ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
ROOT="$ALFWORLD_DATA/json_2.1.1"
SEEN_ROOT="$ROOT/valid_seen"
UNSEEN_ROOT="$ROOT/valid_unseen"

if [[ ! -d "$SEEN_ROOT" ]]; then
  echo "ERROR: valid_seen directory not found: $SEEN_ROOT" >&2
  exit 1
fi
if [[ ! -d "$UNSEEN_ROOT" ]]; then
  echo "ERROR: valid_unseen directory not found: $UNSEEN_ROOT" >&2
  exit 1
fi

mkdir -p "$CFG_DIR" "$EVAL_DIR"

# Find current max item_id from existing train/test mappings
get_max_id() {
  local f1="$CFG_DIR/mappings_train.json"
  local f2="$CFG_DIR/mappings_test.json"
  local max=-1
  if [[ -f "$f1" ]]; then
    local m1
    m1=$(rg '"item_id"\s*:\s*(\d+)' -or '$1' "$f1" | awk 'BEGIN{m=-1}{if($1+0>m)m=$1+0}END{print m}') || true
    [[ -n "$m1" ]] && [[ "$m1" =~ ^[0-9]+$ ]] && (( m1 > max )) && max=$m1
  fi
  if [[ -f "$f2" ]]; then
    local m2
    m2=$(rg '"item_id"\s*:\s*(\d+)' -or '$1' "$f2" | awk 'BEGIN{m=-1}{if($1+0>m)m=$1+0}END{print m}') || true
    [[ -n "$m2" ]] && [[ "$m2" =~ ^[0-9]+$ ]] && (( m2 > max )) && max=$m2
  fi
  echo $max
}

MAX_ID=$(get_max_id)
if [[ "$MAX_ID" -lt 0 ]]; then
  echo "ERROR: Failed to detect max item_id from existing mappings." >&2
  exit 1
fi
START_SEEN=$((MAX_ID + 1))

# Build seen list
SEEN_MAP="$CFG_DIR/mappings_valid_seen.json"
echo "Building $SEEN_MAP starting at id=$START_SEEN"
{
  echo "["
  id=$START_SEEN
  first=1
  # Use LC_ALL=C sort for deterministic ordering
  while IFS= read -r p; do
    task_id=$(basename "$(dirname "$p")")
    task_type=$(basename "$(dirname "$(dirname "$p")")")
    line="    {\"item_id\": $id, \"task_type\": \"$task_type\", \"task_id\": \"$task_id\"}"
    if [[ $first -eq 1 ]]; then
      echo "$line"
      first=0
    else
      echo ","
      echo "$line"
    fi
    id=$((id+1))
  done < <(find "$SEEN_ROOT" -type f -name game.tw-pddl | LC_ALL=C sort)
  echo "]"
} > "$SEEN_MAP"

LAST_SEEN_ID=$((id-1))
COUNT_SEEN=$((LAST_SEEN_ID - START_SEEN + 1))
echo "seen count=$COUNT_SEEN ids=[$START_SEEN,$LAST_SEEN_ID]"

# Build unseen list
START_UNSEEN=$((LAST_SEEN_ID + 1))
UNSEEN_MAP="$CFG_DIR/mappings_valid_unseen.json"
echo "Building $UNSEEN_MAP starting at id=$START_UNSEEN"
{
  echo "["
  id=$START_UNSEEN
  first=1
  while IFS= read -r p; do
    task_id=$(basename "$(dirname "$p")")
    task_type=$(basename "$(dirname "$(dirname "$p")")")
    line="    {\"item_id\": $id, \"task_type\": \"$task_type\", \"task_id\": \"$task_id\"}"
    if [[ $first -eq 1 ]]; then
      echo "$line"
      first=0
    else
      echo ","
      echo "$line"
    fi
    id=$((id+1))
  done < <(find "$UNSEEN_ROOT" -type f -name game.tw-pddl | LC_ALL=C sort)
  echo "]"
} > "$UNSEEN_MAP"

LAST_UNSEEN_ID=$((id-1))
COUNT_UNSEEN=$((LAST_UNSEEN_ID - START_UNSEEN + 1))
echo "unseen count=$COUNT_UNSEEN ids=[$START_UNSEEN,$LAST_UNSEEN_ID]"

# Build eval files
SEEN_EVAL="$EVAL_DIR/alfworld_valid_seen.json"
UNSEEN_EVAL="$EVAL_DIR/alfworld_valid_unseen.json"

echo "Writing $SEEN_EVAL"
{
  echo "["
  first=1
  for ((i=START_SEEN; i<=LAST_SEEN_ID; i++)); do
    line="    {\"item_id\": \"alfworld_${i}\"}"
    if [[ $first -eq 1 ]]; then
      echo "$line"
      first=0
    else
      echo ","
      echo "$line"
    fi
  done
  echo "]"
} > "$SEEN_EVAL"

echo "Writing $UNSEEN_EVAL"
{
  echo "["
  first=1
  for ((i=START_UNSEEN; i<=LAST_UNSEEN_ID; i++)); do
    line="    {\"item_id\": \"alfworld_${i}\"}"
    if [[ $first -eq 1 ]]; then
      echo "$line"
      first=0
    else
      echo ","
      echo "$line"
    fi
  done
  echo "]"
} > "$UNSEEN_EVAL"

echo "Done."

