## Collect Agent Instruct
export OUT_ROOT="data/init_contexts"
export NUM_EXAMPLES=-1               # -1 means all
export ACTION_FORMAT=react           # react | function_calling | code_as_action
for TASK in alfworld sciworld textworld webshop; do
  for SPLIT in train test; do
    export TASKS="$TASK"
    export SPLITS="$SPLIT"
    bash scripts/collect_init_context/collect_agent_instruct.sh
  done
done

for TASK in alfworld; do
  for SPLIT in valid_unseen; do
    export TASKS="$TASK"
    export SPLITS="$SPLIT"
    bash scripts/collect_init_context/collect_agent_instruct.sh
  done
done


## Collect WM Instruct
source uv_alfworld/bin/activate
python scripts/collect_init_context/collect_wm_instruct_alfworld.py \
  --agent_instruct data/init_contexts/alfworld/agent_instruct_train.json \
  --output_file data/init_contexts/alfworld/wm_instruct_train.json \
  --split train

python scripts/collect_init_context/collect_wm_instruct_alfworld.py \
  --agent_instruct data/init_contexts/alfworld/agent_instruct_test.json \
  --output_file data/init_contexts/alfworld/wm_instruct_test.json \
  --split valid_train

python scripts/collect_init_context/collect_wm_instruct_alfworld.py \
  --agent_instruct data/init_contexts/alfworld/agent_instruct_valid_seen.json \
  --output_file data/init_contexts/alfworld/wm_instruct_valid_seen.json \
  --split valid_seen

python scripts/collect_init_context/collect_wm_instruct_alfworld.py \
  --agent_instruct data/init_contexts/alfworld/agent_instruct_valid_unseen.json \
  --output_file data/init_contexts/alfworld/wm_instruct_valid_unseen.json \
  --split valid_unseen


## Note: it will be different pf different runs
source uv_sciworld/bin/activate
python scripts/collect_init_context/collect_wm_instruct_sciworld.py \
  --agent_instruct data/init_contexts/sciworld/agent_instruct_train.json \
  --output_file data/init_contexts/sciworld/wm_instruct_train.json
python scripts/collect_init_context/collect_wm_instruct_sciworld.py \
  --agent_instruct data/init_contexts/sciworld/agent_instruct_test.json \
  --output_file data/init_contexts/sciworld/wm_instruct_test.json

source uv_textworld/bin/activate
python scripts/collect_init_context/collect_wm_instruct_textworld.py \
  --agent_instruct data/init_contexts/textworld/agent_instruct_train.json \
  --output_file data/init_contexts/textworld/wm_instruct_train.json
python scripts/collect_init_context/collect_wm_instruct_textworld.py \
  --agent_instruct data/init_contexts/textworld/agent_instruct_test.json \
  --output_file data/init_contexts/textworld/wm_instruct_test.json

source uv_webshop/bin/activate
python scripts/collect_init_context/collect_wm_instruct_webshop.py \
  --agent_instruct data/init_contexts/webshop/agent_instruct_train.json \
  --output_file data/init_contexts/webshop/wm_instruct_train.json
python scripts/collect_init_context/collect_wm_instruct_webshop.py \
  --agent_instruct data/init_contexts/webshop/agent_instruct_test.json \
  --output_file data/init_contexts/webshop/wm_instruct_test.json