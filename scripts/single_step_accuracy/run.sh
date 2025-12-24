TASK=$1
MODEL=$2
OUTPUT_ROOT=$3

echo "Task: $TASK"
echo "Model: $MODEL"
echo "Output Root: $OUTPUT_ROOT"

source uv_agentgym_rl/bin/activate

if [[ $TASK == "alfworld" ]]; then
    DATA="data/llama_factory/alfworld_test_with_env_195.json"
elif [[ $TASK == "alfworld_valid_seen" ]]; then
    DATA="data/llama_factory/alfworld_valid_seen_with_env_169.json"
elif [[ $TASK == "alfworld_valid_unseen" ]]; then
    DATA="data/llama_factory/alfworld_valid_unseen_with_env_150.json"
elif [[ $TASK == "sciworld" ]]; then
    DATA="data/llama_factory/sciworld_test_with_env_195.json"
elif [[ $TASK == "textworld" ]]; then
    DATA="data/llama_factory/textworld_test_173.json"
elif [[ $TASK == "webshop" ]]; then
    DATA="data/llama_factory/webshop_test_109.json"
elif [[ $TASK == "stabletoolbench" ]]; then
    DATA="data/llama_factory/stabletoolbench_test_2000.json"
else
    echo "Unknown task name: $TASK"
    exit 1
fi

python scripts/single_step_accuracy/run.py --data $DATA --model $MODEL --output_root $OUTPUT_ROOT