export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

seed="42"

# model_path="/cpfs01/shared/public/guohonglin/agentenv_scripts/output/CodeLlama-7b-Instruct-hf/checkpoint-369"
# model_path="/cpfs01/shared/public/guohonglin/agentenv_scripts/output/Llama-2-7b-chat-hf/checkpoint-369"
model_path="/cpfs01/shared/public/guohonglin/agentenv_scripts/Qwen2-7B-Instruct"

# inference_file="/cpfs01/shared/public/guohonglin/agentenv_scripts/AgentEval/webshop_test.json"
inference_file="/cpfs01/shared/public/guohonglin/agentenv_scripts/webshop_test_1.json"

# output_file="/cpfs01/shared/public/guohonglin/agentenv_scripts/webshop_test_result_test_codellama7b.jsonl"
output_file="/cpfs01/shared/public/guohonglin/agentenv_scripts/tmp.jsonl"

chat_template="chatml"
action_format="code_as_action"

# environment parameters
task_name="webshop"
max_round="10"
env_server_base="http://127.0.0.1:8000"

python -u base_eval.py \
        --model_path "${model_path}" \
        --inference_file "${inference_file}" \
        --output_file "${output_file}" \
        --task_name "${task_name}" \
        --seed "${seed}" \
        --max_round "${max_round}" \
        --env_server_base "${env_server_base}" \
        --chat_template "${chat_template}" \
        --action_format "${action_format}" \
        --use_vllm
