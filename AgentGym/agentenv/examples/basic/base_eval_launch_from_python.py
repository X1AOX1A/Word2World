from base_eval import EvalArguments, main

args = EvalArguments(
    model_path="",
    inference_file="",
    output_file="",
    task_name="webshop",
    seed=42,
    max_round=10,
    env_server_base="http://127.0.0.1:8000",
    chat_template="chatml",
    use_vllm=True,
    action_format="code_as_action",
)

if __name__ == "__main__":
    main(vars(args))
