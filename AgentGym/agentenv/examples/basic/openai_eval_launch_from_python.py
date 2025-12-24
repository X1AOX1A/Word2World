from openai_eval import EvalArguments, main

args = EvalArguments(
    api_key="",
    base_url="https://api.deepseek.com",
    model="deepseek-reasoner",
    inference_file="babyai_test.json",
    output_dir="babyai",
    task_name="babyai",
    max_round=20,
    env_server_base="http://127.0.0.1:8000",
)

if __name__ == "__main__":
    main(vars(args))
