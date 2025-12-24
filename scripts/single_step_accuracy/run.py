import os
import json
def write_json(dict_objs, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w+", encoding='utf-8') as f:
        json.dump(dict_objs, f, indent=4, ensure_ascii=False)

def read_json(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        dict_objs = json.load(f)
    return dict_objs


import json
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
class BasicGenerator:
    def __init__(self, vllm_args, sample_args):
        self.llm = LLM(**vllm_args)
        self.sampling_params = SamplingParams(**sample_args)
        self.tokenizer = AutoTokenizer.from_pretrained(vllm_args["model"], trust_remote_code=True)

    def apply_chat_template(self, conversations_list):
        return [self.tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
                for conversations in conversations_list]

    def post_process(self, outputs):
        outputs = [output.outputs[0].text for output in outputs]
        return outputs

    def generate_batch(self, conversations_batch):
        inputs_batch = self.apply_chat_template(conversations_batch)
        outputs_batch = self.llm.generate(inputs_batch, sampling_params=self.sampling_params)
        outputs_batch = self.post_process(outputs_batch)
        assert len(conversations_batch) == len(outputs_batch)
        return outputs_batch

    def unload_model(self):
        import gc
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

        # Delete the llm object and free the memory
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("Successfully delete the llm pipeline and free the GPU memory!")



def read_data(json_path, num_traj=-1):
    # Guard against accidentally passing non-JSON paths (e.g., figure PNGs)
    assert str(json_path).endswith(".json"), f"Not a JSON file: {json_path}"
    data = read_json(json_path)
    conversations = []
    ground_truths = []
    if num_traj>0:
        data = data[:num_traj]
    for item in data:
        n_messages = len(item["messages"])
        if n_messages % 2 != 0:
            n_messages -= 1
        # [system, user, assistant, user, assistant, user, ...]
        # -> conversations1, [system, user], ground_truth1 [assistant]
        # -> conversations2, [system, user, assistant, user], ground_truth2 [assistant]
        for i in range(0, n_messages, 2):
            conversations.append(item["messages"][:i+2])
            ground_truths.append(item["messages"][i+2]["content"])
    print(f"Read {len(conversations)} samples from {json_path}")
    return conversations, ground_truths

def main(json_path, model, fig_path, output_file, num_traj=-1):
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    vllm_args = {
        "model": model,
        "tensor_parallel_size": torch.cuda.device_count(),
        "enforce_eager": True,
    }

    sample_args = {
        "temperature": 0,
        "max_tokens": 2000,
    }

    conversations, ground_truths = read_data(json_path, num_traj=num_traj)
    generator = BasicGenerator(vllm_args, sample_args)
    outputs_batch = generator.generate_batch(conversations)
    generator.unload_model()

    outputs = []
    for conv, gt, pred in zip(conversations, ground_truths, outputs_batch):
        outputs.append({
            "conversation": conv,
            "ground_truth": gt,
            "prediction": pred,
            "steps": len(conv)//2,
            "correct": pred.strip() == gt.strip()
        })
    write_json(outputs, output_file)
    print(f"Written outputs to {output_file}")

    # stat the accuracy for different number of steps
    step_stats = {}
    for output in outputs:
        steps = output["steps"]
        if steps not in step_stats:
            step_stats[steps] = {"correct": 0, "total": 0}
        step_stats[steps]["total"] += 1
        if output["correct"]:
            step_stats[steps]["correct"] += 1
    for steps in sorted(step_stats.keys()):
        stats = step_stats[steps]
        acc = stats["correct"] / stats["total"]
        print(f"Steps: {steps}, Acc: {acc*100:.2f}% ({stats['correct']}/{stats['total']})")

    # plot the accuracy curve
    import matplotlib.pyplot as plt
    steps = sorted(step_stats.keys())
    accs = [step_stats[s]["correct"] / step_stats[s]["total"] for s in steps]
    # create new figure
    plt.figure(figsize=(8, 6))
    plt.plot(steps, accs, marker='o')
    plt.xlabel("Number of Steps")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Steps")
    plt.grid(True)
    # add average accuracy
    avg_acc = sum([step_stats[s]["correct"] for s in steps]) / sum([step_stats[s]["total"] for s in steps])
    print(f"Acc: {avg_acc*100:.2f}%")
    plt.axhline(y=avg_acc, color='r', linestyle='--', label=f'Average Acc: {avg_acc*100:.2f}%')
    plt.legend()
    plt.savefig(fig_path)
    print(f"Saved accuracy curve to {fig_path}")

    return avg_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to the JSON file containing test trajectories.")
    parser.add_argument("--model", type=str, default=None, help="Model name or path for vLLM.")
    parser.add_argument("--output_root", type=str, default="./outputs/single_step_accuracy", help="Root directory for output files.")
    args = parser.parse_args()

    data = args.data
    model = args.model
    fig_name = os.path.join(args.output_root, "accuracy_curve.png")
    output_file = os.path.join(args.output_root, "outputs.jsonl")
    metrics_file = os.path.join(args.output_root, "metrics.json")
    print(f"Evaluating model: {model} on data: {data}")
    print(f"Output fig: {fig_name}")
    print(f"Output file: {output_file}")
    print(f"Metrics file: {metrics_file}")
    avg_acc = main(data, model, fig_name, output_file, num_traj=-1)
    write_json({"average_accuracy": avg_acc}, metrics_file)