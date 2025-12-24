import argparse
import os
import shutil
import sys
from pathlib import Path


def ensure_huggingface_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install with `pip install huggingface_hub`."
        ) from exc


def download_agent_eval(output_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "X1AOX1A/LLMasWorldModels"
    print(f"Downloading dataset '{repo_id}' (repo_type=dataset, revision=main)...")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",
        local_files_only=False,
    )

    src_root = Path(snapshot_path)
    print(f"Snapshot downloaded to cache: {src_root}")
    print(f"Copying files into: {output_dir}")

    # Copy all files while preserving relative paths
    for root, dirs, files in os.walk(src_root):
        rel = os.path.relpath(root, src_root)
        dest_root = output_dir if rel == "." else output_dir / rel
        Path(dest_root).mkdir(parents=True, exist_ok=True)
        for name in files:
            src_file = Path(root) / name
            dest_file = Path(dest_root) / name
            # Overwrite existing files if present
            shutil.copy2(src_file, dest_file)



def main(argv=None):
    parser = argparse.ArgumentParser(description="Download AgentGym/AgentEval dataset into a folder.")
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Path to output directory (default: data)",
    )
    args = parser.parse_args(argv)

    ensure_huggingface_hub()
    output_dir = Path(args.output_dir)
    download_agent_eval(output_dir)

    # rm -rf ~/.cache/alfworld
    # unzip data/alfworld.zip -d ~/.cache
    os.system("rm -rf ~/.cache/alfworld")
    os.system("unzip -o data/alfworld.zip -d ~/.cache")
    # unzip data/textworld.zip
    os.system("unzip -o data/textworld.zip -d data/textworld/")
    # unzip data/webshop.zip
    os.system("unzip -o data/webshop.zip -d AgentGym/agentenv-webshop/webshop/")
    # unzip data/webshop_index.zip
    os.system("unzip -o data/webshop_index.zip -d AgentGym/agentenv-webshop/webshop/")


if __name__ == "__main__":
    sys.exit(main())