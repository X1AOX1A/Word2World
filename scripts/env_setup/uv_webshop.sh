# !/bin/bash
set -ex


curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv uv_webshop --python 3.10
source uv_webshop/bin/activate
uv pip install --upgrade pip
export UV_LINK_MODE=copy

cd AgentGym/agentenv-webshop/webshop
# ./setup.sh -d all

# Install Python Dependencies
uv pip install packaging wheel setuptools
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv pip install vllm==0.8.2
uv pip install -r requirements.txt --no-deps

# conda install mkl
# conda install -c conda-forge faiss-cpu
uv pip install mkl faiss-cpu
uv pip install gdown


# Install Environment Dependencies via `conda`
# conda install -c pytorch faiss-cpu;
# conda install -c conda-forge openjdk=11;
sudo apt-get update
sudo apt install openjdk-11-jdk -y

## We have packed into data/webshop.zip
# # Download dataset into `data` folder via `gdown` command
# mkdir -p data;
# cd data;
# gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib; # items_shuffle_1000 - product scraped info
# gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu; # items_ins_v2_1000 - product attributes
# gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB; # items_shuffle
# gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi; # items_ins_v2
# gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O # items_human_ins
# cd ..

uv pip install packaging wheel
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv pip install vllm==0.8.2
# spacy 3.7.2 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# weasel 0.3.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# The warnings can be safely ignored.

# Build search engine index
uv pip install nltk itsdangerous pyjnius pytz python-dateutil huggingface_hub threadpoolctl onnxruntime setuptools
cd search_engine

## We have packed into data/webshop_index.zip
# mkdir -p resources resources_100 resources_1k resources_100k
# python convert_product_file_format.py # convert items.json => required doc format
# mkdir -p indexes
# ./run_indexing.sh

uv pip install openai azure.identity
uv pip install omegaconf

cd ../..
uv pip install -e .
uv pip install thinc numpy==1.26.4 langcodes spacy==3.7.1
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl"
