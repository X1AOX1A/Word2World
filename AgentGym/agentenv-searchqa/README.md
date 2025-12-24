# Agent Environments - SearchQA

## Setup

```sh
conda env create -f environment.yml
conda activate agentenv-searchqa
pip install -e .
bash ./setup.sh
```

## Launch

```sh
searchqa --host 0.0.0.0 --port 36001
```

## Environment variables

`SEARCHQA_FAISS_GPU`: Force enable RAG server on GPUs

> Other variables please refer to `env_warpper.py` line 50-68

## Item ID

| Item ID         | Description             | Split |
| --------------- | ----------------------- | ----- |
| 0 ~ 3609        | nq Dataset              | Test  |
| 3610 ~ 14922    | triviaqa Dataset        | Test  |
| 14923 ~ 29189   | popqa Dataset           | Test  |
| 29190 ~ 36594   | hotpotqa Dataset        | Test  |
| 36595 ~ 49170   | 2wikimultihopqa Dataset | Test  |
| 49171 ~ 51587   | musique Dataset         | Test  |
| 51588 ~ 51712   | bamboogle Dataset       | Test  |
| 51713 ~ 130880  | nq Dataset              | Train |
| 130881 ~ 221328 | hotpotqa Dataset        | Train |
