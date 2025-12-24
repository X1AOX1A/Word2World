# Agent Environments - Webarena

## Setup

Before setup, please:

- set availiable Web Server urls and OPENAI api key for llm judge.
- **please see [WebArena](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md), build your web server, replace all urls in `setup.sh` and `agentenv-webarena/_init_.py` to your own urls.**

```sh
conda create -n agentenv-webarena python=3.10.13
conda activate agentenv-webarena
source ./setup.sh
```

## Launch

```sh
webarena --host 0.0.0.0 --port 8000
```

## Memory cost for parallel-run

| parallel-run envs | memory utilization |
| ----------------- | ------------------ |
| 1  env            | 1717 MB            |
| 5 envs            | 4702 MB            |
| 10 envs           | 8450 MB            |
| 20 envs           | 16310 MB           |

**We recommend you preserve 1GB memory per Env(interact client) to make sure it runs without crashes**

## BUG

When try action like `goto [url]`, which url is unable to access, program will crash.

> temporarily fixed by adding a url accessible test befor action
> Added retry to `env.reset()` and `env.step()`, you can set `max_retries` in `./webarena/browser_env/envs.py line 162` and `./webarena/browser_env/actions.py line 1202` to change the retry times.
> If all retries failed, the env will return `obs['text'] = "TimeoutError"`

## TIPS

- If you'd like to build websites by yourself, please see [WebArena](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md)
