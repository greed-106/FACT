<div align="center">

# FACT: Failure-Aware Causal Training<br>for World-Action Models

**One causal diffusion transformer — act, then imagine.**

[Quanquan Peng](https://pengqq.com)<sup>\*</sup> · [Yutong Liang](https://lyt0112.com)<sup>\*</sup> · [Rui Yan](https://jerryyan24.github.io) · [Nicklas Hansen](https://nicklashansen.com) · [Xiaolong Wang](https://xiaolonw.github.io)


[![Project Page](https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-fact--wam.github.io-4b8bbe)](https://fact-wam.github.io/)
[![Paper](https://img.shields.io/badge/%F0%9F%93%84%20Paper-PDF-b31b1b)](https://arxiv.org/abs/2608.10232)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-fact--wam-ffd21e)](https://huggingface.co/Bariona/fact-wam)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Spaces-ff7c00)](https://huggingface.co/spaces/Bariona/fact-world-action-model)
[![License](https://img.shields.io/badge/License-Apache%202.0-3da639)](LICENSE)

</div>

## 📖 Overview

**FACT** is a causal world-action model: one causal diffusion transformer jointly denoises **robot actions**, a **task-progress value**, and **future video**, all conditioned on the executed action.

<p align="center">
  <img src="https://fact-wam.github.io/static/images/method/frame_14.png" width="92%" alt="FACT architecture: a shared causal diffusion transformer denoises action, value, and future-video tokens; value and future video condition on the clean action slot, not the noisy one.">
</p>

- **Act, then imagine.** Future video and value condition on the clean action, never the reverse — future prediction sharpens actions without leaking targets, and deployment decodes actions without generating video.
- **Failures teach consequences.** Failure rollouts skip the action-imitation loss but still supervise the observed failed future and a lowered value.
- **Optional best-of-N scoring.** The value head ranks sampled action candidates at inference.

This repository is the official implementation, containing the end-to-end RoboTwin pipeline: **data prep → training → inference → closed-loop simulator evaluation**.

> 🎮 **Try it live** — run the released checkpoint in your browser on Hugging Face Spaces: [Bariona/fact-world-action-model](https://huggingface.co/spaces/Bariona/fact-world-action-model)

| Path | What it is |
| --- | --- |
| `world_action_model/` | Model, trainer, inference pipeline, transforms, config (`configs/robotwin.py`) |
| `fact_train/`, `fact_datasets/` | Training harness and dataset library |
| `scripts/` | CLI entrypoints, run as `uv run --no-sync python -m scripts.<name>` from the repo root |
| `evaluation/robotwin/` | Closed-loop simulator evaluation |

## 🛠️ Installation

```bash
bash setup_env.sh        # locked uv environment, model weights in /data/shared/FACT, RoboTwin download
```

Already have the checkpoint or dataset: `SKIP_MODEL_DOWNLOAD=1 SKIP_DATA_DOWNLOAD=1 bash setup_env.sh`

The script creates `FACT/.venv`; no Conda activation is needed. It stores the
Wan2.2 base model and FACT checkpoint under `/data/shared/FACT/models` by
default; override that root with `FACT_SHARED_ROOT`. Run subsequent commands
as `uv run --no-sync …`.

For Wan2.2, `setup_env.sh` uses `https://hf-mirror.com` and downloads every
large shard with 32 concurrent HTTP Range requests, checking its SHA-256 before
accepting it. Set `HF_PARALLEL_DOWNLOAD_WORKERS` to tune the connection count
or `HF_ENDPOINT` to use another Hugging Face endpoint.

## 📦 Model & Data Download

Equivalent of what `setup_env.sh` does:

```bash
# Wan2.2 base model
uv run --no-sync huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --local-dir ./models/Wan2.2-TI2V-5B-Diffusers

# RoboTwin demonstrations
uv run --no-sync huggingface-cli download Bariona/robotwin-v2 robotwin-v2.tar \
  --repo-type dataset --local-dir ./datasets
tar -xf ./datasets/robotwin-v2.tar -C ./datasets   # -> datasets/RoboTwin/{Clean,Randomized}/<task>/
```

Trained FACT checkpoint ([`Bariona/fact-wam`](https://huggingface.co/Bariona/fact-wam)):

```bash
uv run --no-sync huggingface-cli download Bariona/fact-wam --local-dir ./models/fact-wam
```

## 📊 Data Preprocessing

**1. Norm stats + per-episode T5 embedding caches** (required; subset via `--dataset_glob 'Clean/*'`):

```bash
uv run --no-sync python -m scripts.prepare_robotwin \
  --robotwin_root ./datasets/RoboTwin \
  --wan_model_path ./models/Wan2.2-TI2V-5B-Diffusers \
  --output_dir ./artifacts/robotwin
```

**2. VAE latent cache** — training reads it by default; export `FACT_USE_CACHED_VAE_LATENTS=0` to train from raw video instead:

```bash
uv run --no-sync python -m scripts.compute_vae_latents --batch_size 32   # match training BATCH_SIZE_PER_GPU
```

**3. Dataloader check** (optional):

```bash
uv run --no-sync python -m scripts.test_dataloader --config world_action_model.configs.robotwin \
  --num_workers 0 --batch_size 2 --num_batches 2
```

## 🚀 Training

Edit the USER SETTINGS block at the top of `world_action_model/configs/robotwin.py`, then:

```bash
uv run --no-sync python -m scripts.train --config world_action_model.configs.robotwin.config
```

## ⚡ Inference

Serve a trained checkpoint (to use the released one instead, pass `--transformer_path ./models/fact-wam/transformer --stats_path ./models/fact-wam/norm_stats_delta.json`):

```bash
uv run --no-sync python -m scripts.inference_server \
  --model_id ./models/Wan2.2-TI2V-5B-Diffusers \
  --transformer_path ./experiments/robotwin/models/<checkpoint>/transformer \
  --stats_path ./artifacts/robotwin/norm_stats_delta.json \
  --port 8093
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--verbose --verbose_dir ./tmp/verbose` | Dump each request's input image and output action/value |
| `--return_images` | Decode and return predicted frames (needed for the client-side `VIS_DIR` video dump) |
| `--enable_prefix_cache` | KV-cache the fixed prefix |
| `--skip_future_state_value` | Action-only decoding |

## 🤖 RoboTwin Evaluation

The evaluation launchers target [RoboTwin-Phys](https://github.com/yefeng00/RoboTwin_Phys) and invoke its native `scripts/eval_policy.py` directly. Both the FACT server and the RoboTwin client run in FACT's `.venv`; RoboTwin-Phys is used only as a source and asset checkout. FACT's lockfile includes the simulator packages and aligns the shared simulator/GPU versions with RoboTwin-Phys. WAM-only packages such as Diffusers, Transformers, and Hugging Face Hub keep FACT's compatible versions. RoboTwin-Phys's existing uv environment and lockfile are never read, synchronized, or modified. `TEST_NUM` is passed to RoboTwin-Phys as `--eval_num_episodes`; no patch to the benchmark is needed.

```bash
# settings live in evaluation/robotwin/launch_config.yml
bash evaluation/robotwin/launch_server.sh                                # terminal 1
bash evaluation/robotwin/launch_client.sh beat_block_hammer phys_random_all  # terminal 2
bash evaluation/robotwin/eval_all_tasks.sh phys_random_all 50                # ... or sweep all 50 tasks
```

`phys_random_all` lets RoboTwin-Phys automatically select its nine task-specific PhysTTT configurations. Per-episode results land in RoboTwin-Phys's `eval_result/`; the sweep writes per-task and average success rates to `./eval_runs/<config>_<timestamp>/`. Its CSV uses RoboTwin-Phys's actual attempt count, which includes expert-infeasible evaluation slots.

For dynamic multi-GPU scheduling, use the SQLite-backed launcher below instead of starting `launch_server.sh` yourself. An idle worker immediately claims the next task. By default it starts one persistent FACT inference server and one sequential RoboTwin-Phys worker per GPU; `--workers-per-gpu` creates independent concurrent server/client slots on each selected GPU. It writes `scheduler.sqlite3`, worker/client logs, `results.csv`, and `summary.json` below the specified new output directory.

```bash
uv run --no-sync python -m evaluation.robotwin.eval_all_tasks_multi_gpu \
  --gpu-ids 0,1,2,3 \
  --task-config phys_random_all \
  --test-num 100 \
  --output-dir ./eval_runs/phys_random_all_100_multi_gpu
```

All selected GPUs must have room for the requested number of FACT server and RoboTwin-Phys simulator slots. For example, add `--workers-per-gpu 2` for two concurrent task rollouts per GPU. The launcher assigns one consecutive port per slot starting at `18093`; change `--base-port` if that range is occupied. Run `--dry-run` with the same arguments to create and inspect the 50-job SQLite queue without launching servers.

## 📝 Citation

If you find FACT useful, please consider citing:

```bibtex
@inproceedings{peng2026fact,
  title     = {FACT: Failure-Aware Causal Training for World-Action Models},
  author    = {Peng, Quanquan and Liang, Yutong and Yan, Rui and Hansen, Nicklas and Wang, Xiaolong},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

## 📄 License

This project is released under the [Apache 2.0 license](LICENSE).

## 🙏 Acknowledgements

FACT builds on [Wan2.2](https://github.com/Wan-Video/Wan2.2), [giga-world-policy](https://github.com/open-gigaai/giga-world-policy), [GigaTrain](https://github.com/open-gigaai/giga-train), and [GigaDatasets](https://github.com/open-gigaai/giga-datasets); simulation experiments use the [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) benchmark. Thanks to the authors for open-sourcing their work.
