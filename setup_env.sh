#!/usr/bin/env bash
# End-to-end setup: FACT's locked uv environment, Wan2.2 checkpoint, and
# RoboTwin demonstrations.
#
#   bash setup_env.sh
#
# Overrides: PYTHON_BIN, CUDA_HOME, TORCH_CUDA_ARCH_LIST, FACT_SHARED_ROOT,
# WAN_MODEL_DIR, FACT_CHECKPOINT_DIR, HF_ENDPOINT,
# HF_PARALLEL_DOWNLOAD_WORKERS, HF_TOKEN, SKIP_MODEL_DOWNLOAD=1,
# SKIP_FACT_CHECKPOINT_DOWNLOAD=1, SKIP_DATA_DOWNLOAD=1.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.10}"
FACT_SHARED_ROOT="${FACT_SHARED_ROOT:-/data/shared/FACT}"
MODEL_ROOT="${MODEL_ROOT:-${FACT_SHARED_ROOT}/models}"
WAN_MODEL_DIR="${WAN_MODEL_DIR:-${MODEL_ROOT}/Wan2.2-TI2V-5B-Diffusers}"
FACT_CHECKPOINT_DIR="${FACT_CHECKPOINT_DIR:-${MODEL_ROOT}/fact-wam}"
DATASETS_DIR="${ROOT_DIR}/datasets"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_PARALLEL_DOWNLOAD_WORKERS="${HF_PARALLEL_DOWNLOAD_WORKERS:-32}"
# RoboTwin-Phys's CuRobo build needs CUDA. These defaults match this H100
# machine and can be overridden for another CUDA toolkit or GPU architecture.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

log() { echo "[setup] $*"; }

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python 3.10 was not found at '${PYTHON_BIN}'. Set PYTHON_BIN to a Python 3.10 executable." >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)'; then
  echo "Error: FACT is locked for Python 3.10; '${PYTHON_BIN}' is not Python 3.10." >&2
  exit 1
fi
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "Error: CUDA compiler not found at '${CUDA_HOME}/bin/nvcc'. Set CUDA_HOME before syncing CuRobo." >&2
  exit 1
fi
export CUDA_HOME TORCH_CUDA_ARCH_LIST
export HF_ENDPOINT

# This creates only ${ROOT_DIR}/.venv. It contains the WAM and RoboTwin-Phys
# simulator dependencies, while RoboTwin-Phys's checkout and uv environment
# remain untouched.
cd "${ROOT_DIR}"
log "Synchronizing FACT's locked uv environment ..."
uv sync --locked --python "${PYTHON_BIN}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  uv run --project "${ROOT_DIR}" --no-sync huggingface-cli login --token "${HF_TOKEN}"
fi

# ── Wan2.2 checkpoint ─────────────────────────────────────────────────────────
# The Hub's regular downloader is used only for small configuration files.  The
# nine checkpoint shards are each downloaded with independent byte ranges so a
# high-bandwidth machine does not underuse its network connection.
WAN_SMALL_FILES=(
  model_index.json
  scheduler/scheduler_config.json
  text_encoder/config.json
  text_encoder/model.safetensors.index.json
  tokenizer/special_tokens_map.json
  tokenizer/spiece.model
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
  transformer/config.json
  transformer/diffusion_pytorch_model.safetensors.index.json
  vae/config.json
)
WAN_LARGE_FILES=(
  text_encoder/model-00001-of-00003.safetensors
  text_encoder/model-00002-of-00003.safetensors
  text_encoder/model-00003-of-00003.safetensors
  transformer/diffusion_pytorch_model-00001-of-00005.safetensors
  transformer/diffusion_pytorch_model-00002-of-00005.safetensors
  transformer/diffusion_pytorch_model-00003-of-00005.safetensors
  transformer/diffusion_pytorch_model-00004-of-00005.safetensors
  transformer/diffusion_pytorch_model-00005-of-00005.safetensors
  vae/diffusion_pytorch_model.safetensors
)
WAN_LARGE_SHA256=(
  a8e861969c7433e707cc5a74065d795d36cca07ec96eb6763eb4083df7248f58
  d57d948ece4837d850b7a859a4415121d57cacf8b9ee1d4db200c67f592902d7
  0da9ee284e21d1406df708788db1d502d95d75f69faa25cd26151bf8829b7c5f
  511bec832a201caa410d09c5ce7dbbf8ad2708c345d82038f684fc74cce982be
  7c42724912b1911429125dc50c0e9a49ccbada5a601b657d4ed2e15e7597c193
  e9c3d0c76de786566382f8258101fea973ae37681c6e9fe0e5fe1fb93b806424
  a331121771790939678db6f585553fd5184609f7d02593c699a4d241b0d834c5
  78b655685c47efdb2349f36826bb101264e9f212a16325d584aeb5f53c88e719
  62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b
)

wan_model_ready() {
  local relative_path
  for relative_path in "${WAN_SMALL_FILES[@]}" "${WAN_LARGE_FILES[@]}"; do
    [[ -f "${WAN_MODEL_DIR}/${relative_path}" ]] || return 1
  done
}

if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  if wan_model_ready; then
    log "Wan model already present at ${WAN_MODEL_DIR}; skipping download."
  else
    log "Downloading Wan-AI/Wan2.2-TI2V-5B-Diffusers -> ${WAN_MODEL_DIR}"
    mkdir -p "${WAN_MODEL_DIR}"
    uv run --project "${ROOT_DIR}" --no-sync huggingface-cli download \
      Wan-AI/Wan2.2-TI2V-5B-Diffusers "${WAN_SMALL_FILES[@]}" \
      --local-dir "${WAN_MODEL_DIR}"
    for index in "${!WAN_LARGE_FILES[@]}"; do
      relative_path="${WAN_LARGE_FILES[index]}"
      if [[ ! -f "${WAN_MODEL_DIR}/${relative_path}" ]]; then
        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/download_hf_parallel.py" \
          Wan-AI/Wan2.2-TI2V-5B-Diffusers "${relative_path}" \
          "${WAN_MODEL_DIR}/${relative_path}" \
          --workers "${HF_PARALLEL_DOWNLOAD_WORKERS}" \
          --sha256 "${WAN_LARGE_SHA256[index]}"
      fi
    done
  fi
fi

# ── FACT inference checkpoint ─────────────────────────────────────────────────
if [[ "${SKIP_FACT_CHECKPOINT_DOWNLOAD:-0}" != "1" ]]; then
  if [[ -f "${FACT_CHECKPOINT_DIR}/transformer/diffusion_pytorch_model.bin" ]] \
     && [[ -f "${FACT_CHECKPOINT_DIR}/norm_stats_delta.json" ]]; then
    log "FACT checkpoint already present at ${FACT_CHECKPOINT_DIR}; skipping download."
  else
    mkdir -p "${FACT_CHECKPOINT_DIR}"
    log "Downloading Bariona/fact-wam -> ${FACT_CHECKPOINT_DIR}"
    uv run --project "${ROOT_DIR}" --no-sync \
      huggingface-cli download Bariona/fact-wam --local-dir "${FACT_CHECKPOINT_DIR}"
  fi
fi

# ── RoboTwin dataset ──────────────────────────────────────────────────────────
if [[ "${SKIP_DATA_DOWNLOAD:-0}" != "1" ]]; then
  if [[ -d "${DATASETS_DIR}/RoboTwin/Clean" ]] || [[ -d "${DATASETS_DIR}/RoboTwin/Randomized" ]]; then
    log "RoboTwin dataset already present at ${DATASETS_DIR}/RoboTwin; skipping download."
  else
    mkdir -p "${DATASETS_DIR}"
    log "Downloading Bariona/robotwin-v2 -> ${DATASETS_DIR}"
    uv run --project "${ROOT_DIR}" --no-sync \
      huggingface-cli download Bariona/robotwin-v2 robotwin-v2.tar \
      --repo-type dataset --local-dir "${DATASETS_DIR}"
    log "Extracting robotwin-v2.tar ..."
    tar -xf "${DATASETS_DIR}/robotwin-v2.tar" -C "${DATASETS_DIR}"
  fi
fi

log "Done. Run FACT commands with 'uv run --no-sync …'; see README.md."
