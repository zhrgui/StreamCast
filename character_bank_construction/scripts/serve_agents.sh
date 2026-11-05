#!/usr/bin/env bash
#
# Launch the three character-bank vLLM endpoints on a single 3-GPU node.
# Self-contained: no sibling scripts required.
#
# Layout: one row per service in the SERVICES table below
# ("<gpu_index> <tag> <port> <model_id>"). Each service is pinned to its
# GPU via CUDA_VISIBLE_DEVICES, started in its own process group so Ctrl+C
# tears down vLLM + all workers, and its stdout/stderr is redirected to
# ${LOG_DIR}/<tag>.log.
#
# Global overrides (apply to every service unless you edit the table):
#   HOST, MAX_MODEL_LEN, MAX_IMAGES, GPU_UTIL, TENSOR_PARALLEL,
#   HF_TOKEN, EXTRA_VLLM_ARGS, LOG_DIR.

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_IMAGES="${MAX_IMAGES:-512}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${HERE}/logs}"
mkdir -p "${LOG_DIR}"

# Propagate HF token for gated / private repos when given.
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
fi

# Format: "<gpu> <tag> <port> <model_id>"
SERVICES=(
  "0 name_linking   8000 <MODEL_PATH>"
  "1 face_filtering 8001 Qwen/Qwen3-VL-8B-Instruct"
  "2 name_filtering 8002 Qwen/Qwen3.5-9B"
)

# Served-model-name registered with vLLM: HF id unchanged, local path -> basename.
served_name() {
  local mid="$1"
  if [[ "${mid}" == /* ]]; then basename "${mid}"; else echo "${mid}"; fi
}

start_vllm() {
  local gpu="$1" tag="$2" port="$3" model_id="$4"

  if [[ -e "${model_id}" && ! -d "${model_id}" ]]; then
    echo "[ERROR] ${tag}: model path is not a directory: ${model_id}" >&2
    return 1
  fi

  local served log
  served="$(served_name "${model_id}")"
  log="${LOG_DIR}/${tag}.log"

  echo "[serve_all] GPU ${gpu} <- ${tag}  port=${port}  model=${model_id}"
  echo "             log=${log}  served-as=${served}"

  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu}" setsid vllm serve "${model_id}" \
      --host "${HOST}" \
      --port "${port}" \
      --dtype bfloat16 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}}" \
      --gpu-memory-utilization "${GPU_UTIL}" \
      --tensor-parallel-size "${TENSOR_PARALLEL}" \
      --trust-remote-code \
      --served-model-name "${served}" \
      ${EXTRA_VLLM_ARGS:-} >"${log}" 2>&1 &
}

PIDS=()
shutdown() {
  trap - INT TERM
  echo
  echo "[serve_all] received signal, shutting down ${#PIDS[@]} services ..."
  for pid in "${PIDS[@]}"; do
    # Negative pid -> signal the whole process group (vLLM + its workers).
    kill -TERM -"${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done
}
trap shutdown INT TERM

for entry in "${SERVICES[@]}"; do
  read -r GPU TAG PORT MODEL_ID <<<"${entry}"
  start_vllm "${GPU}" "${TAG}" "${PORT}" "${MODEL_ID}"
  PIDS+=("$!")
done

echo "[serve_all] launched PIDs: ${PIDS[*]}"
echo "[serve_all] endpoints:"
for entry in "${SERVICES[@]}"; do
  read -r _ TAG PORT _ <<<"${entry}"
  printf "             %-15s http://%s:%s/v1\n" "${TAG}" "${HOST}" "${PORT}"
done
echo "[serve_all] follow logs:    tail -f ${LOG_DIR}/*.log"

# Wait for all children; propagate the first non-zero exit but still drain
# the rest so logs don't get truncated mid-write.
status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    rc=$?
    [[ ${status} -eq 0 ]] && status=${rc}
  fi
done
exit "${status}"
