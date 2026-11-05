#!/usr/bin/env bash
#
# Single-node orchestrator: launches the three vLLM agent endpoints on
# GPUs 0/1/2 (via serve_agents.sh) and then runs run_pipeline.py on GPU 3.
# When the pipeline exits (success, error, or Ctrl+C), the agent servers
# are torn down via the SIGTERM handler already wired up inside
# serve_agents.sh.
#
# Any extra CLI args to this script are forwarded to run_pipeline.py, including
# its required path flags (see README.md), e.g.:
#   ./run_all.sh --imdbid tt0330373 --verbose --track_file ... --output_dir ...
#
# Overrides:
#   PIPELINE_GPU   GPU index used by run_pipeline.py (default: 3)
#   LOG_DIR        Where serve_agents.sh writes per-service logs.
#   READY_TIMEOUT  Per-endpoint readiness timeout in seconds (default: 1800).

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/.." && pwd)"

PIPELINE_GPU="${PIPELINE_GPU:-3}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
mkdir -p "${LOG_DIR}"

# Ports must match the SERVICES table in serve_agents.sh.
NAME_LINKING_PORT=8000    # vlm_linker_url
FACE_FILTERING_PORT=8001  # base_vlm_url
NAME_FILTERING_PORT=8002  # llm_url

# --- start the three agents in the background -----------------------------
echo "[run_all] starting serve_agents.sh on GPUs 0,1,2 (logs: ${LOG_DIR}/*.log) ..."
LOG_DIR="${LOG_DIR}" bash "${HERE}/serve_agents.sh" &
SERVE_PID=$!

cleanup() {
  trap - INT TERM EXIT
  if kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo
    echo "[run_all] stopping serve_agents.sh (pid ${SERVE_PID}) ..."
    # serve_agents.sh has its own SIGTERM handler that kills each vLLM
    # process group, so a plain TERM to the wrapper shell is enough.
    kill -TERM "${SERVE_PID}" 2>/dev/null || true
    wait "${SERVE_PID}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

# --- wait for each endpoint to become healthy -----------------------------
wait_ready() {
  local port="$1" tag="$2"
  local url="http://127.0.0.1:${port}/v1/models"
  local deadline=$(( SECONDS + READY_TIMEOUT ))
  echo "[run_all] waiting for ${tag} on port ${port} ..."
  while true; do
    if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
      echo "[run_all] serve_agents.sh exited before ${tag} came up." >&2
      echo "[run_all] inspect ${LOG_DIR}/${tag}.log for details." >&2
      exit 1
    fi
    if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      echo "[run_all]   ${tag} ready."
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "[run_all] timed out waiting for ${tag} after ${READY_TIMEOUT}s." >&2
      exit 1
    fi
    sleep 5
  done
}

wait_ready "${NAME_LINKING_PORT}"   "name_linking"
wait_ready "${FACE_FILTERING_PORT}" "face_filtering"
wait_ready "${NAME_FILTERING_PORT}" "name_filtering"

# --- run the pipeline on the remaining GPU --------------------------------
echo "[run_all] launching run_pipeline.py on GPU ${PIPELINE_GPU} ..."
CUDA_VISIBLE_DEVICES="${PIPELINE_GPU}" \
  python "${REPO_ROOT}/run_pipeline.py" \
    --base_vlm_url   "http://127.0.0.1:${FACE_FILTERING_PORT}/v1" \
    --vlm_linker_url "http://127.0.0.1:${NAME_LINKING_PORT}/v1" \
    --llm_url        "http://127.0.0.1:${NAME_FILTERING_PORT}/v1" \
    "$@"

echo "[run_all] pipeline finished."
