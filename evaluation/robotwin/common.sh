# Shared helpers for launch_server.sh / launch_client.sh.
# Expects SCRIPT_DIR, REPO_ROOT, and LAUNCH_CONFIG_PATH to be set by the caller.

require_uv_project() {
  local project_path="$1"
  local project_name="$2"

  if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required to run ${project_name}." >&2
    exit 1
  fi
  if [[ ! -f "${project_path}/pyproject.toml" ]]; then
    echo "Error: ${project_name} uv project not found at '${project_path}'." >&2
    echo "       Set the corresponding *_UV_PROJECT variable to a checkout containing pyproject.toml." >&2
    exit 1
  fi
}

# Export the given launch_config.yml section as shell variables (explicit
# environment variables keep precedence). Parsing always uses FACT's own uv
# project; it never imports RoboTwin-Phys packages into the FACT environment.
load_launch_config() {
  if [[ ! -f "${LAUNCH_CONFIG_PATH}" ]]; then
    return 0
  fi

  local fact_project exports
  fact_project="${FACT_UV_PROJECT:-${REPO_ROOT}}"
  require_uv_project "${fact_project}" "FACT"

  # `if !` so `set -e` cannot kill the shell before the error prints.
  if ! exports=$(uv run --project "${fact_project}" --no-sync python \
      "${SCRIPT_DIR}/resolve_launch_config.py" \
      --config "${LAUNCH_CONFIG_PATH}" --section "$1"); then
    echo "Error: could not read ${LAUNCH_CONFIG_PATH} using FACT's uv environment." >&2
    echo "       Run 'uv sync --locked' in ${fact_project} first." >&2
    exit 1
  fi

  eval "${exports}"
}
