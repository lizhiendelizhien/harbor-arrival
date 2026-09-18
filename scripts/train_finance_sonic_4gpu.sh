#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for the finance Sonic trainer.  NUM_ENVS is deliberately
# per rank, matching the table-tennis launchers: the effective global batch is
# NUM_ENVS * NUM_GPUS.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
ORIGINAL_ARGS=("$@")
echo "[finance-launcher] Preparing Finance Sonic launcher (set PYTHON_BIN to skip environment probing)" >&2
LOGGER="${LOGGER:-tensorboard}"

# The simulator environment often contains torch but not the small finance
# extras.  Reuse a project-local dependency bundle (or the shared smoke bundle
# when present) without changing the selected interpreter.  An explicit
# FINANCE_PYTHONPATH/LOCAL_PYTHON_DEPS takes precedence over auto-discovery.
LOCAL_PYTHON_DEPS="${LOCAL_PYTHON_DEPS:-}"
FINANCE_PYTHONPATH="${FINANCE_PYTHONPATH:-${LOCAL_PYTHON_DEPS}}"
if [[ -z "${FINANCE_PYTHONPATH}" ]]; then
  for dependency_path in "${REPO_ROOT}/.python_deps" "/tmp/finance-sonic-deps"; do
    if [[ -d "${dependency_path}" ]]; then
      FINANCE_PYTHONPATH="${FINANCE_PYTHONPATH:+${FINANCE_PYTHONPATH}:}${dependency_path}"
    fi
  done
fi
if [[ -n "${FINANCE_PYTHONPATH}" ]]; then
  case ":${PYTHONPATH:-}:" in
    *":${FINANCE_PYTHONPATH}:"*) ;;
    *)
      PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${FINANCE_PYTHONPATH}"
      export PYTHONPATH
      echo "[finance-launcher] Using finance dependency path: ${FINANCE_PYTHONPATH}" >&2
      ;;
  esac
fi

die() {
  echo "[finance-launcher] $*" >&2
  exit 2
}

python_missing_modules() {
  "${1}" -c \
    'import importlib.util; names=("torch", "numpy", "omegaconf", "tensordict", "vector_quantize_pytorch"); print(" ".join(name for name in names if importlib.util.find_spec(name) is None))' \
    2>/dev/null
}

python_has_finance_dependencies() {
  local missing
  missing="$(python_missing_modules "${1}")" || return 1
  [[ -z "${missing}" ]] || return 1
  [[ "${LOGGER}" != "tensorboard" ]] || python_has_tensorboard "${1}"
}

python_has_tensorboard() {
  "${1}" -c 'from torch.utils.tensorboard import SummaryWriter' >/dev/null 2>&1
}

positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

# Prefer an explicitly selected interpreter.  Otherwise probe common project
# environments and keep the first one with the complete finance dependency set.
# A system Python may exist while being unrelated to the environment used for
# training.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_CANDIDATES=()
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_CANDIDATES+=("${VIRTUAL_ENV}/bin/python")
  fi
  PYTHON_CANDIDATES+=(
    "${REPO_ROOT}/.venv/bin/python"
    "${REPO_ROOT}/.venv_sim/bin/python"
    "${REPO_ROOT}/../AeroStep_Sonic/.venv_sim/bin/python"
    "${REPO_ROOT}/../Isaac-GR00T/.venv/bin/python"
  )
  if command -v python >/dev/null 2>&1; then
    PYTHON_CANDIDATES+=("$(command -v python)")
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CANDIDATES+=("$(command -v python3)")
  fi

  PYTHON_BIN=""
  PYTHON_TORCH_BIN=""
  for candidate in "${PYTHON_CANDIDATES[@]}"; do
    [[ -x "${candidate}" ]] || continue
    if "${candidate}" -c 'import torch' >/dev/null 2>&1; then
      [[ -n "${PYTHON_TORCH_BIN}" ]] || PYTHON_TORCH_BIN="${candidate}"
      if python_has_finance_dependencies "${candidate}"; then
        PYTHON_BIN="${candidate}"
        break
      fi
    fi
  done
  # If no candidate has the optional finance packages yet, retain a
  # torch-capable interpreter so the preflight below can name what is missing.
  if [[ -z "${PYTHON_BIN}" && -n "${PYTHON_TORCH_BIN}" ]]; then
    PYTHON_BIN="${PYTHON_TORCH_BIN}"
  fi
  # Preserve the usual error path below when no usable interpreter exists.
  if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v python3)"
    else
      PYTHON_BIN="python3"
    fi
  fi
fi
TASK="${TASK:-finance_sonic_archive}"
DEVICE="${DEVICE:-cuda}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100000}"
SEED="${SEED:-42}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
THREADS="${THREADS:-1}"
if [[ -z "${SEQUENCE_LENGTH_EXPLICIT:-}" ]]; then
  SEQUENCE_LENGTH_EXPLICIT=0
  [[ -n "${SEQUENCE_LENGTH:-}" ]] && SEQUENCE_LENGTH_EXPLICIT=1
fi
if [[ -z "${ROLLOUT_STEPS_EXPLICIT:-}" ]]; then
  ROLLOUT_STEPS_EXPLICIT=0
  [[ -n "${ROLLOUT_STEPS:-}" ]] && ROLLOUT_STEPS_EXPLICIT=1
fi
if [[ -z "${EPOCHS_EXPLICIT:-}" ]]; then
  EPOCHS_EXPLICIT=0
  [[ -n "${EPOCHS:-}" ]] && EPOCHS_EXPLICIT=1
fi
if [[ -z "${NUM_MINIBATCHES_EXPLICIT:-}" ]]; then
  NUM_MINIBATCHES_EXPLICIT=0
  [[ -n "${NUM_MINIBATCHES:-}" ]] && NUM_MINIBATCHES_EXPLICIT=1
fi
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-64}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
EPOCHS="${EPOCHS:-5}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"

# The archive root is the directory containing the three companion files and
# the trajectories/ directory.  Accepting the latter as input is convenient
# for users who copied the path directly from a data listing.
if [[ -z "${TRAJECTORY_INDEX_EXPLICIT:-}" ]]; then
  TRAJECTORY_INDEX_EXPLICIT=0
  [[ -n "${TRAJECTORY_INDEX:-}" ]] && TRAJECTORY_INDEX_EXPLICIT=1
fi
if [[ -z "${TRAJECTORY_METADATA_EXPLICIT:-}" ]]; then
  TRAJECTORY_METADATA_EXPLICIT=0
  [[ -n "${TRAJECTORY_METADATA:-}" ]] && TRAJECTORY_METADATA_EXPLICIT=1
fi
if [[ -z "${TRAJECTORY_MANIFEST_EXPLICIT:-}" ]]; then
  TRAJECTORY_MANIFEST_EXPLICIT=0
  [[ -n "${TRAJECTORY_MANIFEST:-}" ]] && TRAJECTORY_MANIFEST_EXPLICIT=1
fi
if [[ -z "${TRAJECTORY_ROOT_EXPLICIT:-}" ]]; then
  TRAJECTORY_ROOT_EXPLICIT=0
  [[ -n "${TRAJECTORY_ROOT:-}" ]] && TRAJECTORY_ROOT_EXPLICIT=1
fi
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-${REPO_ROOT}/data/us_socket/月线轨迹重构}"
if [[ "$(basename -- "${TRAJECTORY_ROOT}")" == "trajectories" \
      && ! -e "${TRAJECTORY_ROOT}/trajectory_index.csv" ]]; then
  TRAJECTORY_ROOT="$(dirname -- "${TRAJECTORY_ROOT}")"
fi
TRAJECTORY_INDEX="${TRAJECTORY_INDEX:-${TRAJECTORY_ROOT}/trajectory_index.csv}"
TRAJECTORY_METADATA="${TRAJECTORY_METADATA:-${TRAJECTORY_ROOT}/metadata.pkl}"
TRAJECTORY_MANIFEST="${TRAJECTORY_MANIFEST:-${TRAJECTORY_ROOT}/manifest.json}"
CONTEXT_CANONICAL="${CONTEXT_CANONICAL:-${FINANCE_CONTEXT_CANONICAL:-${CANONICAL:-}}}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
TORCHRUN_STANDALONE="${TORCHRUN_STANDALONE:-0}"
FINANCE_MULTI_GPU_LAUNCHER="${FINANCE_MULTI_GPU_LAUNCHER:-${TABLE_TENNIS_MULTI_GPU_LAUNCHER:-torchrun}}"
FINANCE_DRY_RUN="${FINANCE_DRY_RUN:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
NCCL_ALGO="${NCCL_ALGO:-Ring}"
NCCL_PROTO="${NCCL_PROTO:-Simple}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Keep the default visible-device list useful on a four-GPU host while still
# allowing CPU smoke tests to set CUDA_VISIBLE_DEVICES to the empty string.
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  NUM_GPUS="$1"
  shift
fi

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  CUDA_VISIBLE_DEVICES="0,1,2,3"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS}}"

RUN_NAME="${RUN_NAME:-${TASK}_${NUM_GPUS}gpu_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/finance_train_logs/${RUN_NAME}}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-/tmp/finance_launch_logs}"
LOG_FILE="${LOG_FILE:-${LAUNCH_LOG_DIR}/${RUN_NAME}.log}"
AUTO_TMUX="${AUTO_TMUX:-0}"
TMUX_ATTACH="${TMUX_ATTACH:-0}"
TMUX_SESSION="${TMUX_SESSION:-${TASK}_${NUM_GPUS}gpu}"
TMUX_ENV_FILE="${TMUX_ENV_FILE:-/tmp/finance_${TMUX_SESSION}.env}"
TMUX_RUN_FILE="${TMUX_RUN_FILE:-/tmp/finance_${TMUX_SESSION}.sh}"
TMUX_LOG_FILE="${TMUX_LOG_FILE:-/tmp/finance_${TMUX_SESSION}.log}"

positive_integer "${NUM_GPUS}" || die "NUM_GPUS must be a positive integer: ${NUM_GPUS}"
positive_integer "${NUM_ENVS}" || die "NUM_ENVS must be a positive integer: ${NUM_ENVS}"
positive_integer "${MAX_ITERATIONS}" || die "MAX_ITERATIONS must be a positive integer: ${MAX_ITERATIONS}"
positive_integer "${SAVE_INTERVAL}" || die "SAVE_INTERVAL must be a positive integer: ${SAVE_INTERVAL}"
positive_integer "${THREADS}" || die "THREADS must be a positive integer: ${THREADS}"
positive_integer "${SEQUENCE_LENGTH}" || die "SEQUENCE_LENGTH must be a positive integer: ${SEQUENCE_LENGTH}"
positive_integer "${ROLLOUT_STEPS}" || die "ROLLOUT_STEPS must be a positive integer: ${ROLLOUT_STEPS}"
positive_integer "${EPOCHS}" || die "EPOCHS must be a positive integer: ${EPOCHS}"
positive_integer "${NUM_MINIBATCHES}" || die "NUM_MINIBATCHES must be a positive integer: ${NUM_MINIBATCHES}"
positive_integer "${NPROC_PER_NODE}" || die "NPROC_PER_NODE must be a positive integer: ${NPROC_PER_NODE}"
case "${LOGGER}" in
  tensorboard|none) ;;
  *) die "LOGGER must be tensorboard or none: ${LOGGER}" ;;
esac
if [[ "${NPROC_PER_NODE}" != "${NUM_GPUS}" ]]; then
  die "NPROC_PER_NODE=${NPROC_PER_NODE} must match NUM_GPUS=${NUM_GPUS}"
fi
nonnegative_integer "${SEED}" || die "SEED must be a non-negative integer: ${SEED}"
positive_integer "${MASTER_PORT}" || die "MASTER_PORT must be a positive integer: ${MASTER_PORT}"
if (( MASTER_PORT > 65535 )); then
  die "MASTER_PORT must be <= 65535: ${MASTER_PORT}"
fi
if [[ "${TORCHRUN_STANDALONE}" != "0" && "${TORCHRUN_STANDALONE}" != "1" ]]; then
  die "TORCHRUN_STANDALONE must be 0 or 1: ${TORCHRUN_STANDALONE}"
fi
if [[ "${FINANCE_DRY_RUN}" != "0" && "${FINANCE_DRY_RUN}" != "1" ]]; then
  die "FINANCE_DRY_RUN must be 0 or 1: ${FINANCE_DRY_RUN}"
fi
if [[ "${AUTO_TMUX}" != "0" && "${AUTO_TMUX}" != "1" ]]; then
  die "AUTO_TMUX must be 0 or 1: ${AUTO_TMUX}"
fi
if [[ "${TMUX_ATTACH}" != "0" && "${TMUX_ATTACH}" != "1" ]]; then
  die "TMUX_ATTACH must be 0 or 1: ${TMUX_ATTACH}"
fi
if [[ "${FINANCE_MULTI_GPU_LAUNCHER}" != "torchrun" ]]; then
  die "FINANCE_MULTI_GPU_LAUNCHER must be 'torchrun' for true DDP, got '${FINANCE_MULTI_GPU_LAUNCHER}'"
fi

# Normalize the checkpoint spellings used by older launch scripts.  The Python
# trainer intentionally has one unambiguous --resume PATH interface.
RESUME_PATH=""
RESUME_SPECIFIED=0
SHOW_HELP=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  arg="$1"
  case "${arg}" in
    --resume)
      shift
      if [[ $# -eq 0 ]]; then
        die "--resume requires a checkpoint path (or --checkpoint_path PATH)"
      fi
      if [[ "$1" == "--checkpoint_path" || "$1" == "--checkpoint-path" ]]; then
        shift
        [[ $# -gt 0 && "$1" != --* ]] || die "${arg} --checkpoint_path requires a checkpoint path"
      elif [[ "$1" == --checkpoint_path=* || "$1" == --checkpoint-path=* ]]; then
        candidate="${1#*=}"
        [[ -n "${candidate}" ]] || die "${1%%=*} requires a checkpoint path"
        shift
        if (( RESUME_SPECIFIED )) && [[ "${RESUME_PATH}" != "${candidate}" ]]; then
          die "Conflicting resume checkpoints: ${RESUME_PATH} and ${candidate}"
        fi
        RESUME_PATH="${candidate}"
        RESUME_SPECIFIED=1
        continue
      elif [[ "$1" == --* ]]; then
        die "--resume requires a checkpoint path; got '$1'"
      fi
      candidate="$1"
      shift
      if (( RESUME_SPECIFIED )) && [[ "${RESUME_PATH}" != "${candidate}" ]]; then
        die "Conflicting resume checkpoints: ${RESUME_PATH} and ${candidate}"
      fi
      RESUME_PATH="${candidate}"
      RESUME_SPECIFIED=1
      ;;
    --resume=*)
      candidate="${arg#--resume=}"
      [[ -n "${candidate}" ]] || die "--resume= requires a checkpoint path"
      if (( RESUME_SPECIFIED )) && [[ "${RESUME_PATH}" != "${candidate}" ]]; then
        die "Conflicting resume checkpoints: ${RESUME_PATH} and ${candidate}"
      fi
      RESUME_PATH="${candidate}"
      RESUME_SPECIFIED=1
      shift
      ;;
    --checkpoint_path|--checkpoint-path)
      flag="${arg}"
      shift
      [[ $# -gt 0 && "$1" != --* ]] || die "${flag} requires a checkpoint path"
      candidate="$1"
      shift
      if (( RESUME_SPECIFIED )) && [[ "${RESUME_PATH}" != "${candidate}" ]]; then
        die "Conflicting resume checkpoints: ${RESUME_PATH} and ${candidate}"
      fi
      RESUME_PATH="${candidate}"
      RESUME_SPECIFIED=1
      ;;
    --checkpoint_path=*|--checkpoint-path=*)
      candidate="${arg#*=}"
      [[ -n "${candidate}" ]] || die "${arg%%=*} requires a checkpoint path"
      if (( RESUME_SPECIFIED )) && [[ "${RESUME_PATH}" != "${candidate}" ]]; then
        die "Conflicting resume checkpoints: ${RESUME_PATH} and ${candidate}"
      fi
      RESUME_PATH="${candidate}"
      RESUME_SPECIFIED=1
      shift
      ;;
    --trajectory-root)
      shift
      [[ $# -gt 0 && "$1" != --* ]] || die "--trajectory-root requires an archive root"
      TRAJECTORY_ROOT="$1"
      TRAJECTORY_ROOT_EXPLICIT=1
      shift
      if [[ "$(basename -- "${TRAJECTORY_ROOT}")" == "trajectories" \
            && ! -e "${TRAJECTORY_ROOT}/trajectory_index.csv" ]]; then
        TRAJECTORY_ROOT="$(dirname -- "${TRAJECTORY_ROOT}")"
      fi
      if (( ! TRAJECTORY_INDEX_EXPLICIT )); then
        TRAJECTORY_INDEX="${TRAJECTORY_ROOT}/trajectory_index.csv"
      fi
      if (( ! TRAJECTORY_METADATA_EXPLICIT )); then
        TRAJECTORY_METADATA="${TRAJECTORY_ROOT}/metadata.pkl"
      fi
      if (( ! TRAJECTORY_MANIFEST_EXPLICIT )); then
        TRAJECTORY_MANIFEST="${TRAJECTORY_ROOT}/manifest.json"
      fi
      ;;
    --trajectory-root=*)
      TRAJECTORY_ROOT="${arg#--trajectory-root=}"
      [[ -n "${TRAJECTORY_ROOT}" ]] || die "--trajectory-root= requires an archive root"
      TRAJECTORY_ROOT_EXPLICIT=1
      shift
      if [[ "$(basename -- "${TRAJECTORY_ROOT}")" == "trajectories" \
            && ! -e "${TRAJECTORY_ROOT}/trajectory_index.csv" ]]; then
        TRAJECTORY_ROOT="$(dirname -- "${TRAJECTORY_ROOT}")"
      fi
      if (( ! TRAJECTORY_INDEX_EXPLICIT )); then
        TRAJECTORY_INDEX="${TRAJECTORY_ROOT}/trajectory_index.csv"
      fi
      if (( ! TRAJECTORY_METADATA_EXPLICIT )); then
        TRAJECTORY_METADATA="${TRAJECTORY_ROOT}/metadata.pkl"
      fi
      if (( ! TRAJECTORY_MANIFEST_EXPLICIT )); then
        TRAJECTORY_MANIFEST="${TRAJECTORY_ROOT}/manifest.json"
      fi
      ;;
    --trajectory-index|--trajectory-metadata|--trajectory-manifest)
      flag="${arg}"
      shift
      [[ $# -gt 0 && "$1" != --* ]] || die "${flag} requires a file path"
      case "${flag}" in
        --trajectory-index)
          TRAJECTORY_INDEX="$1"
          TRAJECTORY_INDEX_EXPLICIT=1
          ;;
        --trajectory-metadata)
          TRAJECTORY_METADATA="$1"
          TRAJECTORY_METADATA_EXPLICIT=1
          ;;
        --trajectory-manifest)
          TRAJECTORY_MANIFEST="$1"
          TRAJECTORY_MANIFEST_EXPLICIT=1
          ;;
      esac
      shift
      ;;
    --trajectory-index=*|--trajectory-metadata=*|--trajectory-manifest=*)
      value="${arg#*=}"
      [[ -n "${value}" ]] || die "${arg%%=*} requires a file path"
      case "${arg%%=*}" in
        --trajectory-index)
          TRAJECTORY_INDEX="${value}"
          TRAJECTORY_INDEX_EXPLICIT=1
          ;;
        --trajectory-metadata)
          TRAJECTORY_METADATA="${value}"
          TRAJECTORY_METADATA_EXPLICIT=1
          ;;
        --trajectory-manifest)
          TRAJECTORY_MANIFEST="${value}"
          TRAJECTORY_MANIFEST_EXPLICIT=1
          ;;
      esac
      shift
      ;;
    --device)
      shift
      [[ $# -gt 0 && "$1" != --* ]] || die "--device requires a value"
      DEVICE="$1"
      shift
      ;;
    --device=*)
      DEVICE="${arg#--device=}"
      [[ -n "${DEVICE}" ]] || die "--device= requires a value"
      shift
      ;;
    --output-dir)
      shift
      [[ $# -gt 0 && "$1" != --* ]] || die "--output-dir requires a directory"
      OUTPUT_DIR="$1"
      shift
      ;;
    --output-dir=*)
      OUTPUT_DIR="${arg#--output-dir=}"
      [[ -n "${OUTPUT_DIR}" ]] || die "--output-dir= requires a directory"
      shift
      ;;
    --dry-run|--print-only)
      FINANCE_DRY_RUN=1
      shift
      ;;
    --help|-h)
      SHOW_HELP=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("${arg}")
      shift
      ;;
  esac
done

# A checkpoint carries the authoritative source recipe.  Do not pass the
# launcher's default archive to a canonical-source checkpoint; doing so would
# make the Python resolver treat unrelated archive arguments as a conflict.
# An explicit root or companion override remains available for archive
# relocation and is intentionally passed through on resume.
USE_ARCHIVE_SOURCE=1
if (( RESUME_SPECIFIED && !TRAJECTORY_ROOT_EXPLICIT && !TRAJECTORY_INDEX_EXPLICIT \
      && !TRAJECTORY_METADATA_EXPLICIT && !TRAJECTORY_MANIFEST_EXPLICIT )); then
  USE_ARCHIVE_SOURCE=0
fi

if [[ -n "${CURRICULUM_STAGES:-}" ]]; then
  echo "[finance-launcher] CURRICULUM_STAGES=${CURRICULUM_STAGES} is ignored; the finance environment has no curriculum stages" >&2
fi

if (( SHOW_HELP )); then
  cat <<'USAGE'
Usage: scripts/train_finance_sonic_4gpu.sh [NUM_GPUS] [options]

Launches scripts.train_finance_sonic with torchrun.  NUM_ENVS is per rank.
The default data source is data/us_socket/月线轨迹重构, whose companion files
are trajectory_index.csv, metadata.pkl, and manifest.json.

Common environment variables:
  NUM_GPUS, NUM_ENVS, MAX_ITERATIONS, SEED, DEVICE, PYTHON_BIN, LOGGER
  FINANCE_PYTHONPATH, LOCAL_PYTHON_DEPS
  TRAJECTORY_ROOT, TRAJECTORY_INDEX, TRAJECTORY_METADATA, TRAJECTORY_MANIFEST
  OUTPUT_DIR, MASTER_ADDR, MASTER_PORT, TORCHRUN_STANDALONE, NPROC_PER_NODE
  NCCL_DEBUG, NCCL_P2P_DISABLE, NCCL_SHM_DISABLE, NCCL_IB_DISABLE
  NCCL_COLLNET_ENABLE, NCCL_CUMEM_ENABLE, NCCL_NVLS_ENABLE, NCCL_ALGO, NCCL_PROTO
  TORCH_NCCL_ASYNC_ERROR_HANDLING, PYTORCH_CUDA_ALLOC_CONF
  AUTO_TMUX, TMUX_SESSION, TMUX_ATTACH, TMUX_ENV_FILE, TMUX_RUN_FILE, TMUX_LOG_FILE

Resume forms accepted:
  --resume CHECKPOINT
  --resume --checkpoint_path CHECKPOINT
  --checkpoint_path CHECKPOINT

Set FINANCE_DRY_RUN=1 or pass --dry-run to print the resolved command.
NPROC_PER_NODE must match NUM_GPUS; NUM_ENVS remains the per-rank environment
count. NCCL and thread settings are exported to every torchrun worker and
printed in the launch log. Set AUTO_TMUX=1 to detach the same finance command
into a tmux session; TMUX_ATTACH=1 attaches immediately. The wrapper preserves
the archive/resume arguments and keeps a failed session open for inspection.
When PYTHON_BIN is omitted, the launcher probes common project virtual
environments and selects the first interpreter with the complete finance
dependency set.  If only a torch-capable interpreter is found, the preflight
reports the missing packages.  Install them with `$PYTHON_BIN -m pip install -e '.[finance]'`.
If `FINANCE_PYTHONPATH` or `LOCAL_PYTHON_DEPS` is set, that dependency bundle
is appended to `PYTHONPATH`; otherwise existing `.python_deps` and
`/tmp/finance-sonic-deps` bundles are used automatically when present.
LOGGER=tensorboard validates a real torch.utils.tensorboard import before
torchrun; use LOGGER=none to run without the TensorBoard dependency.
On resume, archive arguments are omitted unless TRAJECTORY_ROOT or a companion
file is explicitly set, so the checkpoint's saved source recipe is authoritative.
USAGE
  exit 0
fi

# Keep the long-running launcher behavior close to the table-tennis entrypoint
# without carrying over Isaac/JAX-specific rank isolation.  The wrapper stores
# the resolved finance invocation, so a detached run is reproducible and does
# not depend on the caller's shell after this process exits.
if [[ "${AUTO_TMUX}" == "1" && -z "${FINANCE_IN_TMUX:-}" && -z "${TMUX:-}" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[finance-launcher] AUTO_TMUX=1 but tmux is not installed; continuing in the current shell" >&2
    AUTO_TMUX=0
  elif tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "[finance-launcher] tmux session already exists: ${TMUX_SESSION}"
    echo "  Attach: tmux attach -t ${TMUX_SESSION}"
    exit 0
  else
    mkdir -p "$(dirname -- "${TMUX_ENV_FILE}")" \
      "$(dirname -- "${TMUX_RUN_FILE}")" \
      "$(dirname -- "${TMUX_LOG_FILE}")"
    : > "${TMUX_ENV_FILE}"
    : > "${TMUX_LOG_FILE}"

    # Use shell assignments instead of declare -p for scalars so the snapshot
    # is portable and easy to inspect.  ORIGINAL_ARGS is an array and is kept
    # with declare -p to preserve argument boundaries and spaces.
    TMUX_ENV_VARS=(
      PATH PYTHONPATH FINANCE_PYTHONPATH LOCAL_PYTHON_DEPS VIRTUAL_ENV CONDA_PREFIX PYTHON_BIN
      TASK DEVICE LOGGER NUM_GPUS NPROC_PER_NODE NUM_ENVS MAX_ITERATIONS SEED SAVE_INTERVAL THREADS
      SEQUENCE_LENGTH_EXPLICIT ROLLOUT_STEPS_EXPLICIT EPOCHS_EXPLICIT NUM_MINIBATCHES_EXPLICIT
      SEQUENCE_LENGTH ROLLOUT_STEPS EPOCHS NUM_MINIBATCHES
      TRAJECTORY_ROOT_EXPLICIT TRAJECTORY_INDEX_EXPLICIT TRAJECTORY_METADATA_EXPLICIT TRAJECTORY_MANIFEST_EXPLICIT
      TRAJECTORY_ROOT TRAJECTORY_INDEX TRAJECTORY_METADATA TRAJECTORY_MANIFEST CONTEXT_CANONICAL
      NORMALIZATION REFERENCE_CONFIG START END START_PERIOD END_PERIOD SYMBOLS TINY CANONICAL
      FINANCE_CONTEXT_CANONICAL
      MASTER_ADDR MASTER_PORT TORCHRUN_STANDALONE FINANCE_MULTI_GPU_LAUNCHER FINANCE_DRY_RUN
      OMP_NUM_THREADS MKL_NUM_THREADS CUDA_VISIBLE_DEVICES
      NCCL_DEBUG NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_DISABLE NCCL_COLLNET_ENABLE
      NCCL_CUMEM_ENABLE NCCL_NVLS_ENABLE NCCL_ALGO NCCL_PROTO TORCH_NCCL_ASYNC_ERROR_HANDLING
      PYTORCH_CUDA_ALLOC_CONF
      OUTPUT_DIR RUN_NAME LAUNCH_LOG_DIR LOG_FILE
      AUTO_TMUX TMUX_ATTACH TMUX_SESSION TMUX_ENV_FILE TMUX_RUN_FILE TMUX_LOG_FILE
    )
    for var in "${TMUX_ENV_VARS[@]}"; do
      if [[ -v "${var}" ]]; then
        printf '%s=%q\n' "${var}" "${!var}" >> "${TMUX_ENV_FILE}"
      fi
    done
    declare -p ORIGINAL_ARGS >> "${TMUX_ENV_FILE}"

    printf -v _finance_repo_q '%q' "${REPO_ROOT}"
    printf -v _finance_launcher_q '%q' "${REPO_ROOT}/scripts/train_finance_sonic_4gpu.sh"
    printf -v _finance_env_q '%q' "${TMUX_ENV_FILE}"
    printf -v _finance_log_q '%q' "${TMUX_LOG_FILE}"
    printf -v _finance_args_comment '%q ' "${ORIGINAL_ARGS[@]}"
    cat > "${TMUX_RUN_FILE}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${_finance_repo_q}
source ${_finance_env_q}
# Original launcher arguments (also preserved as ORIGINAL_ARGS in the env snapshot):
# ${_finance_args_comment}
export PATH PYTHONPATH FINANCE_PYTHONPATH LOCAL_PYTHON_DEPS VIRTUAL_ENV CONDA_PREFIX PYTHON_BIN
export TASK DEVICE LOGGER NUM_GPUS NPROC_PER_NODE NUM_ENVS MAX_ITERATIONS SEED SAVE_INTERVAL THREADS
export SEQUENCE_LENGTH_EXPLICIT ROLLOUT_STEPS_EXPLICIT EPOCHS_EXPLICIT NUM_MINIBATCHES_EXPLICIT
export SEQUENCE_LENGTH ROLLOUT_STEPS EPOCHS NUM_MINIBATCHES
export TRAJECTORY_ROOT_EXPLICIT TRAJECTORY_INDEX_EXPLICIT TRAJECTORY_METADATA_EXPLICIT TRAJECTORY_MANIFEST_EXPLICIT
export TRAJECTORY_ROOT TRAJECTORY_INDEX TRAJECTORY_METADATA TRAJECTORY_MANIFEST CONTEXT_CANONICAL
export NORMALIZATION REFERENCE_CONFIG START END START_PERIOD END_PERIOD SYMBOLS TINY CANONICAL FINANCE_CONTEXT_CANONICAL
export MASTER_ADDR MASTER_PORT TORCHRUN_STANDALONE FINANCE_MULTI_GPU_LAUNCHER FINANCE_DRY_RUN
export OMP_NUM_THREADS MKL_NUM_THREADS CUDA_VISIBLE_DEVICES
export NCCL_DEBUG NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_DISABLE NCCL_COLLNET_ENABLE \\
  NCCL_CUMEM_ENABLE NCCL_NVLS_ENABLE NCCL_ALGO NCCL_PROTO TORCH_NCCL_ASYNC_ERROR_HANDLING \\
  PYTORCH_CUDA_ALLOC_CONF
export OUTPUT_DIR RUN_NAME LAUNCH_LOG_DIR LOG_FILE
export AUTO_TMUX=0
export FINANCE_IN_TMUX=1
set +e
bash ${_finance_launcher_q} "\${ORIGINAL_ARGS[@]}" 2>&1 | tee -a ${_finance_log_q}
status=\${PIPESTATUS[0]}
set -e
if [[ "\${status}" -ne 0 ]]; then
  echo "[finance-launcher] training exited with status \${status}"
  echo "[finance-launcher] log: ${TMUX_LOG_FILE}"
  echo "[finance-launcher] keeping tmux session open for inspection"
  exec bash
fi
exit 0
EOF
    chmod +x "${TMUX_RUN_FILE}"
    tmux new-session -d -s "${TMUX_SESSION}" "${TMUX_RUN_FILE}"
    echo "[finance-launcher] Started Finance Sonic DDP training in tmux session: ${TMUX_SESSION}"
    echo "  Attach: tmux attach -t ${TMUX_SESSION}"
    echo "  Log: ${TMUX_LOG_FILE}"
    echo "  Kill: tmux kill-session -t ${TMUX_SESSION}"
    if [[ "${TMUX_ATTACH}" == "1" ]]; then
      exec tmux attach -t "${TMUX_SESSION}"
    fi
    exit 0
  fi
fi

EFFECTIVE_DEVICE="${DEVICE}"
if [[ "${FINANCE_DRY_RUN}" == "0" && "${EFFECTIVE_DEVICE}" != "cpu" \
      && "${EFFECTIVE_DEVICE}" != "CPU" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    die "CUDA_VISIBLE_DEVICES is empty while DEVICE=${EFFECTIVE_DEVICE}; use DEVICE=cpu for a CPU run"
  fi
  IFS=',' read -r -a _visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
  if (( ${#_visible_devices[@]} < NUM_GPUS )); then
    die "Requested NUM_GPUS=${NUM_GPUS}, but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} has only ${#_visible_devices[@]} entries"
  fi
fi

if (( RESUME_SPECIFIED )) && [[ "${FINANCE_DRY_RUN}" == "0" && ! -f "${RESUME_PATH}" ]]; then
  die "Resume checkpoint does not exist: ${RESUME_PATH}"
fi

if [[ "${PYTHON_BIN}" == */* && ! -x "${PYTHON_BIN}" && "${FINANCE_DRY_RUN}" == "0" ]]; then
  die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
fi
if [[ "${FINANCE_DRY_RUN}" == "0" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  die "Python executable not found: ${PYTHON_BIN}"
fi
if [[ "${FINANCE_DRY_RUN}" == "0" ]]; then
  missing_modules="$(python_missing_modules "${PYTHON_BIN}")" \
    || die "Unable to execute Python interpreter: ${PYTHON_BIN}"
  if [[ -n "${missing_modules}" ]]; then
    install_hint="${PYTHON_BIN} -m pip install -e '.[finance]'"
    die "Python ${PYTHON_BIN} is missing finance dependencies: ${missing_modules}. Use a torch-capable environment and install them with ${install_hint}, or set PYTHON_BIN/PYTHONPATH."
  fi
  if [[ "${LOGGER}" == "tensorboard" ]] && ! python_has_tensorboard "${PYTHON_BIN}"; then
    install_hint="${PYTHON_BIN} -m pip install -e '.[finance]'"
    die "LOGGER=tensorboard requires torch.utils.tensorboard in ${PYTHON_BIN}. Install it with ${install_hint}, or set LOGGER=none."
  fi
fi
if (( USE_ARCHIVE_SOURCE )) && [[ "${FINANCE_DRY_RUN}" == "0" ]]; then
  for archive_file in "${TRAJECTORY_INDEX}" "${TRAJECTORY_METADATA}" "${TRAJECTORY_MANIFEST}"; do
    [[ -r "${archive_file}" ]] || die "Trajectory archive companion is not readable: ${archive_file}"
  done
fi

TRAIN_COMMAND=("${PYTHON_BIN}" -m torch.distributed.run)
if [[ "${TORCHRUN_STANDALONE}" == "1" ]]; then
  TRAIN_COMMAND+=(--standalone)
else
  TRAIN_COMMAND+=(--rdzv-backend=static "--master-addr=${MASTER_ADDR}" "--master-port=${MASTER_PORT}")
fi
TRAIN_COMMAND+=(--nproc_per_node="${NPROC_PER_NODE}" -m scripts.train_finance_sonic
  --distributed)
if (( USE_ARCHIVE_SOURCE )); then
  TRAIN_COMMAND+=(--trajectory-root "${TRAJECTORY_ROOT}"
    --trajectory-index "${TRAJECTORY_INDEX}"
    --trajectory-metadata "${TRAJECTORY_METADATA}"
    --trajectory-manifest "${TRAJECTORY_MANIFEST}")
fi
TRAIN_COMMAND+=(--output-dir "${OUTPUT_DIR}"
  --iterations "${MAX_ITERATIONS}"
  --num-envs "${NUM_ENVS}"
  --threads "${THREADS}"
  --seed "${SEED}"
  --save-interval "${SAVE_INTERVAL}"
  --device "${DEVICE}"
  --logger "${LOGGER}")

# On resume, these values are part of the checkpoint contract.  Omitting an
# unspecified override lets the Python entry point inherit them safely.
if (( ! RESUME_SPECIFIED || SEQUENCE_LENGTH_EXPLICIT )); then
  TRAIN_COMMAND+=(--sequence-length "${SEQUENCE_LENGTH}")
fi
if (( ! RESUME_SPECIFIED || ROLLOUT_STEPS_EXPLICIT )); then
  TRAIN_COMMAND+=(--rollout-steps "${ROLLOUT_STEPS}")
fi
if (( ! RESUME_SPECIFIED || EPOCHS_EXPLICIT )); then
  TRAIN_COMMAND+=(--epochs "${EPOCHS}")
fi
if (( ! RESUME_SPECIFIED || NUM_MINIBATCHES_EXPLICIT )); then
  TRAIN_COMMAND+=(--num-minibatches "${NUM_MINIBATCHES}")
fi

if [[ -n "${CONTEXT_CANONICAL}" ]]; then
  TRAIN_COMMAND+=(--canonical "${CONTEXT_CANONICAL}")
fi
if [[ -n "${NORMALIZATION:-}" ]]; then
  TRAIN_COMMAND+=(--normalization "${NORMALIZATION}")
fi
if [[ -n "${REFERENCE_CONFIG:-}" ]]; then
  TRAIN_COMMAND+=(--reference-config "${REFERENCE_CONFIG}")
fi
START_VALUE="${START:-${START_PERIOD:-}}"
END_VALUE="${END:-${END_PERIOD:-}}"
if [[ -n "${START_VALUE}" ]]; then
  TRAIN_COMMAND+=(--start "${START_VALUE}")
fi
if [[ -n "${END_VALUE}" ]]; then
  TRAIN_COMMAND+=(--end "${END_VALUE}")
fi
if [[ -n "${SYMBOLS:-}" ]]; then
  symbol_text="${SYMBOLS//,/ }"
  read -r -a symbol_values <<< "${symbol_text}"
  if (( ${#symbol_values[@]} > 0 )); then
    TRAIN_COMMAND+=(--symbols "${symbol_values[@]}")
  fi
fi
if [[ "${TINY:-0}" == "1" ]]; then
  TRAIN_COMMAND+=(--tiny)
fi
if (( RESUME_SPECIFIED )); then
  TRAIN_COMMAND+=(--resume "${RESUME_PATH}")
fi
TRAIN_COMMAND+=("${EXTRA_ARGS[@]}")

export CUDA_VISIBLE_DEVICES OMP_NUM_THREADS MKL_NUM_THREADS NUM_GPUS NPROC_PER_NODE \
  NCCL_DEBUG NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_DISABLE NCCL_COLLNET_ENABLE \
  NCCL_CUMEM_ENABLE NCCL_NVLS_ENABLE NCCL_ALGO NCCL_PROTO \
  TORCH_NCCL_ASYNC_ERROR_HANDLING PYTORCH_CUDA_ALLOC_CONF \
  MASTER_ADDR MASTER_PORT LOGGER

mkdir -p "${LAUNCH_LOG_DIR}" "$(dirname -- "${LOG_FILE}")"
{
  echo "[finance-launcher] Starting Finance Sonic DDP training"
  echo "  task=${TASK}"
  echo "  logger=${LOGGER}"
  echo "  python=${PYTHON_BIN}"
  echo "  pythonpath=${PYTHONPATH:-}"
  echo "  device=${EFFECTIVE_DEVICE}"
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "  NUM_GPUS=${NUM_GPUS}"
  echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "  NUM_ENVS=${NUM_ENVS} per rank (global=$((NUM_ENVS * NUM_GPUS)))"
  echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  echo "  MKL_NUM_THREADS=${MKL_NUM_THREADS}"
  echo "  NCCL_DEBUG=${NCCL_DEBUG}"
  echo "  NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
  echo "  NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE}"
  echo "  NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
  echo "  NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE}"
  echo "  NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE}"
  echo "  NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE}"
  echo "  NCCL_ALGO=${NCCL_ALGO}"
  echo "  NCCL_PROTO=${NCCL_PROTO}"
  echo "  TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}"
  echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
  echo "  AUTO_TMUX=${AUTO_TMUX}"
  if (( USE_ARCHIVE_SOURCE )); then
    echo "  trajectory_root=${TRAJECTORY_ROOT}"
  else
    echo "  trajectory_root=(inherited from checkpoint reference recipe)"
  fi
  echo "  output_dir=${OUTPUT_DIR}"
  if [[ "${LOGGER}" == "tensorboard" ]]; then
    echo "  tensorboard_dir=${OUTPUT_DIR}/tensorboard"
    echo "  tensorboard_cmd=tensorboard --logdir ${OUTPUT_DIR}/tensorboard --host 0.0.0.0 --port 6006"
  fi
  echo "  seed=${SEED}"
  echo "  master=${MASTER_ADDR}:${MASTER_PORT}"
  printf '  command='
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '\n'
  echo "  log_file=${LOG_FILE}"
} | tee "${LOG_FILE}"

if [[ "${FINANCE_DRY_RUN}" == "1" ]]; then
  echo "[finance-launcher] FINANCE_DRY_RUN=1; command not executed" | tee -a "${LOG_FILE}"
  exit 0
fi

set +e
"${TRAIN_COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e
if (( status != 0 )); then
  echo "[finance-launcher] training exited with status ${status}; see ${LOG_FILE}" >&2
fi
exit "${status}"
