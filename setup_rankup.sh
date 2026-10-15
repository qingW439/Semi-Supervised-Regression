#!/usr/bin/env bash
# RankUp / Yelp Review: only install and validate the software/data environment.
# Official paper: https://arxiv.org/abs/2410.22124 (NeurIPS 2024)
# Official code: https://github.com/pm25/Semi-Supervised-Regression
# Reproduction branch: https://github.com/qingW439/Semi-Supervised-Regression/tree/rankup-reproduction
# Runs on standard Linux x86_64 + NVIDIA GPU cloud images. Does NOT run experiments,
# delete model checkpoints, write any experiment records, or push to GitHub.
set -Eeuo pipefail
trap 'printf "\n[ERROR] Installation failed at line %s (exit %s). Existing experiments are untouched.\n" "$LINENO" "$?" >&2' ERR

# Choose a directory on the server's PERSISTENT/LARGE DATA DISK if needed:
#   RANKUP_ROOT=/workspace/rankup bash setup_rankup.sh
ROOT="${RANKUP_ROOT:-$HOME/rankup_workspace}"
REPO="$ROOT/Semi-Supervised-Regression"
ENV="$ROOT/envs/rankup"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
REPO_URL="${RANKUP_REPO_URL:-https://github.com/qingW439/Semi-Supervised-Regression.git}"
BRANCH="${RANKUP_BRANCH:-rankup-reproduction}"
MODEL_ID='google/bert_uncased_L-4_H-512_A-8'
DATA_URL="${YELP_URL:-https://huggingface.co/datasets/py97/Yelp-Review/resolve/main/YelpReview.tar.gz}"
DATA_MD5='4c3e3736f3dc2c175f5ff9b0f69e6043'
CONDA=''

log() { printf '\n[RankUp setup] %s\n' "$*"; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
has() { command -v "$1" >/dev/null 2>&1; }

[[ "$(uname -s)" == Linux ]] || die 'Linux only.'
[[ "$(uname -m)" == x86_64 ]] || die 'This installer supports x86_64 Linux GPU instances; ARM/Jetson require different PyTorch wheels.'
has nvidia-smi || die 'No nvidia-smi. Choose a NVIDIA GPU instance with its driver installed.'
nvidia-smi -L >/dev/null || die 'GPU driver is unavailable. Verify nvidia-smi first.'

mkdir -p "$ROOT" "$ROOT/envs" "$ROOT/downloads" "$HF_HOME"
ROOT="$(cd "$ROOT" && pwd -P)"
REPO="$ROOT/Semi-Supervised-Regression"
ENV="$ROOT/envs/rankup"

# Install base command-line tools only when needed; never modify NVIDIA drivers.
install_os_packages() {
  local installer=() pkgs=() c
  for c in git curl tar gzip sha256sum md5sum bzip2; do
    has "$c" || pkgs+=("$c")
  done
  [[ ${#pkgs[@]} -gt 0 ]] || return 0
  log "Installing missing OS tools: ${pkgs[*]}"
  if [[ "$(id -u)" != 0 ]]; then
    has sudo || die "Missing ${pkgs[*]}; need sudo/root or a GPU image with prerequisites installed."
    installer=(sudo)
  fi
  if has apt-get; then
    "${installer[@]}" apt-get update -qq
    "${installer[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
      git curl ca-certificates tar gzip bzip2 xz-utils coreutils
  elif has dnf; then
    "${installer[@]}" dnf install -y git curl ca-certificates tar gzip bzip2 xz coreutils
  elif has yum; then
    "${installer[@]}" yum install -y git curl ca-certificates tar gzip bzip2 xz coreutils
  else
    die 'No apt-get/dnf/yum. Please install git, curl, tar, gzip, sha256sum, md5sum, bzip2 first.'
  fi
  for c in git curl tar gzip sha256sum md5sum bzip2; do
    has "$c" || die "Missing command: $c"
  done
}
install_os_packages

# CUDA runtime is supplied by PyTorch wheels: a system CUDA toolkit is not required.
# The driver must be new enough for the selected runtime.
select_pytorch() {
  local driver major cap cc_major=0
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')"
  major="${driver%%.*}"
  [[ "$major" =~ ^[0-9]+$ ]] || die "Unexpected NVIDIA driver version: $driver"
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
  if [[ "$cap" =~ ^([0-9]+)\.([0-9]+)$ ]]; then cc_major="${BASH_REMATCH[1]}"; fi

  # Normal GPU cloud cards: preserve the prior tested software stack.
  TORCH_VERSION=2.4.1; VISION_VERSION=0.19.1; AUDIO_VERSION=2.4.1
  if (( cc_major >= 10 )); then
    # Blackwell GPUs (e.g. RTX 5090) need sm_120 support and CUDA 12.8 wheels.
    (( major >= 570 )) || die "Blackwell GPU ($cap) needs a newer driver (current: $driver)."
    TORCH_VERSION=2.7.1; VISION_VERSION=0.22.1; AUDIO_VERSION=2.7.1
    CUDA_INDEX=cu128
    log 'Blackwell GPU detected: using newer PyTorch, NOT an identical software-version reproduction.'
  elif (( major >= 550 )); then
    CUDA_INDEX=cu124
  elif (( major >= 530 )); then
    CUDA_INDEX=cu121
  elif (( major >= 520 )); then
    CUDA_INDEX=cu118
  else
    die "NVIDIA driver $driver too old. Choose a GPU image with a newer driver."
  fi
  if [[ -n "${RANKUP_TORCH_CUDA:-}" ]]; then
    CUDA_INDEX="$RANKUP_TORCH_CUDA"
    case "$CUDA_INDEX" in cu118|cu121|cu124|cu128) ;; *) die 'RANKUP_TORCH_CUDA must be cu118/cu121/cu124/cu128.';; esac
    if (( cc_major >= 10 )) && [[ "$CUDA_INDEX" != cu128 ]]; then
      die 'Blackwell GPUs require cu128 in this installer.'
    fi
    if (( cc_major < 10 )) && [[ "$CUDA_INDEX" == cu128 ]]; then
      die 'cu128 override only supported with the newer Blackwell PyTorch stack here.'
    fi
    case "$CUDA_INDEX" in
      cu118) (( major >= 520 )) || die 'Driver too old for cu118.';;
      cu121) (( major >= 530 )) || die 'Driver too old for cu121.';;
      cu124) (( major >= 550 )) || die 'Driver too old for cu124.';;
      cu128) (( major >= 570 )) || die 'Driver too old for cu128.';;
    esac
  fi
  log "NVIDIA driver=$driver; compute capability=${cap:-unknown}; PyTorch=$TORCH_VERSION ($CUDA_INDEX)"
}
select_pytorch

# Conda may be absent on rented GPU machines. Install self-contained Miniforge.
find_conda() {
  if [[ -x "$ROOT/miniforge3/bin/conda" ]]; then
    CONDA="$ROOT/miniforge3/bin/conda"
  elif has conda; then
    CONDA="$(command -v conda)"
  else
    log 'Installing Miniforge in the workspace (without changing shell startup files)...'
    local url installer expected
    url='https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh'
    installer="$ROOT/downloads/Miniforge3-Linux-x86_64.sh"
    curl -fL --retry 5 --retry-all-errors -o "$installer" "$url"
    curl -fL --retry 5 --retry-all-errors -o "$installer.sha256" "$url.sha256"
    expected="$(awk 'NR==1 {print $1}' "$installer.sha256")"
    [[ "$expected" =~ ^[[:xdigit:]]{64}$ ]] || die 'Miniforge installer SHA256 checksum file is invalid.'
    printf '%s  %s\n' "$expected" "$installer" | sha256sum -c -
    bash "$installer" -b -p "$ROOT/miniforge3"
    CONDA="$ROOT/miniforge3/bin/conda"
  fi
  [[ -x "$CONDA" ]] || die 'Conda executable not found.'
}
find_conda

# Never force checkout/pull on a project directory that might contain experiments.
if [[ -d "$REPO/.git" ]]; then
  log "Using existing repository (unchanged): $REPO"
else
  [[ ! -e "$REPO" ]] || die "Directory exists but is not a git repository: $REPO"
  log "Cloning $BRANCH from $REPO_URL ..."
  git -c http.version=HTTP/1.1 clone -b "$BRANCH" --single-branch "$REPO_URL" "$REPO"
fi
for path in train.py eval.py requirements.txt \
  config/reproduction/supervised_yelp_review_lb250_s0.yaml \
  config/reproduction/rankup_yelp_review_lb250_s0.yaml \
  config/reproduction/rankuprda_yelp_review_lb250_s0.yaml; do
  [[ -f "$REPO/$path" ]] || die "Missing file $REPO/$path (wrong branch?)"
done
# Protect checkpoints from the repository's explicit delete-on-overwrite behavior.
for method in supervised rankup rankuprda; do
  cfg="$REPO/config/reproduction/${method}_yelp_review_lb250_s0.yaml"
  if grep -Eiq '^overwrite:[[:space:]]*(true|yes|1)[[:space:]]*$' "$cfg"; then
    die "Unsafe overwrite=True in $cfg. This setup script will not modify tracked configs."
  fi
done

# Protect any environment that already exists; only install if genuinely absent.
if [[ -x "$ENV/bin/python" ]]; then
  log "Reusing existing Python environment, no automatic pip upgrades: $ENV"
else
  [[ ! -e "$ENV" ]] || die "Incomplete env directory at $ENV; inspect it rather than deleting it automatically."
  log "Creating isolated Python 3.10 environment: $ENV"
  "$CONDA" create -y -p "$ENV" python=3.10 pip
  PY="$ENV/bin/python"
  "$PY" -m pip install --upgrade 'pip<26' setuptools wheel
  "$PY" -m pip install --index-url "https://download.pytorch.org/whl/$CUDA_INDEX" \
    "torch==$TORCH_VERSION" "torchvision==$VISION_VERSION" "torchaudio==$AUDIO_VERSION"
  cat > "$ROOT/pip_constraints_rankup.txt" <<EOF
numpy==1.26.4
transformers==4.46.3
huggingface-hub==0.26.5
torch==$TORCH_VERSION
torchvision==$VISION_VERSION
torchaudio==$AUDIO_VERSION
EOF
  "$PY" -m pip install -c "$ROOT/pip_constraints_rankup.txt" -r "$REPO/requirements.txt"
fi
PY="$ENV/bin/python"
"$PY" -m pip check

# Verify that PyTorch/CUDA works and that the repository imports successfully.
log 'Checking Python, CUDA and RankUp imports...'
(cd "$REPO" && "$PY" - <<'PY'
import sys, torch, transformers, numpy
import torchvision, torchaudio
from semilearn.nets.bert.bert import bert_base
from semilearn.datasets.nlp_datasets.yelp_review import YELP_REVIEW
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.split('+')[0] in ('2.4.1','2.7.1'), torch.__version__
assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'
_ = torch.tensor([1.0], device='cuda:0').square()
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__, '| CUDA runtime:', torch.version.cuda)
print('transformers:', transformers.__version__, '| numpy:', numpy.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('Source imports / GPU operations: OK')
PY
)

# Warm HF cache. Evaluation requires config.json even when the checkpoint has all weights.
log "Checking/downloading pretrained $MODEL_ID ..."
"$PY" - <<'PY'
from transformers import BertConfig, BertModel, AutoTokenizer
name = 'google/bert_uncased_L-4_H-512_A-8'
config = BertConfig.from_pretrained(name)
assert (config.hidden_size, config.num_hidden_layers) == (512, 4)
BertModel.from_pretrained(name)
AutoTokenizer.from_pretrained(name)
print('Pretrained Bert-Small model/config/tokenizer: OK')
PY

# Use the exact processed dataset expected by the fork; prevent unsafe tar extraction.
DATA_DIR="$REPO/data/yelp_review"
DATA_FILES="$DATA_DIR/YelpReview"
if [[ -s "$DATA_FILES/train.json" && -s "$DATA_FILES/dev.json" && -s "$DATA_FILES/test.json" ]]; then
  log 'Existing Yelp train/dev/test JSON found: will not replace it.'
else
  mkdir -p "$DATA_DIR"
  if [[ -n "${YELP_ARCHIVE:-}" ]]; then
    ARCHIVE="$YELP_ARCHIVE"
    [[ -f "$ARCHIVE" ]] || die "YELP_ARCHIVE not found: $ARCHIVE"
  else
    ARCHIVE="$ROOT/downloads/YelpReview.tar.gz"
    if [[ ! -s "$ARCHIVE" ]]; then
      log 'Downloading processed Yelp Review archive (large file)...'
      curl -fL --retry 5 --retry-all-errors -C - -o "$ARCHIVE" "$DATA_URL"
    fi
  fi
  printf '%s  %s\n' "$DATA_MD5" "$ARCHIVE" | md5sum -c - || die \
    'Dataset MD5 mismatch. Inspect/remove the corrupt .tar.gz download, then rerun.'
  "$PY" - "$ARCHIVE" "$DATA_DIR" <<'PY'
import sys, tarfile
from pathlib import Path
source, destination = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
with tarfile.open(source, 'r:gz') as archive:
    for member in archive.getmembers():
        p = (destination / member.name).resolve()
        if not p.is_relative_to(destination) or member.issym() or member.islnk() or member.isdev():
            raise ValueError(f'Unsafe archive member: {member.name}')
    archive.extractall(destination)
print('Extracted the verified Yelp Review dataset.')
PY
fi
"$PY" - "$DATA_FILES" <<'PY'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
for split, expected in [('train',250000), ('dev',25000), ('test',50000)]:
    path = base / f'{split}.json'
    with path.open(encoding='utf-8') as file:
        samples = json.load(file)
    assert len(samples) == expected, f'{split}: expected {expected}, found {len(samples)}'
    first = next(iter(samples.values()))
    assert {'ori', 'label'}.issubset(first), first.keys()
    print(f'{split}: {len(samples):,} rows verified')
PY

# Convenience file: use `source "$ROOT/activate_rankup.sh"` before training.
# Export variables but do not change the user shell config (~/.bashrc).
ACTIVATE="$ROOT/activate_rankup.sh"
{
  printf 'export RANKUP_ROOT=%q\n' "$ROOT"
  printf 'export HF_HOME=%q\n' "$HF_HOME"
  printf 'export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n'
  printf 'export PATH=%q:"$PATH"\n' "$ENV/bin"
} > "$ACTIVATE"

{
  printf 'date=%s\n' "$(date -Is)"
  printf 'code=%s\n' "$REPO"
  printf 'commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
  printf 'env=%s\n' "$ENV"
  printf 'hf_cache=%s\n' "$HF_HOME"
  printf 'cuda_wheel=%s\n' "$CUDA_INDEX"
  printf 'pytorch=%s\n' "$TORCH_VERSION"
  printf 'dataset=%s\n' "$DATA_FILES"
  printf '\nNote: YAML keeps uratio=1 in the tested fork, while paper Table 9 lists 7.\n'
} > "$ROOT/setup_report.txt"
"$PY" -m pip freeze > "$ROOT/pip_freeze.txt"

log 'SETUP COMPLETE (training has NOT started).'
printf 'Activate: source %q\n' "$ACTIVATE"
printf 'Project:  %s\n' "$REPO"
printf 'Env:      %s\n' "$ENV"
printf 'Report:   %s\n' "$ROOT/setup_report.txt"
