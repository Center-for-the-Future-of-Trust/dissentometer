#!/usr/bin/env bash
# Create a Python virtual environment for the Groq date‑tagging pipeline.
# Target: any SLURM HPC, present working directory (PWD), no site-specific paths.
#
# Usage:
#   bash env_bootstrap.sh
#   source ./venvs/groq_tagging/bin/activate
#   source ./venvs/groq_tagging/bin/postactivate
#
set -euo pipefail

# Override Groq token limits if desired
export GROQ_TPM=30000
export GROQ_RPM=60
export GROQ_TPD=2000000
export GROQ_RPD=28800

# ===== Layout rooted at PWD =====
ROOT_DIR="${ROOT_DIR:-$PWD}"
ENV_NAME="${ENV_NAME:-groq_tagging}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/venvs/${ENV_NAME}}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs}"
TOKEN_DIR="${TOKEN_DIR:-${ROOT_DIR}/token_state}"

echo "[INFO] Root:    ${ROOT_DIR}"
echo "[INFO] Venv:    ${VENV_DIR}"
echo "[INFO] Data:    ${DATA_DIR}"
echo "[INFO] Outputs: ${OUT_DIR}"
echo "[INFO] Tokens:  ${TOKEN_DIR}"

mkdir -p "${ROOT_DIR}/venvs" "${DATA_DIR}" "${OUT_DIR}" "${TOKEN_DIR}"

# ===== Python: use whatever python3 is on PATH =====
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# Activate and upgrade pip
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Install runtime deps your tagger needs (extend as required)
python -m pip install pandas requests regex

# ===== Post-activate: convenient defaults for rate-limits + token-state =====
POST_ACTIVATE="${VENV_DIR}/bin/postactivate"
cat > "${POST_ACTIVATE}" <<'EOF'
#!/usr/bin/env bash
# Load sensible defaults for rate limiting and shared token-state.
export GROQ_RPM="${GROQ_RPM:-30}"
export GROQ_TPM="${GROQ_TPM:-6000}"
export GROQ_RPD="${GROQ_RPD:-14400}"
export GROQ_TPD="${GROQ_TPD:-500000}"
# Token-state directory (shared across array tasks). Default to ./token_state under the submit dir.
export TOKEN_STATE_DIR="${TOKEN_STATE_DIR:-__TOKEN_DIR__}"
EOF
# inject selected token dir
sed -i "s|__TOKEN_DIR__|${TOKEN_DIR}|g" "${POST_ACTIVATE}"
chmod +x "${POST_ACTIVATE}"

echo
echo "✅ Venv ready at: ${VENV_DIR}"
echo "To activate:  source ${VENV_DIR}/bin/activate && source ${VENV_DIR}/bin/postactivate"
echo "You can change defaults by exporting ROOT_DIR, VENV_DIR, DATA_DIR, OUT_DIR, TOKEN_DIR before running."
