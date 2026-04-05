#!/usr/bin/env bash
# deploy.sh — Install or update reachy-emotion on a Reachy Mini robot over SSH.
#
# Usage:
#   ./deploy.sh <robot-ip>
#   ./deploy.sh <robot-ip> --user reachy
#   ./deploy.sh <robot-ip> --endpoint 34.x.x.x:50051
#
# The script:
#   1. Reads GEMINI_API_KEY and EMOTION_CLOUD_ENDPOINT from your local .env
#   2. SSHes into the robot and clones / pulls the latest code
#   3. Runs install.sh on the robot (system deps + Python package)
#   4. Writes .env on the robot with your API keys
#   5. Restarts the Reachy daemon so the new app is picked up
#
# First-time key setup (do once after finding the robot password):
#   ssh-keygen -t ed25519 -C "reachy-mini" -f ~/.ssh/reachy_ed25519
#   ssh-copy-id -i ~/.ssh/reachy_ed25519.pub reachy@<robot-ip>

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ROBOT_USER="reachy"
ROBOT_DIR="/home/reachy/reachy-emotion"
ROBOT_SSH_KEY="${HOME}/.ssh/reachy_ed25519"
REPO_URL="$(git remote get-url origin 2>/dev/null || echo 'https://github.com/saurabh947/reachy-emotion.git')"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

ROBOT_IP=""
OVERRIDE_ENDPOINT=""

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <robot-ip> [--user <user>] [--key <path>] [--endpoint <host:port>]"
  echo ""
  echo "  --user       SSH username on the robot (default: reachy)"
  echo "  --key        Path to SSH private key (default: ~/.ssh/reachy_ed25519)"
  echo "  --endpoint   Override EMOTION_CLOUD_ENDPOINT (default: from local .env)"
  exit 1
fi

ROBOT_IP="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      ROBOT_USER="$2"; shift 2 ;;
    --key)
      ROBOT_SSH_KEY="$2"; shift 2 ;;
    --endpoint)
      OVERRIDE_ENDPOINT="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Build SSH options: use the dedicated key, skip host-key prompts on first connect
SSH_OPTS="-i ${ROBOT_SSH_KEY} -o StrictHostKeyChecking=accept-new -o BatchMode=yes"

ROBOT="${ROBOT_USER}@${ROBOT_IP}"

# ---------------------------------------------------------------------------
# Load local .env
# ---------------------------------------------------------------------------

LOCAL_ENV="$(dirname "$0")/.env"

_read_env() {
  local key="$1"
  local file="$2"
  # Extract value from KEY=value, ignoring comments and blanks
  grep -E "^${key}=" "$file" 2>/dev/null | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'" | xargs
}

GEMINI_API_KEY=""
EMOTION_CLOUD_ENDPOINT=""

if [[ -f "$LOCAL_ENV" ]]; then
  GEMINI_API_KEY="$(_read_env GEMINI_API_KEY "$LOCAL_ENV")"
  EMOTION_CLOUD_ENDPOINT="$(_read_env EMOTION_CLOUD_ENDPOINT "$LOCAL_ENV")"
fi

# CLI --endpoint flag overrides .env
if [[ -n "$OVERRIDE_ENDPOINT" ]]; then
  EMOTION_CLOUD_ENDPOINT="$OVERRIDE_ENDPOINT"
fi

# ---------------------------------------------------------------------------
# Validate required config
# ---------------------------------------------------------------------------

if [[ -z "$GEMINI_API_KEY" ]]; then
  echo "ERROR: GEMINI_API_KEY is not set."
  echo "  Add it to .env:  GEMINI_API_KEY=your_key_here"
  exit 1
fi

if [[ -z "$EMOTION_CLOUD_ENDPOINT" ]]; then
  echo "ERROR: EMOTION_CLOUD_ENDPOINT is not set."
  echo "  Add it to .env:  EMOTION_CLOUD_ENDPOINT=34.x.x.x:50051"
  echo "  Or pass:         --endpoint 34.x.x.x:50051"
  exit 1
fi

# ---------------------------------------------------------------------------
# Pre-flight summary
# ---------------------------------------------------------------------------

echo ""
echo "=== reachy-emotion deploy ==="
echo "  Robot:              ${ROBOT}"
echo "  SSH key:            ${ROBOT_SSH_KEY}"
echo "  Install dir:        ${ROBOT_DIR}"
echo "  Repo:               ${REPO_URL}"
echo "  GEMINI_API_KEY:     ${GEMINI_API_KEY:0:8}…  (truncated)"
echo "  EMOTION_CLOUD:      ${EMOTION_CLOUD_ENDPOINT}"
echo ""

# ---------------------------------------------------------------------------
# SSH — clone or update
# ---------------------------------------------------------------------------

echo "→ Connecting to robot and syncing code …"

ssh ${SSH_OPTS} "${ROBOT}" bash <<REMOTE
set -euo pipefail

REPO_URL="${REPO_URL}"
ROBOT_DIR="${ROBOT_DIR}"

if [ -d "\${ROBOT_DIR}/.git" ]; then
  echo "  [git] Pulling latest changes …"
  git -C "\${ROBOT_DIR}" fetch --quiet origin
  git -C "\${ROBOT_DIR}" reset --hard origin/main
  git -C "\${ROBOT_DIR}" clean -fd --quiet
else
  echo "  [git] Cloning repo into \${ROBOT_DIR} …"
  mkdir -p "\$(dirname "\${ROBOT_DIR}")"
  git clone --quiet "\${REPO_URL}" "\${ROBOT_DIR}"
fi
REMOTE

echo "  ✓ Code synced."

# ---------------------------------------------------------------------------
# SSH — write .env before install (install.sh won't overwrite an existing .env)
# ---------------------------------------------------------------------------

echo "→ Writing .env on robot …"

ssh ${SSH_OPTS} "${ROBOT}" bash <<REMOTE
set -euo pipefail

cat > "${ROBOT_DIR}/.env" <<'ENV'
# Auto-generated by deploy.sh — edit here or re-run deploy.sh to update.

GEMINI_API_KEY=${GEMINI_API_KEY}
EMOTION_CLOUD_ENDPOINT=${EMOTION_CLOUD_ENDPOINT}

# Optional overrides (uncomment to use)
# GEMINI_MODEL=gemini-2.5-flash
# GEMINI_SYSTEM_PROMPT=You are Reachy, a friendly robot.
ENV

chmod 600 "${ROBOT_DIR}/.env"
echo "  ✓ .env written."
REMOTE

# ---------------------------------------------------------------------------
# SSH — run install.sh
# ---------------------------------------------------------------------------

echo "→ Running install.sh on robot …"

ssh ${SSH_OPTS} "${ROBOT}" bash <<REMOTE
set -euo pipefail
cd "${ROBOT_DIR}"
chmod +x install.sh
# --skip-sys on re-deploys is faster; remove the flag on first install.
# The script detects what's already installed, so it's safe to always run.
./install.sh
REMOTE

echo "  ✓ Install complete."

# ---------------------------------------------------------------------------
# SSH — restart Reachy daemon
# ---------------------------------------------------------------------------

echo "→ Restarting Reachy daemon …"

ssh ${SSH_OPTS} "${ROBOT}" bash <<REMOTE
set -euo pipefail

# Try systemd first (common on robot), fall back to a process signal.
if systemctl is-active --quiet reachy-mini 2>/dev/null; then
  sudo systemctl restart reachy-mini
  echo "  ✓ reachy-mini service restarted (systemd)."
elif systemctl is-active --quiet reachy-mini-daemon 2>/dev/null; then
  sudo systemctl restart reachy-mini-daemon
  echo "  ✓ reachy-mini-daemon service restarted (systemd)."
else
  echo "  ⚠️  No running daemon found via systemd."
  echo "     Start it manually on the robot:"
  echo "       reachy-mini-daemon"
fi
REMOTE

# ---------------------------------------------------------------------------
# SSH — verify installation
# ---------------------------------------------------------------------------

echo "→ Verifying installation …"

ssh ${SSH_OPTS} "${ROBOT}" bash <<REMOTE
set -euo pipefail
cd "${ROBOT_DIR}"

echo "  Python package:"
pip show reachy-emotion 2>/dev/null | grep -E "^(Name|Version|Location)" | sed 's/^/    /'

echo "  System deps:"
reachy-emotion-setup --check-only 2>&1 | sed 's/^/    /' || true

echo "  Cloud endpoint configured:"
grep EMOTION_CLOUD_ENDPOINT .env | sed 's/^/    /'
REMOTE

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "✅ reachy-emotion deployed to ${ROBOT_IP}"
echo ""
echo "  To run the app on the robot:"
echo "    ssh -i ${ROBOT_SSH_KEY} ${ROBOT}"
echo "    reachy-emotion"
echo ""
echo "  Or open the dashboard:  http://${ROBOT_IP}:8000"
echo ""
