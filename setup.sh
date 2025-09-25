#!/usr/bin/env bash
set -euo pipefail

# One-click environment setup for getmail.py and helpers
# - Creates Python venv (.venv)
# - Installs dependencies
# - Creates/updates newsletter.yml and optional .env

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$THIS_DIR"

EMAIL=""
MAILBOX="[Gmail]/All Mail"
BATCH_SIZE="200"
PROXY=""
YES="false"
OVERWRITE_CONFIG="false"
NO_VENV="false"
NO_INSTALL="false"

print_help() {
  cat <<'EOF'
Usage: bash setup.sh [options]

Options:
  --email EMAIL            Set Gmail address for config
  --mailbox NAME           Mailbox to use (default: [Gmail]/All Mail)
  --batch-size N           Fetch batch size (default: 200)
  --proxy URL              HTTP/HTTPS proxy (e.g., http://127.0.0.1:1087)
  -y, --yes                Non-interactive; accept defaults where possible
  --overwrite-config       Overwrite existing newsletter.yml
  --no-venv                Skip creating/using .venv
  --no-install             Skip pip install (assumes deps already installed)
  -h, --help               Show this help

Notes:
  - The script can write GMAIL_APP_PASSWORD into .env (optional).
  - For security, avoid passing passwords via CLI args.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --email) EMAIL="${2:-}"; shift 2 ;;
    --mailbox) MAILBOX="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --proxy) PROXY="${2:-}"; shift 2 ;;
    -y|--yes) YES="true"; shift ;;
    --overwrite-config) OVERWRITE_CONFIG="true"; shift ;;
    --no-venv) NO_VENV="true"; shift ;;
    --no-install) NO_INSTALL="true"; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown option: $1" >&2; print_help; exit 2 ;;
  esac
done

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found" >&2; exit 1; }; }

PYTHON_BIN="${PYTHON:-python3}"
need_cmd "$PYTHON_BIN"

# Check Python version >= 3.9
"$PYTHON_BIN" - <<'PY' || { echo "ERROR: Python >=3.9 required" >&2; exit 1; }
import sys
major, minor = sys.version_info[:2]
assert (major, minor) >= (3, 9)
PY

if [[ "$NO_VENV" != "true" ]]; then
  if [[ ! -d .venv ]]; then
    echo "Creating virtualenv in .venv ..."
    "$PYTHON_BIN" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PYTHON_BIN="python"
fi

if [[ "$NO_INSTALL" != "true" ]]; then
  echo "Upgrading pip and installing dependencies ..."
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel >/dev/null
  "$PYTHON_BIN" -m pip install pyyaml imapclient html2text python-dotenv >/dev/null
fi

# Prepare newsletter.yml
CONFIG_PATH="newsletter.yml"
if [[ -f "$CONFIG_PATH" && "$OVERWRITE_CONFIG" != "true" ]]; then
  echo "Config exists: $CONFIG_PATH (use --overwrite-config to rebuild)"
else
  if [[ -z "$EMAIL" && "$YES" != "true" ]]; then
    read -r -p "Enter your Gmail address (e.g., you@gmail.com): " EMAIL
  fi
  EMAIL=${EMAIL:-your_email@gmail.com}

  # Ask for password to embed into YAML (or use env / placeholder)
  YAML_PASSWORD=""
  if [[ "$YES" == "true" ]]; then
    YAML_PASSWORD="${GMAIL_APP_PASSWORD:-}"
  else
    echo "You can embed the Gmail App Password into newsletter.yml (stored locally)."
    read -r -s -p "Enter Gmail App Password for YAML (leave empty to write placeholder): " YAML_PASSWORD; echo
  fi
  YAML_PASSWORD=${YAML_PASSWORD:-your_gmail_app_password}

  # Optional proxy
  if [[ -z "$PROXY" && "$YES" != "true" ]]; then
    read -r -p "HTTP/HTTPS proxy (e.g., http://127.0.0.1:1087) [empty for none]: " PROXY
  fi

  cat > "$CONFIG_PATH" <<EOF
email:
  provider: gmail
  username: "$EMAIL"
  password: "$YAML_PASSWORD"
  imap:
    host: imap.gmail.com
    port: 993
    ssl: true
    mailbox: "$MAILBOX"
runtime:
  out_dir: "runtime/mails"
  state_file: "runtime/state.json"
fetch:
  batch_size: ${BATCH_SIZE}
EOF

  if [[ -n "$PROXY" ]]; then
    cat >> "$CONFIG_PATH" <<EOF
proxy: $PROXY
EOF
  fi
  echo "Wrote $CONFIG_PATH"
fi

# Prepare .env (optional)
ENV_PATH=".env"
WRITE_ENV="false"
if [[ -f "$ENV_PATH" ]]; then
  echo "Env file exists: $ENV_PATH (keeping as is)"
else
  if [[ "$YES" == "true" ]]; then
    # If env already in process, persist it; else write placeholder
    if [[ -n "${GMAIL_APP_PASSWORD:-}" ]]; then
      printf 'GMAIL_APP_PASSWORD=%s\n' "$GMAIL_APP_PASSWORD" > "$ENV_PATH"
      WRITE_ENV="true"
    else
      printf 'GMAIL_APP_PASSWORD=%s\n' "your_gmail_app_password" > "$ENV_PATH"
      WRITE_ENV="true"
    fi
  else
    echo "Optionally set your Gmail App Password to .env now."
    read -r -p "Write password to .env? [y/N]: " ans
    if [[ "${ans:-}" =~ ^[Yy]$ ]]; then
      read -r -s -p "Enter Gmail App Password: " pw; echo
      printf 'GMAIL_APP_PASSWORD=%s\n' "$pw" > "$ENV_PATH"
      WRITE_ENV="true"
    else
      # Create example env if none exists
      printf 'GMAIL_APP_PASSWORD=%s\n' "your_gmail_app_password" > ".env.example"
      echo "Created .env.example (you can copy to .env and edit)."
    fi
  fi
fi

# Create runtime directories
mkdir -p runtime/mails
mkdir -p runtime/read

# Quick smoke test
echo "Running quick dependency check ..."
"$PYTHON_BIN" - <<'PY'
try:
    import yaml, imaplib, email, ssl  # stdlib + PyYAML
    try:
        import imapclient  # optional, but should be installed
    except Exception:
        pass
    try:
        import html2text
    except Exception:
        pass
    print("OK: Python deps import succeeded.")
except Exception as e:
    raise SystemExit(f"Dependency check failed: {e}")
PY

echo ""
echo "Setup complete. Next steps:"
if [[ "$NO_VENV" != "true" ]]; then
  echo "  1) Activate venv:    source .venv/bin/activate"
else
  echo "  1) (Using system Python; ensure deps remain available)"
fi
echo "  2) Verify newsletter.yml has correct email.password (or update it)"
echo "  3) (Optional) Put GMAIL_APP_PASSWORD in .env for your own use"
echo "  4) Run fetcher:      python getmail.py --config newsletter.yml -v --limit 10"
echo "  5) Process mails:    python readmail.py --dry-run -v   # or without --dry-run"
echo "  6) Continuous loop:  python runloop.py"
echo ""
echo "Markdown files will be saved under runtime/mails/"
