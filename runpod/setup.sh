#!/usr/bin/env bash
set -euo pipefail

# runpod/setup.sh
# Secure installer for code-server (https://code.visualstudio.com/docs/remote/code-server)
# This script installs code-server, creates a systemd service to run it as the current user,
# and configures it to bind to localhost:8080 with a generated password so access is via SSH tunnel.

# Usage: sudo ./setup.sh

USER_TO_INSTALL=${SUDO_USER:-$USER}
HOME_DIR=$(eval echo "~$USER_TO_INSTALL")
CONFIG_DIR="$HOME_DIR/.config/code-server"
SERVICE_FILE="/etc/systemd/system/code-server.service"

echo "Installing code-server for user: $USER_TO_INSTALL"

# Detect package manager and install prerequisites
if command -v apt-get >/dev/null 2>&1; then
  pkgmgr=apt
  echo "Using apt to install prerequisites"
  apt-get update
  apt-get install -y curl ca-certificates tar openssh-server
elif command -v yum >/dev/null 2>&1; then
  pkgmgr=yum
  echo "Using yum to install prerequisites"
  yum install -y curl ca-certificates tar openssh-server
else
  echo "Unknown package manager. Please install curl/ca-certificates/tar manually and re-run this script." >&2
  exit 1
fi

# Install code-server (official install script)
if ! command -v code-server >/dev/null 2>&1; then
  echo "Downloading and installing code-server..."
  curl -fsSL https://code-server.dev/install.sh | sh
else
  echo "code-server already installed"
fi

# Generate a random password if none provided
if [ -z "${CODE_SERVER_PASSWORD:-}" ]; then
  CODE_SERVER_PASSWORD=$(openssl rand -base64 24)
  echo "Generated random code-server password"
fi

mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/config.yaml" <<EOF
bind-addr: 127.0.0.1:8080
auth: password
password: "$CODE_SERVER_PASSWORD"
cert: false
EOF

chown -R "$USER_TO_INSTALL":"$USER_TO_INSTALL" "$CONFIG_DIR"

# Create systemd service
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=code-server
After=network.target

[Service]
Type=simple
User=$USER_TO_INSTALL
Environment=PATH=/usr/bin:/usr/local/bin
ExecStart=/usr/bin/code-server --config "$CONFIG_DIR/config.yaml"
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now code-server.service

echo "---"
echo "code-server installed and started as a systemd service. It is bound to 127.0.0.1:8080 and requires SSH tunneling to access."
echo "Connect with: ssh -L 8080:127.0.0.1:8080 $USER_TO_INSTALL@<RUNPOD_HOST>"
echo "Then open: http://localhost:8080 in your browser and use the following password:"
echo
echo "${CODE_SERVER_PASSWORD}"
echo

echo "If you prefer to expose code-server via a public TLS-enabled reverse proxy (nginx), modify the config above (set cert: true) and configure nginx as a proxy."

exit 0
