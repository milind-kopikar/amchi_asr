#!/usr/bin/env python3
"""Setup a new RunPod instance for VS Code Remote-SSH.

Actions performed:
- Adds/updates an SSH config Host entry for the pod
- Copies a local public key to the pod's ~/.ssh/authorized_keys (prompts for password if needed)
- Tests PTY allocation (ssh ... 'tty')
- Updates a .code-workspace file to add `remote.SSH.remotePlatform` mapping and enable `remote.SSH.showLoginTerminal`

Usage:
  python scripts/setup_runpod_remote.py --host 157.157.221.29 --port 31603 --user root \
      --pubkey ~/.ssh/runpod_ed25519.pub --identity ~/.ssh/runpod_ed25519 \
      --workspace ../konkani_asr.code-workspace --host-alias runpod-large

Note: copying the public key uses an SSH connection that may prompt for a password.
"""

from pathlib import Path
import argparse
import subprocess
import sys
import json
import shutil
import re

HOME = Path.home()
SSH_CONFIG = HOME / '.ssh' / 'config'

HOST_BLOCK_TMPL = '''# RunPod host entry (added by setup_runpod_remote.py)
Host {alias}
  HostName {host}
  User {user}
  Port {port}
  IdentityFile "{identity}"
  IdentitiesOnly yes
'''


def safe_backup(path: Path):
    bak = path.with_suffix(path.suffix + '.bak') if path.exists() else None
    if bak and not bak.exists():
        shutil.copy(path, bak)
        print(f'Backed up {path} -> {bak}')


def ensure_ssh_config_alias(alias, host, port, user, identity):
    SSH_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    if not SSH_CONFIG.exists():
        SSH_CONFIG.write_text('')
    config = SSH_CONFIG.read_text()
    # Replace existing Host block for alias if present
    pattern = re.compile(r"(^|\n)Host\s+" + re.escape(alias) + r"[\s\S]*?(?=\nHost\s|\Z)", re.M)
    block = HOST_BLOCK_TMPL.format(alias=alias, host=host, port=port, user=user, identity=identity)
    if pattern.search(config):
        config = pattern.sub('\n' + block, config)
        SSH_CONFIG.write_text(config)
        print(f'Updated existing Host entry for {alias} in {SSH_CONFIG}')
    else:
        with SSH_CONFIG.open('a', encoding='utf-8') as f:
            f.write('\n' + block)
        print(f'Appended Host entry for {alias} to {SSH_CONFIG}')


def copy_pubkey_to_remote(pubkey_path: Path, host: str, port: int, user: str):
    if not pubkey_path.exists():
        print(f'Public key not found: {pubkey_path}', file=sys.stderr)
        return False
    pubkey = pubkey_path.read_text().strip() + '\n'

    cmd = ['ssh', '-p', str(port), f'{user}@{host}', 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys']
    print('Copying public key to remote (you may be prompted for password)...')
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.communicate(pubkey.encode('utf-8'))
        if p.returncode == 0:
            print('Public key appended to remote authorized_keys')
            return True
        else:
            print('Failed to append public key (ssh exited non-zero)', file=sys.stderr)
            return False
    except Exception as e:
        print('Error running ssh to copy pubkey:', e, file=sys.stderr)
        return False


def test_pty(host_alias=None, host=None, port=None, user=None, identity=None):
    if host_alias:
        cmd = ['ssh', '-vvv', '-tt', host_alias, 'tty']
    else:
        cmd = ['ssh', '-i', str(identity), '-p', str(port), f'{user}@{host}', 'tty']
    print('Testing PTY allocation...')
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print('STDOUT:\n', res.stdout)
        print('STDERR:\n', res.stderr)
        if res.returncode == 0 and '/dev/pts' in res.stdout:
            print('PTY allocated successfully:', res.stdout.strip())
            return True
        else:
            print('PTY test failed (see output above)')
            return False
    except subprocess.TimeoutExpired:
        print('PTY test timed out', file=sys.stderr)
        return False


def update_workspace_settings(workspace_path: Path, alias: str):
    if not workspace_path.exists():
        print(f'Workspace file not found: {workspace_path}. Skipping workspace update.')
        return
    text = workspace_path.read_text()
    try:
        data = json.loads(text)
    except Exception:
        print('Workspace file is not strict JSON (may be JSONC). Skipping automated edit. Please add the following to your workspace settings manually:')
        print(json.dumps({"remote.SSH.remotePlatform": {alias: "linux"}, "remote.SSH.showLoginTerminal": True}, indent=2))
        return
    settings = data.get('settings', {})
    settings.setdefault('remote.SSH.remotePlatform', {})[alias] = 'linux'
    settings['remote.SSH.showLoginTerminal'] = True
    data['settings'] = settings
    workspace_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f'Updated workspace settings in {workspace_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, default=22)
    p.add_argument('--user', default='root')
    p.add_argument('--pubkey', default=str(HOME / '.ssh' / 'runpod_ed25519.pub'))
    p.add_argument('--identity', default=str(HOME / '.ssh' / 'runpod_ed25519'))
    p.add_argument('--workspace', default='konkani_asr.code-workspace')
    p.add_argument('--host-alias', default='runpod-large')
    args = p.parse_args()

    ensure_ssh_config_alias(args.host_alias, args.host, args.port, args.user, args.identity)
    # attempt to copy pubkey
    pubkey_path = Path(args.pubkey).expanduser()
    ok = copy_pubkey_to_remote(pubkey_path, args.host, args.port, args.user)
    if not ok:
        print('Warning: pubkey copy failed. You can paste the public key into your RunPod UI (recommended) or rerun this script.')
    # test PTY
    identity = Path(args.identity).expanduser()
    test_pty(host_alias=args.host_alias, identity=identity, host=args.host, port=args.port, user=args.user)
    # update workspace
    ws_path = Path(args.workspace)
    update_workspace_settings(ws_path, args.host_alias)

    print('\nDone. If the PTY test succeeded, try connecting from VS Code (Remote-SSH: Connect to Host... ->', args.host_alias, ')')
