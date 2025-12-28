# RunPod `code-server` setup and testing

This document explains how to install and connect to `code-server` on a RunPod instance and how to verify editing and execution works from VS Code.

## Quick steps

1. SSH into your RunPod instance (you said you have this working):

   ssh user@<RUNPOD_HOST>

2. Upload this repo or pull it from GitHub on the instance:

   git clone <your-repo> && cd konkani_asr

3. Run the installer as root (allows systemd service creation):

   sudo bash runpod/setup.sh

   The script will print a randomly generated password for `code-server` and configure it to bind to localhost:8080.

4. Create an SSH tunnel from your local machine (do this on your laptop):

   ssh -L 8080:127.0.0.1:8080 user@<RUNPOD_HOST>

5. Open your browser to `http://localhost:8080` and enter the password printed by the setup script.

6. (Recommended) Instead of exposing code-server publicly, use VS Code Remote-SSH:

   In your local `~/.ssh/config` add:

   Host runpod
     HostName <RUNPOD_HOST>
     User <user>
     IdentityFile ~/.ssh/id_ed25519

   Then in local VS Code: `Remote-SSH: Connect to Host... -> runpod`.

## Verifying edits & execution

1. On the RunPod instance, create or pull the repository and navigate to the project folder in code-server.
2. Create this simple test file: `scripts/runpod_test.py`

   ```py
   print('Hello from RunPod')
   with open('runpod_test_output.txt', 'w') as f:
       f.write('Hello from RunPod')
   ```

3. Edit the file in VS Code (remote) and save.
4. Run it in the remote terminal: `python scripts/runpod_test.py` and verify `runpod_test_output.txt` is created and contains the expected text.

## Troubleshooting

- If code-server fails to start, check `systemctl status code-server` and `journalctl -u code-server -n 200`.
- If the VS Code server fails to install during Remote-SSH, ensure the instance has `tar`, `sh`, and can write to home directory.
- If the instance is ephemeral, push important changes to Git and push artifacts to S3.

## Security notes

- The installer binds code-server to `127.0.0.1:8080` and uses password auth by default. This requires SSH tunneling, which is safe.
- For production-style access, place `code-server` behind an authenticated reverse proxy with TLS.
