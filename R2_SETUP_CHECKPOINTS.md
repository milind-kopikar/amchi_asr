# Set Up R2 for Amchi Checkpoints

Use Cloudflare R2 to store your checkpoints so the RunPod serverless worker can download them via `CHECKPOINT_URL` (no need to bake the checkpoint into the Docker image).

---

## Step 1: Create an R2 bucket (Cloudflare Dashboard)

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com).
2. Go to **R2 Object Storage** (left sidebar, or **Workers & Pages** → **R2**).
3. Click **Create bucket**.
4. **Bucket name:** e.g. `amchi-checkpoints` (lowercase, numbers, hyphens only; 3–63 chars).
5. **Location:** choose a region (e.g. Automatic or one close to your RunPod region).
6. Click **Create bucket**.

You don’t need to give the bucket a “link” to me—you’ll use the bucket name and API credentials in the next steps.

---

## Step 2: Create R2 API credentials

The upload script needs an **Access Key ID** and **Secret Access Key** (S3-compatible).

1. In the R2 section, click **Manage R2 API Tokens** (top right).
2. Click **Create API token**.
3. **Token name:** e.g. `amchi-upload`.
4. **Permissions:** **Object Read & Write** (or **Admin Read & Write** if you prefer).
5. Optionally restrict to the bucket you created (e.g. `amchi-checkpoints`).
6. Click **Create API Token**.
7. **Copy and save** the **Access Key ID** and **Secret Access Key** immediately (the secret is shown only once). You’ll also need your **Account ID** (in the R2 overview or right sidebar).

---

## Step 3: Upload the checkpoint from your RunPod pod

The checkpoints live on your RunPod pod. Run the upload script **on that pod** (it has the file and will send it to R2). **I can’t run this for you**—you run it in your RunPod terminal with your R2 env vars set.

**On your RunPod pod:**

```bash
cd /workspace/amchi_asr
pip install boto3   # if not already installed

# Use the R2 env vars you set (account id, access key, secret key, bucket name)
# Then run:

# Upload default best checkpoint; R2 key = results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt
python scripts/upload_checkpoint_to_r2.py --public-url
```

- The script uploads to an **object key that mirrors your repo path** (e.g. `results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt`) so you always know which checkpoint is where. You can add more checkpoints later with `--file path/to/other.ckpt` and the key will follow the same layout.
- `--public-url` prints the object key and instructions to get a **public URL** (no expiry). Follow the steps it prints, then use that URL as `CHECKPOINT_URL` in RunPod.
- To upload a different file:  
  `python scripts/upload_checkpoint_to_r2.py --file results/2026-03-01_run/checkpoints/best.ckpt --public-url`  
  (key will be `results/2026-03-01_run/checkpoints/best.ckpt`.)

**Optional:** To avoid putting secrets in the shell history, use a small env file and `set -a; source .env.r2; set +a` (and add `.env.r2` to `.gitignore`).

---

## Step 4: Public URL (recommended — no expiry)

1. After upload with `--public-url`, the script prints the **object key** (e.g. `results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt`).
2. In **Cloudflare Dashboard** → **R2** → your bucket → **Settings** → turn on **Public access** (Allow Access). Note the **public bucket URL** (e.g. `https://pub-xxxx.r2.dev`).
3. Your checkpoint’s public URL is: **`<public-bucket-url>/<object-key>`** (e.g. `https://pub-xxxx.r2.dev/results/2026-02-13_marathi_amchi_20epoch/checkpoints/marathi_amchi_20epoch-epoch=18-val_wer=0.550.ckpt`).
4. In **RunPod** → your Serverless endpoint → **Environment Variables** → set **`CHECKPOINT_URL`** to that full URL. Save.
5. New workers will download the checkpoint from R2 when they start. The URL does not expire.

**Alternative (presigned):** If you prefer not to make the bucket public, use `python scripts/upload_checkpoint_to_r2.py --presigned-expiry 604800` and set `CHECKPOINT_URL` to the printed URL. You’ll need to refresh it before it expires (e.g. every 7 days).

---

## Summary

| Step | Where | What |
|------|--------|------|
| 1 | Cloudflare Dashboard | Create R2 bucket (e.g. `amchi-checkpoints`). |
| 2 | Cloudflare Dashboard | Create R2 API token; save Account ID, Access Key ID, Secret Access Key. |
| 3 | RunPod pod | Set R2 env vars; run `python scripts/upload_checkpoint_to_r2.py --public-url`. Checkpoints are stored under keys like `results/2026-02-13_marathi_amchi_20epoch/checkpoints/...` so you know which is where. |
| 4 | Cloudflare Dashboard | Enable Public access on the bucket; note the public bucket URL. |
| 5 | RunPod endpoint | Set `CHECKPOINT_URL` to `<public-bucket-url>/<object-key>` (no expiry). |

You run the upload script on your RunPod pod; I can’t access your R2 or your pod. For more checkpoints later, run the script with `--file path/to/other.ckpt` and the R2 key will mirror the path.
