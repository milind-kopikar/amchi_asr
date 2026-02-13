# Run: 2026-02-13 Marathi Amchi 20 Epochs

- **Date:** 2026-02-13  
- **GPU:** NVIDIA A40 (CUDA_VISIBLE_DEVICES=0)  
- **Config:** `configs/marathi_amchi_20epoch.yaml`  
- **Epochs:** 20  
- **Base model:** `models/indicconformer_stt_mr_hybrid_ctc_rnnt_large/indicconformer_stt_mr_hybrid_rnnt_large.nemo`  

**Data (manifest copies in this folder):**
- Train: `manifest_train.jsonl` (from data/amchi/train)
- Dev:   `manifest_dev.jsonl` (from data/amchi/dev)
- Test:  `manifest_test.jsonl` (from data/amchi/test)

**Checkpoints:** Best 3 by val_wer + last checkpoint in `checkpoints/`.

**Log:** `finetune.log` (stdout/stderr from the run).  
**Metadata:** `run_metadata.json`

Training was started in the background with GPU forced (`CUDA_VISIBLE_DEVICES=0`). When complete, check `experiments/<timestamp>/` for `epoch_metrics.csv`, `samples_epoch_XX.json` (WER+CER), and `final_test_results.json` (WER+CER).
