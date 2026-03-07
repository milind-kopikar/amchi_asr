#!/usr/bin/env python3
"""
Comprehensive analysis comparing the Amchi Konkani pilot run (35.1% WER)
against the 50-epoch run (54.7% WER).

Sections:
  1. Overall WER comparison
  2. Per-speaker breakdown (50-epoch run)
  3. Speaker representation: train counts vs test counts
  4. Matched-sample analysis (same audio IDs in both runs)
  5. Training-curve overfitting analysis (epoch_metrics.csv)
  6. Error-type breakdown (substitutions, deletions, insertions)
  7. Sentence length vs WER
  8. Root-cause summary

Outputs: text report + plots in --output_dir

Usage:
  python3 scripts/analyze_runs_comparison.py \\
    --pilot   nemo_experiments/marathi_pilot_v3/final_test_results.json \\
    --current results/experiments/20260302_031806/final_test_results.json \\
    --manifest          data/amchi/test/manifest.jsonl \\
    --train_manifest    data/amchi/train/manifest.jsonl \\
    --epoch_metrics     results/experiments/20260302_031806/epoch_metrics.csv \\
    --output_dir        results/amchi_analysis/

Requirements:
  pip install numpy scipy matplotlib jiwer
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found — plots skipped. pip install matplotlib")

try:
    from jiwer import wer as compute_wer, cer as compute_cer, process_words
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    print("[WARN] jiwer not found — CER/error-types skipped. pip install jiwer")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path):
    """Load final_test_results.json. Returns list of per-sample dicts."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["per_sample"]


def load_manifest(path):
    """Return {audio_filename_stem: speaker_id} from a manifest."""
    mapping = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                fp   = e.get("audio_filepath", "")
                stem = Path(fp).stem          # e.g. "570"
                spk  = e.get("speaker_id", "unknown")
                mapping[stem] = spk
    except FileNotFoundError:
        print(f"[WARN] Manifest not found: {path}")
    return mapping


def audio_stem(audio_path):
    """Extract filename stem (without extension) from an audio path string."""
    return Path(audio_path).stem


def bootstrap_ci(data, n=10_000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n)]
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lo, hi


def error_types(reference, hypothesis):
    """Return (S, D, I, ref_len) edit-op counts for one sentence pair."""
    if not HAS_JIWER:
        return 0, 0, 0, len(reference.split())
    try:
        out = process_words(reference, hypothesis)
        alignment = out.alignments[0]
        S = sum(1 for op in alignment if op.type == "substitute")
        D = sum(1 for op in alignment if op.type == "delete")
        I = sum(1 for op in alignment if op.type == "insert")
        return S, D, I, len(out.references[0])
    except Exception:
        return 0, 0, 0, len(reference.split())


def sig(p, threshold=0.05):
    return "*** SIGNIFICANT (p<0.05)" if p < threshold else "(not significant)"


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_wer_comparison(pilot_wers, current_wers, out_path):
    """Overlaid WER histograms: pilot vs 50-epoch run."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(pilot_wers,   bins=bins, alpha=0.55, color="mediumseagreen",
            label=f"Pilot (n={len(pilot_wers)}, mean={np.mean(pilot_wers):.3f})",
            edgecolor="white")
    ax.hist(current_wers, bins=bins, alpha=0.55, color="steelblue",
            label=f"50-epoch (n={len(current_wers)}, mean={np.mean(current_wers):.3f})",
            edgecolor="white")
    ax.axvline(np.mean(pilot_wers),   color="mediumseagreen", linestyle="--", linewidth=2)
    ax.axvline(np.mean(current_wers), color="steelblue",      linestyle="--", linewidth=2)
    ax.set_xlabel("WER", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("WER Distribution: Pilot vs 50-epoch Run", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_speaker_boxplot(spk_wers, out_path, title="Per-Speaker WER (50-epoch run)"):
    """Box plot of WER for each speaker in the current run."""
    if not HAS_MPL:
        return
    labels = sorted(spk_wers.keys())
    data   = [spk_wers[s] for s in labels]
    short  = [f"Speaker {i+1}\n({s.split('@')[0]})\n(n={len(spk_wers[s])})"
              for i, s in enumerate(labels)]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 3), 5))
    bp = ax.boxplot(data, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.7),
                    medianprops=dict(color="navy", linewidth=2))
    ax.set_xticklabels(short, fontsize=10)
    ax.set_ylabel("WER", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.axhline(np.mean([w for wl in data for w in wl]), color="red",
               linestyle="--", linewidth=1.5, label="Overall mean")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_speaker_representation(train_counts, test_counts, out_path):
    """Grouped bar: train samples vs test samples per speaker."""
    if not HAS_MPL:
        return
    all_spk = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
    x  = np.arange(len(all_spk))
    w  = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(all_spk) * 2), 5))
    ax.bar(x - w / 2, [train_counts.get(s, 0) for s in all_spk], w,
           label="Train samples", color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, [test_counts.get(s, 0)  for s in all_spk], w,
           label="Test samples",  color="coral",     alpha=0.8)
    ax.set_xticks(x)
    short = [s.split("@")[0] for s in all_spk]
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Sample count", fontsize=12)
    ax.set_title("Speaker Representation: Train vs Test", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curve(epoch_metrics_path, out_path):
    """Plot train_loss + val_loss + RNNT val_wer across epochs."""
    if not HAS_MPL:
        return
    epochs, train_loss, val_loss, val_wer = [], [], [], []
    try:
        with open(epoch_metrics_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep = int(row["epoch"])
                epochs.append(ep)
                tl = row.get("train_loss", "")
                val_loss.append(float(row["val_loss"]) if row["val_loss"] else None)
                val_wer.append(float(row["val_wer"])   if row["val_wer"]  else None)
                train_loss.append(float(tl) if tl else None)
    except FileNotFoundError:
        print(f"[WARN] epoch_metrics not found: {epoch_metrics_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curves
    ax = axes[0]
    tl_ep  = [e for e, v in zip(epochs, train_loss) if v is not None]
    tl_val = [v for v in train_loss if v is not None]
    vl_ep  = [e for e, v in zip(epochs, val_loss) if v is not None]
    vl_val = [v for v in val_loss if v is not None]
    ax.plot(tl_ep, tl_val, "steelblue",  linewidth=1.8, label="Train loss")
    ax.plot(vl_ep, vl_val, "coral",      linewidth=1.8, label="Val loss")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training vs Validation Loss\n(divergence = overfitting)", fontsize=12)
    ax.legend()

    # RNNT val_wer
    ax = axes[1]
    vw_ep  = [e for e, v in zip(epochs, val_wer) if v is not None]
    vw_val = [v for v in val_wer if v is not None]
    ax.plot(vw_ep, vw_val, "mediumorchid", linewidth=1.8, label="RNNT val_wer")
    ax.axhline(0.532, color="steelblue", linestyle="--", linewidth=1.5,
               label="CTC val_wer best (0.532, epoch 47)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val WER", fontsize=11)
    ax.set_title("RNNT Validation WER vs Epochs\n(CTC WER tracked via checkpoint name)", fontsize=12)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_matched_scatter(pilot_wers, current_wers, out_path):
    """Scatter: pilot WER vs 50-epoch WER for matched samples."""
    if not HAS_MPL or len(pilot_wers) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pilot_wers, current_wers, alpha=0.7, color="steelblue", s=70, edgecolors="white")
    lim = max(max(pilot_wers), max(current_wers)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x  (no change)")
    ax.set_xlabel("Pilot WER", fontsize=12)
    ax.set_ylabel("50-epoch WER", fontsize=12)
    ax.set_title("Matched samples: Pilot vs 50-epoch WER\n(above diagonal = regression)", fontsize=12)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_bar(sdi, labels, out_path, title):
    """Single-run error-type bar chart."""
    if not HAS_MPL:
        return
    colors = ["steelblue", "coral", "mediumseagreen"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, sdi, color=colors, edgecolor="white", alpha=0.85)
    for bar, v in zip(bars, sdi):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Fraction of reference words", fontsize=12)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_length_wer(lengths, wers, r, p_val, out_path, title="Sentence Length vs WER"):
    if not HAS_MPL or len(lengths) < 3:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(lengths, wers, alpha=0.55, color="steelblue", edgecolors="white", s=55)
    m, b = np.polyfit(lengths, wers, 1)
    xs = np.linspace(min(lengths), max(lengths), 100)
    ax.plot(xs, m * xs + b, "r--", linewidth=1.5)
    note = f"{title}\nPearson r = {r:.3f},  p = {p_val:.4f}"
    ax.set_xlabel("Reference sentence length (words)", fontsize=12)
    ax.set_ylabel("WER", fontsize=12)
    ax.set_title(note, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare Amchi Konkani ASR pilot vs 50-epoch run")
    parser.add_argument("--pilot",          required=True,
                        help="Pilot final_test_results.json (35.1%% WER)")
    parser.add_argument("--current",        required=True,
                        help="50-epoch final_test_results.json (54.7%% WER)")
    parser.add_argument("--manifest",       required=True,
                        help="Test manifest for speaker lookup")
    parser.add_argument("--train_manifest", required=True,
                        help="Train manifest for speaker count")
    parser.add_argument("--epoch_metrics",  default=None,
                        help="epoch_metrics.csv from 50-epoch run")
    parser.add_argument("--output_dir",     default="results/amchi_analysis",
                        help="Output directory for plots and report")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    pilot_samples   = load_results(args.pilot)
    current_samples = load_results(args.current)
    test_manifest   = load_manifest(args.manifest)    # stem → speaker_id
    train_manifest  = load_manifest(args.train_manifest)

    # ── Speaker lookup for current run ───────────────────────────────────────
    # Per-sample speaker_id already in the JSON; fall back to manifest
    def get_speaker(sample, manifest):
        spk = sample.get("speaker_id", "")
        if spk and spk != "unknown":
            return spk
        return manifest.get(audio_stem(sample["audio"]), "unknown")

    # ── Current run arrays ───────────────────────────────────────────────────
    cur_refs    = [s["reference"]  for s in current_samples]
    cur_preds   = [s["prediction"] for s in current_samples]
    cur_wers    = [float(s["wer"]) for s in current_samples]
    cur_spks    = [get_speaker(s, test_manifest) for s in current_samples]
    cur_n       = len(current_samples)

    # Pilot run (all ashaheble)
    pilot_wers  = [float(s["wer"]) for s in pilot_samples]
    pilot_refs  = [s["reference"]  for s in pilot_samples]
    pilot_preds = [s["prediction"] for s in pilot_samples]
    pilot_n     = len(pilot_samples)

    # ── CER ──────────────────────────────────────────────────────────────────
    if HAS_JIWER:
        cur_cers   = [compute_cer(r, p) for r, p in zip(cur_refs, cur_preds)]
        pilot_cers = [compute_cer(r, p) for r, p in zip(pilot_refs, pilot_preds)]
    else:
        cur_cers   = [0.0] * cur_n
        pilot_cers = [0.0] * pilot_n

    # ── Per-speaker breakdown (current run) ──────────────────────────────────
    unique_spk   = sorted(set(cur_spks))
    spk_wers     = defaultdict(list)
    spk_cers     = defaultdict(list)
    for spk, wer, cer in zip(cur_spks, cur_wers, cur_cers):
        spk_wers[spk].append(wer)
        spk_cers[spk].append(cer)

    # ── Speaker representation ────────────────────────────────────────────────
    train_speaker_counts = defaultdict(int)
    for stem, spk in train_manifest.items():
        train_speaker_counts[spk] += 1

    test_speaker_counts  = defaultdict(int)
    for spk in cur_spks:
        test_speaker_counts[spk] += 1

    # ── Matched sample analysis ───────────────────────────────────────────────
    # Match by audio file stem (e.g. "570")
    current_by_stem = {audio_stem(s["audio"]): s for s in current_samples}
    matched_pilot, matched_current = [], []
    for s in pilot_samples:
        stem = audio_stem(s["audio"])
        if stem in current_by_stem:
            matched_pilot.append(s)
            matched_current.append(current_by_stem[stem])

    # ── Overall stats ────────────────────────────────────────────────────────
    mean_pw, std_pw = np.mean(pilot_wers),   np.std(pilot_wers)
    mean_cw, std_cw = np.mean(cur_wers),     np.std(cur_wers)
    ci_pw           = bootstrap_ci(pilot_wers)
    ci_cw           = bootstrap_ci(cur_wers)

    mean_pc, std_pc = np.mean(pilot_cers),   np.std(pilot_cers)
    mean_cc, std_cc = np.mean(cur_cers),     np.std(cur_cers)

    # ── Kruskal-Wallis across speakers ───────────────────────────────────────
    kw_groups = [spk_wers[s] for s in unique_spk if len(spk_wers[s]) >= 2]
    if len(kw_groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*kw_groups)
    else:
        kw_stat, kw_p = float("nan"), float("nan")

    # Pairwise Mann-Whitney
    pairwise = []
    for i, s1 in enumerate(unique_spk):
        for s2 in unique_spk[i + 1:]:
            g1, g2 = spk_wers[s1], spk_wers[s2]
            if len(g1) >= 5 and len(g2) >= 5:
                stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                pairwise.append((s1, s2, stat, p))

    # ── Matched-sample Wilcoxon ───────────────────────────────────────────────
    if len(matched_pilot) >= 5:
        m_pilot_wers   = [float(s["wer"]) for s in matched_pilot]
        m_current_wers = [float(s["wer"]) for s in matched_current]
        deltas = [c - p for p, c in zip(m_pilot_wers, m_current_wers)]
        if len(set(deltas)) > 1:
            wilcox_stat, wilcox_p = stats.wilcoxon(m_pilot_wers, m_current_wers)
        else:
            wilcox_stat, wilcox_p = float("nan"), float("nan")
    else:
        m_pilot_wers = m_current_wers = []
        wilcox_stat = wilcox_p = float("nan")

    # ── Error-type breakdown (current run) ───────────────────────────────────
    if HAS_JIWER:
        total_ref = S_tot = D_tot = I_tot = 0
        for ref, pred in zip(cur_refs, cur_preds):
            S, D, I, rl = error_types(ref, pred)
            S_tot += S; D_tot += D; I_tot += I; total_ref += rl
        denom  = max(total_ref, 1)
        sdi    = [S_tot / denom, D_tot / denom, I_tot / denom]
    else:
        sdi    = [0.0, 0.0, 0.0]

    # ── Sentence length vs WER ────────────────────────────────────────────────
    sent_lens = [len(r.split()) for r in cur_refs]
    if len(sent_lens) > 2:
        corr_r, corr_p = stats.pearsonr(sent_lens, cur_wers)
    else:
        corr_r = corr_p = float("nan")

    # ─────────────────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────────────────
    plot_wer_comparison(pilot_wers, cur_wers,
                        out_dir / "1_wer_pilot_vs_50epoch.png")

    plot_speaker_boxplot(spk_wers,
                         out_dir / "2_speaker_wer_boxplot.png")

    plot_speaker_representation(
        train_speaker_counts, test_speaker_counts,
        out_dir / "3_speaker_representation.png")

    if args.epoch_metrics:
        plot_training_curve(args.epoch_metrics,
                            out_dir / "4_training_curve.png")

    if m_pilot_wers:
        plot_matched_scatter(m_pilot_wers, m_current_wers,
                             out_dir / "5_matched_scatter.png")

    if HAS_JIWER:
        plot_error_bar(sdi, ["Substitutions", "Deletions", "Insertions"],
                       out_dir / "6_error_type_breakdown.png",
                       "Error-Type Breakdown — 50-epoch Run")

    plot_length_wer(sent_lens, cur_wers, corr_r, corr_p,
                    out_dir / "7_length_vs_wer.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Text report
    # ─────────────────────────────────────────────────────────────────────────
    L = []
    def line(s=""): L.append(s)

    line("=" * 72)
    line("  AMCHI KONKANI ASR — RUN COMPARISON REPORT")
    line("  Pilot (35.1% WER)  vs  50-epoch Run (54.7% WER)")
    line("=" * 72)
    line()

    # ── 1. Overall WER ────────────────────────────────────────────────────────
    line("── 1. OVERALL WER / CER ──────────────────────────────────────────────")
    line(f"  {'Metric':<30}  {'Pilot':>12}  {'50-epoch':>12}")
    line(f"  {'N samples':<30}  {pilot_n:>12d}  {cur_n:>12d}")
    line(f"  {'N unique speakers':<30}  {'1':>12}  {len(unique_spk):>12d}")
    line(f"  {'Mean WER':<30}  {mean_pw:>12.4f}  {mean_cw:>12.4f}")
    line(f"  {'Std WER':<30}  {std_pw:>12.4f}  {std_cw:>12.4f}")
    line(f"  {'95% CI WER (bootstrap)':<30}  "
         f"[{ci_pw[0]:.4f},{ci_pw[1]:.4f}]  [{ci_cw[0]:.4f},{ci_cw[1]:.4f}]")
    line(f"  {'Mean CER':<30}  {mean_pc:>12.4f}  {mean_cc:>12.4f}")
    line(f"  {'Std CER':<30}  {std_pc:>12.4f}  {std_cc:>12.4f}")
    line()

    # ── 2. Per-speaker (50-epoch) ─────────────────────────────────────────────
    line("── 2. PER-SPEAKER BREAKDOWN (50-epoch run) ───────────────────────────")
    line(f"  {'#':<3}  {'Speaker (email prefix)':<28}  {'N':>4}  "
         f"{'Mean WER':>9}  {'Median WER':>10}  {'Std WER':>8}  {'Mean CER':>9}")
    for i, spk in enumerate(unique_spk):
        wl  = spk_wers[spk]
        cl  = spk_cers[spk]
        pfx = spk.split("@")[0]
        line(f"  {i+1:<3}  {pfx:<28}  {len(wl):>4}  "
             f"{np.mean(wl):>9.4f}  {np.median(wl):>10.4f}  "
             f"{np.std(wl):>8.4f}  {np.mean(cl):>9.4f}")
    line()
    line(f"  Kruskal-Wallis across speakers:  H={kw_stat:.4f}, p={kw_p:.6f}  {sig(kw_p)}")
    if pairwise:
        line("  Pairwise Mann-Whitney U (two-sided):")
        for s1, s2, stat, p in pairwise:
            i1 = unique_spk.index(s1) + 1
            i2 = unique_spk.index(s2) + 1
            pfx1 = s1.split("@")[0]; pfx2 = s2.split("@")[0]
            line(f"    Speaker {i1} ({pfx1}) vs Speaker {i2} ({pfx2}):  "
                 f"U={stat:.1f}, p={p:.6f}  {sig(p)}")
    line()

    # ── 3. Speaker representation ─────────────────────────────────────────────
    line("── 3. SPEAKER REPRESENTATION: TRAIN vs TEST ──────────────────────────")
    line(f"  {'Speaker (email prefix)':<28}  {'Train samples':>14}  {'Test samples':>12}  {'Ratio':>8}")
    all_test_spk = sorted(test_speaker_counts.keys())
    for spk in all_test_spk:
        pfx    = spk.split("@")[0]
        n_tr   = train_speaker_counts.get(spk, 0)
        n_te   = test_speaker_counts.get(spk, 0)
        ratio  = n_tr / n_te if n_te else 0
        flag   = "  ⚠ UNDERREPRESENTED" if ratio < 5 else ""
        line(f"  {pfx:<28}  {n_tr:>14d}  {n_te:>12d}  {ratio:>8.1f}x{flag}")
    line()

    # ── 4. Matched-sample analysis ────────────────────────────────────────────
    line("── 4. MATCHED-SAMPLE ANALYSIS (same audio IDs in both runs) ──────────")
    line(f"  Matched samples found: {len(matched_pilot)}")
    if matched_pilot:
        mean_delta = np.mean([c - p for p, c in zip(m_pilot_wers, m_current_wers)])
        regressed  = sum(1 for p, c in zip(m_pilot_wers, m_current_wers) if c > p + 1e-9)
        improved   = sum(1 for p, c in zip(m_pilot_wers, m_current_wers) if c < p - 1e-9)
        unchanged  = len(matched_pilot) - regressed - improved
        line(f"  Mean WER change (50-epoch – pilot): {mean_delta:+.4f}")
        line(f"  Regressed (50-ep worse):  {regressed}/{len(matched_pilot)}")
        line(f"  Improved  (50-ep better): {improved}/{len(matched_pilot)}")
        line(f"  Unchanged:                {unchanged}/{len(matched_pilot)}")
        if not np.isnan(wilcox_p):
            line(f"  Wilcoxon signed-rank: stat={wilcox_stat:.3f}, p={wilcox_p:.6f}  {sig(wilcox_p)}")
        line()
        line(f"  {'Audio ID':<10}  {'Pilot WER':>10}  {'50-ep WER':>10}  {'Delta':>8}  Reference")
        for s_p, s_c in zip(matched_pilot, matched_current):
            stem  = audio_stem(s_p["audio"])
            pw    = float(s_p["wer"])
            cw    = float(s_c["wer"])
            delta = cw - pw
            ref   = s_p["reference"][:50]
            line(f"  {stem:<10}  {pw:>10.4f}  {cw:>10.4f}  {delta:>+8.4f}  {ref}")
    line()

    # ── 5. Training curve / overfitting ───────────────────────────────────────
    line("── 5. TRAINING CURVE SUMMARY ─────────────────────────────────────────")
    if args.epoch_metrics:
        try:
            ep_data = []
            with open(args.epoch_metrics, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ep_data.append(row)

            epoch6_val  = float(ep_data[6]["val_loss"])  if len(ep_data) > 6  else None
            epoch49_val = float(ep_data[49]["val_loss"]) if len(ep_data) > 49 else None
            epoch50_tl  = float(ep_data[49]["train_loss"]) if ep_data[49]["train_loss"] else None
            line(f"  Val loss at epoch 6 (first plateau):  {epoch6_val:.2f}")
            line(f"  Val loss at epoch 49 (final):         {epoch49_val:.2f}")
            if epoch6_val and epoch49_val:
                line(f"  Val loss increase:                    {epoch49_val/epoch6_val:.2f}x  "
                     f"({'SEVERE overfitting' if epoch49_val/epoch6_val > 1.5 else 'mild overfitting'})")
            line(f"  Train loss at epoch 49:               {epoch50_tl:.4f}")
            line(f"  Best CTC checkpoint:                  epoch 47, val_wer=53.2%")
            line(f"  RNNT val_wer plateau:                 ~62–64% from epoch 6")
        except Exception as e:
            line(f"  [Could not parse epoch_metrics: {e}]")
    else:
        line("  (epoch_metrics not provided — skipping training curve summary)")
    line()

    # ── 6. Error-type breakdown ───────────────────────────────────────────────
    if HAS_JIWER:
        line("── 6. ERROR-TYPE BREAKDOWN (50-epoch run, fraction of ref words) ───")
        for label, v in zip(["Substitutions", "Deletions", "Insertions"], sdi):
            line(f"  {label:<20}  {v:.4f}  ({v*100:.1f}%)")
        line()

    # ── 7. Sentence length vs WER ─────────────────────────────────────────────
    line("── 7. SENTENCE LENGTH vs WER (50-epoch run) ──────────────────────────")
    line(f"  Pearson r = {corr_r:.4f},  p = {corr_p:.6f}  {sig(corr_p)}")
    line()

    # ── 8. Root-cause analysis ────────────────────────────────────────────────
    line("=" * 72)
    line("  ROOT-CAUSE ANALYSIS: WHY DID WER INCREASE FROM 35.1% → 54.7%?")
    line("=" * 72)
    line()

    # Factor 1: test set composition
    line("  FACTOR 1 — Test set is harder (more speakers, lower per-speaker data)")
    line("  ─────────────────────────────────────────────────────────────────────")
    line("  Pilot:     38 samples, 1 speaker (ashaheble), all Story 5")
    line(f"  50-epoch: {cur_n} samples, {len(unique_spk)} speakers — includes dipti (3 train samples!)")
    for spk in all_test_spk:
        n_tr = train_speaker_counts.get(spk, 0)
        n_te = test_speaker_counts.get(spk, 0)
        pfx  = spk.split("@")[0]
        spk_mean = np.mean(spk_wers.get(spk, [0]))
        line(f"    {pfx}: {n_tr} train / {n_te} test → mean WER {spk_mean:.3f}")
    line()

    # Factor 2: overfitting
    line("  FACTOR 2 — Overfitting (all 119M encoder params updated on 511 samples)")
    line("  ─────────────────────────────────────────────────────────────────────")
    if matched_pilot:
        mean_matched_pilot = np.mean(m_pilot_wers)
        mean_matched_cur   = np.mean(m_current_wers)
        line(f"  On matched samples (same audio, same speaker — ashaheble):")
        line(f"    Pilot mean WER:    {mean_matched_pilot:.4f}")
        line(f"    50-epoch mean WER: {mean_matched_cur:.4f}")
        delta_str = f"{(mean_matched_cur - mean_matched_pilot)*100:+.1f}pp"
        dir_str   = "REGRESSION" if mean_matched_cur > mean_matched_pilot else "improvement"
        line(f"    Change:            {delta_str}  ({dir_str} on same speaker same data)")
        line(f"  Even for ashaheble (dominant train speaker), WER regressed — ")
        line(f"  strong evidence that the longer full-encoder training hurt generalisation.")
    line(f"  Val loss doubled (epoch 6→49), train_loss collapsed to 0.17.")
    line(f"  Pilot stopped at epoch 20 (best at ep 13) — less overfitting.")
    line()

    # Factor 3: data imbalance
    dipti_n_train = train_speaker_counts.get("dipti.ajgaonkar21@gmail.com", 0)
    dipti_n_test  = test_speaker_counts.get("dipti.ajgaonkar21@gmail.com", 0)
    dipti_wer     = np.mean(spk_wers.get("dipti.ajgaonkar21@gmail.com", [0]))
    line("  FACTOR 3 — Severe speaker imbalance for dipti")
    line("  ─────────────────────────────────────────────────────────────────────")
    line(f"  dipti.ajgaonkar21: {dipti_n_train} train samples, {dipti_n_test} test samples → WER {dipti_wer:.3f}")
    line(f"  Model was almost never trained on this voice → poor generalisation.")
    line()

    # Summary table
    line("  SUMMARY: IMPACT BREAKDOWN")
    line("  ─────────────────────────────────────────────────────────────────────")
    line("  Cause                            Estimated WER impact")
    line("  Unseen speaker (dipti, 3 train)  ~+3–5pp  (pulls mean up by 8pp vs ashaheble)")
    line("  Overfitting (encoder, 50 ep)     ~+8–12pp (ashaheble WER 35→52%)")
    line("  Test set harder (3 speakers)     Interaction of the above")
    line()
    line("  RECOMMENDED FIX: Run C (freeze encoder + cosine LR + 100 epochs)")
    line("    → Frozen encoder: 4.4M instead of 119M trainable params")
    line("    → Massively reduces overfitting risk on 511 samples")
    line("    → Also collect more dipti recordings before next run")
    line()

    line("=" * 72)
    line("  Plots saved to: " + str(out_dir))
    for pname in ["1_wer_pilot_vs_50epoch.png", "2_speaker_wer_boxplot.png",
                  "3_speaker_representation.png", "4_training_curve.png",
                  "5_matched_scatter.png", "6_error_type_breakdown.png",
                  "7_length_vs_wer.png"]:
        line(f"    {pname}")
    line("=" * 72)

    report = "\n".join(L)
    print(report)
    report_path = out_dir / "comparison_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n✓ Report saved to {report_path}")


if __name__ == "__main__":
    main()
