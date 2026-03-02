#!/usr/bin/env python3
"""
Statistical analysis of Amchi Konkani ASR results.

Computes:
  1. WER and CER histograms (raw ASR vs post-processed, overlaid)
  2. Per-speaker WER/CER breakdown — box plots + mean/std table
  3. Kruskal-Wallis test: is WER significantly different across speakers?
  4. Wilcoxon signed-rank test: does post-processing significantly improve WER?
  5. Cohen's d effect size for post-processing improvement
  6. % improved / unchanged / degraded by post-processing
  7. Bootstrap 95% CI for mean WER and CER
  8. Error type decomposition (substitutions, deletions, insertions)
  9. Sentence length vs WER scatter + Pearson correlation
  10. Raw vs post-processed WER scatter (below diagonal = improved)
  11. Summary report (text) + all plots saved to output_dir

Usage:
  python3 scripts/analyze_results.py \\
    --results  nemo_experiments/amchi_konkani_50epoch/experiments/<ts>/postprocessed_results.json \\
    --manifest data/amchi/test/manifest.jsonl \\
    --output_dir results/amchi_analysis/

Requirements:
  pip install numpy scipy matplotlib jiwer
"""

import argparse
import json
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
    print("[WARN] matplotlib not found — plots will be skipped. pip install matplotlib")

try:
    from jiwer import wer as compute_wer, cer as compute_cer, process_words
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    print("[WARN] jiwer not found — CER/error types skipped. pip install jiwer")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(data, n=10_000, ci=0.95, seed=42):
    """Return (lo, hi) bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n)]
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lo, hi


def error_types(reference, hypothesis):
    """Return (S, D, I, ref_len) edit-operation counts for one sentence pair."""
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


def load_manifest_speakers(manifest_path):
    """Return {audio_filepath: speaker_id} from the test manifest."""
    mapping = {}
    try:
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                fp = entry.get("audio_filepath", "")
                spk = entry.get("speaker_id", "unknown")
                mapping[fp] = spk
    except FileNotFoundError:
        print(f"[WARN] Manifest not found: {manifest_path}. Speaker analysis will use 'unknown'.")
    return mapping


def short_speaker_label(speaker_id: str, idx: int) -> str:
    """Shorten an email-style speaker_id to 'Speaker N' for display."""
    return f"Speaker {idx + 1}"


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_histogram(raw, post, xlabel, title, path, bins=20):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(raw,  bins=bins, alpha=0.6, color="steelblue", label="Raw ASR",   edgecolor="white")
    ax.hist(post, bins=bins, alpha=0.6, color="coral",     label="Post-proc", edgecolor="white")
    ax.axvline(np.mean(raw),  color="steelblue", linestyle="--", linewidth=1.5,
               label=f"Mean raw  = {np.mean(raw):.3f}")
    ax.axvline(np.mean(post), color="coral",     linestyle="--", linewidth=1.5,
               label=f"Mean post = {np.mean(post):.3f}")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_speaker_boxplot(speaker_raw, speaker_post, speaker_labels, ylabel, title, path):
    if not HAS_MPL:
        return
    n_spk = len(speaker_labels)
    fig, ax = plt.subplots(figsize=(max(6, n_spk * 3), 5))
    gap = 2.5
    pos_raw  = [i * gap        for i in range(n_spk)]
    pos_post = [i * gap + 0.85 for i in range(n_spk)]

    bp1 = ax.boxplot([speaker_raw[s]  for s in speaker_labels], positions=pos_raw,  widths=0.7,
                     patch_artist=True,
                     boxprops=dict(facecolor="steelblue", alpha=0.7),
                     medianprops=dict(color="navy", linewidth=2))
    bp2 = ax.boxplot([speaker_post[s] for s in speaker_labels], positions=pos_post, widths=0.7,
                     patch_artist=True,
                     boxprops=dict(facecolor="coral", alpha=0.7),
                     medianprops=dict(color="darkred", linewidth=2))

    ax.set_xticks([i * gap + 0.425 for i in range(n_spk)])
    ax.set_xticklabels(
        [f"Speaker {i + 1}\n(n={len(speaker_raw[s])})" for i, s in enumerate(speaker_labels)],
        fontsize=11
    )
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    raw_patch  = mpatches.Patch(color="steelblue", alpha=0.7, label="Raw ASR")
    post_patch = mpatches.Patch(color="coral",     alpha=0.7, label="Post-proc")
    ax.legend(handles=[raw_patch, post_patch])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_scatter(x, y, xlabel, ylabel, title, path, corr_r=None, corr_p=None):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.6, color="steelblue", edgecolors="white", s=60)
    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 100)
        ax.plot(xs, m * xs + b, "r--", linewidth=1.5)
    note = title
    if corr_r is not None and not np.isnan(corr_r):
        note += f"\nPearson r = {corr_r:.3f},  p = {corr_p:.4f}"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(note, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_error_type_bar(raw_sdi, post_sdi, path):
    """raw_sdi / post_sdi are (S, D, I) as fractions of total reference words."""
    if not HAS_MPL:
        return
    labels = ["Substitutions", "Deletions", "Insertions"]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, raw_sdi,  w, label="Raw ASR",   color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, post_sdi, w, label="Post-proc", color="coral",     alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Fraction of reference words", fontsize=12)
    ax.set_title("Error Type Breakdown: Raw ASR vs Post-processed", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_improvement_bar(improved, unchanged, degraded, path):
    if not HAS_MPL:
        return
    total = improved + unchanged + degraded
    fracs = [improved / total * 100, unchanged / total * 100, degraded / total * 100]
    labels = [f"Improved\n({improved})", f"Unchanged\n({unchanged})", f"Degraded\n({degraded})"]
    colors = ["mediumseagreen", "lightgray", "tomato"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, fracs, color=colors, edgecolor="white", linewidth=0.5)
    for bar, frac in zip(bars, fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{frac:.1f}%", ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("Percentage of samples (%)", fontsize=12)
    ax.set_title("Post-processing Impact per Sample", fontsize=13)
    ax.set_ylim(0, max(fracs) * 1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_raw_vs_post_scatter(raw_wer, post_wer, path):
    """Scatter: x=raw WER, y=post WER. Points below y=x diagonal improved."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(raw_wer, post_wer, alpha=0.6, color="steelblue", edgecolors="white", s=60)
    lim = max(max(raw_wer), max(post_wer)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x  (no change)")
    ax.set_xlabel("Raw ASR WER", fontsize=12)
    ax.set_ylabel("Post-processed WER", fontsize=12)
    ax.set_title("Raw vs Post-processed WER\n(below diagonal = improved)", fontsize=13)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of Konkani ASR results")
    parser.add_argument("--results",    required=True,
                        help="postprocessed_results.json produced by postprocess_asr.py")
    parser.add_argument("--manifest",   required=True,
                        help="data/amchi/test/manifest.jsonl (should include speaker_id field)")
    parser.add_argument("--output_dir", default="results/amchi_analysis",
                        help="Directory for plots and report (created if absent)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load results JSON ────────────────────────────────────────────────────
    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    samples = data["per_sample"]
    n = len(samples)

    refs       = [s["reference"]                        for s in samples]
    raw_preds  = [s["prediction"]                       for s in samples]
    post_preds = [s.get("corrected", s["prediction"])   for s in samples]
    raw_wer    = [float(s["wer_before"])                for s in samples]
    post_wer   = [float(s["wer_after"])                 for s in samples]
    audio_fps  = [s["audio"]                            for s in samples]

    # ── Speaker lookup from manifest ─────────────────────────────────────────
    speaker_map = load_manifest_speakers(args.manifest)
    speakers    = [speaker_map.get(fp, "unknown") for fp in audio_fps]
    unique_spk  = sorted(set(speakers))

    # ── CER per sample ───────────────────────────────────────────────────────
    if HAS_JIWER:
        raw_cer  = [compute_cer(r, p) for r, p in zip(refs, raw_preds)]
        post_cer = [compute_cer(r, p) for r, p in zip(refs, post_preds)]
    else:
        raw_cer = post_cer = [0.0] * n

    # ── Overall summary stats ─────────────────────────────────────────────────
    mean_rw, std_rw   = np.mean(raw_wer),  np.std(raw_wer)
    mean_pw, std_pw   = np.mean(post_wer), np.std(post_wer)
    mean_rc, std_rc   = np.mean(raw_cer),  np.std(raw_cer)
    mean_pc, std_pc   = np.mean(post_cer), np.std(post_cer)
    ci_rw  = bootstrap_ci(raw_wer)
    ci_pw  = bootstrap_ci(post_wer)
    ci_rc  = bootstrap_ci(raw_cer)
    ci_pc  = bootstrap_ci(post_cer)

    # ── Statistical tests: raw vs post WER (Wilcoxon signed-rank) ────────────
    wer_deltas = [r - p for r, p in zip(raw_wer, post_wer)]
    if len(set(wer_deltas)) > 1:
        wilcox_stat, wilcox_p = stats.wilcoxon(raw_wer, post_wer)
    else:
        wilcox_stat, wilcox_p = float("nan"), float("nan")
    pooled_std = np.sqrt((std_rw ** 2 + std_pw ** 2) / 2) or 1e-9
    cohens_d   = (mean_rw - mean_pw) / pooled_std

    # Improvement counts
    improved  = sum(1 for d in wer_deltas if d >  1e-9)
    degraded  = sum(1 for d in wer_deltas if d < -1e-9)
    unchanged = n - improved - degraded

    # ── Per-speaker breakdown ─────────────────────────────────────────────────
    spk_raw_wer  = defaultdict(list)
    spk_post_wer = defaultdict(list)
    spk_raw_cer  = defaultdict(list)
    for spk, rw, pw, rc in zip(speakers, raw_wer, post_wer, raw_cer):
        spk_raw_wer[spk].append(rw)
        spk_post_wer[spk].append(pw)
        spk_raw_cer[spk].append(rc)

    # Kruskal-Wallis across speakers
    kw_groups = [spk_raw_wer[s] for s in unique_spk if len(spk_raw_wer[s]) >= 2]
    if len(kw_groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*kw_groups)
    else:
        kw_stat, kw_p = float("nan"), float("nan")

    # Pairwise Wilcoxon between speakers (if >= 2 speakers with >= 5 samples each)
    pairwise = []
    for i, s1 in enumerate(unique_spk):
        for s2 in unique_spk[i + 1:]:
            g1, g2 = spk_raw_wer[s1], spk_raw_wer[s2]
            if len(g1) >= 5 and len(g2) >= 5:
                stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                pairwise.append((s1, s2, stat, p))

    # ── Error type decomposition ──────────────────────────────────────────────
    if HAS_JIWER:
        total_ref = raw_S = raw_D = raw_I = post_S = post_D = post_I = 0
        for ref, rp, pp in zip(refs, raw_preds, post_preds):
            S, D, I, rl = error_types(ref, rp)
            raw_S += S; raw_D += D; raw_I += I; total_ref += rl
            S2, D2, I2, _ = error_types(ref, pp)
            post_S += S2; post_D += D2; post_I += I2
        denom = max(total_ref, 1)
        raw_sdi  = [raw_S  / denom, raw_D  / denom, raw_I  / denom]
        post_sdi = [post_S / denom, post_D / denom, post_I / denom]
    else:
        raw_sdi = post_sdi = [0.0, 0.0, 0.0]

    # ── Sentence length vs WER ────────────────────────────────────────────────
    sent_len = [len(r.split()) for r in refs]
    if len(sent_len) > 2:
        corr_r, corr_p = stats.pearsonr(sent_len, raw_wer)
    else:
        corr_r = corr_p = float("nan")

    # ─────────────────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────────────────
    save_histogram(raw_wer, post_wer, "WER",
                   "WER Distribution: Raw ASR vs Post-processed",
                   out_dir / "wer_histogram.png")

    save_histogram(raw_cer, post_cer, "CER",
                   "CER Distribution: Raw ASR vs Post-processed",
                   out_dir / "cer_histogram.png")

    save_speaker_boxplot(spk_raw_wer, spk_post_wer, unique_spk,
                         "WER", "WER by Speaker (Raw ASR vs Post-processed)",
                         out_dir / "speaker_wer_boxplot.png")

    save_raw_vs_post_scatter(raw_wer, post_wer,
                             out_dir / "raw_vs_post_scatter.png")

    save_error_type_bar(raw_sdi, post_sdi,
                        out_dir / "error_type_breakdown.png")

    save_improvement_bar(improved, unchanged, degraded,
                         out_dir / "improvement_breakdown.png")

    save_scatter(sent_len, raw_wer,
                 "Sentence Length (words)", "Raw ASR WER",
                 "Sentence Length vs WER",
                 out_dir / "length_vs_wer.png",
                 corr_r=corr_r, corr_p=corr_p)

    # ─────────────────────────────────────────────────────────────────────────
    # Text report
    # ─────────────────────────────────────────────────────────────────────────
    def sig(p, threshold=0.05):
        return "*** significant (p<0.05)" if p < threshold else "(not significant)"

    def effect_label(d):
        d = abs(d)
        return "large" if d > 0.8 else "medium" if d > 0.5 else "small"

    lines = []
    L = lines.append

    L("=" * 72)
    L("  AMCHI KONKANI ASR — STATISTICAL ANALYSIS REPORT")
    L("=" * 72)
    L(f"  Samples analysed : {n}")
    L(f"  Speakers found   : {len(unique_spk)}")
    L("")

    L("── 1. OVERALL WER / CER ──────────────────────────────────────────────")
    L(f"  {'Metric':<32}  {'Raw ASR':>12}  {'Post-proc':>12}")
    L(f"  {'Mean WER':<32}  {mean_rw:12.4f}  {mean_pw:12.4f}")
    L(f"  {'Std WER':<32}  {std_rw:12.4f}  {std_pw:12.4f}")
    L(f"  {'95% CI WER (bootstrap)':<32}  "
      f"[{ci_rw[0]:.4f}, {ci_rw[1]:.4f}]  [{ci_pw[0]:.4f}, {ci_pw[1]:.4f}]")
    L(f"  {'Mean CER':<32}  {mean_rc:12.4f}  {mean_pc:12.4f}")
    L(f"  {'Std CER':<32}  {std_rc:12.4f}  {std_pc:12.4f}")
    L(f"  {'95% CI CER (bootstrap)':<32}  "
      f"[{ci_rc[0]:.4f}, {ci_rc[1]:.4f}]  [{ci_pc[0]:.4f}, {ci_pc[1]:.4f}]")
    L("")

    L("── 2. RAW vs POST-PROCESSED: SIGNIFICANCE ────────────────────────────")
    L(f"  Wilcoxon signed-rank stat : {wilcox_stat:.4f}")
    L(f"  Wilcoxon p-value          : {wilcox_p:.6f}  {sig(wilcox_p)}")
    L(f"  Cohen's d (effect size)   : {cohens_d:.4f}  ({effect_label(cohens_d)})")
    L("")

    L("── 3. POST-PROCESSING IMPACT PER SAMPLE ──────────────────────────────")
    L(f"  Improved  : {improved:3d} / {n}  ({improved / n * 100:.1f}%)")
    L(f"  Unchanged : {unchanged:3d} / {n}  ({unchanged / n * 100:.1f}%)")
    L(f"  Degraded  : {degraded:3d} / {n}  ({degraded / n * 100:.1f}%)")
    L("")

    L("── 4. PER-SPEAKER BREAKDOWN ──────────────────────────────────────────")
    L(f"  {'Speaker':<12}  {'N':>4}  {'Mean WER raw':>14}  {'Mean WER post':>14}  "
      f"{'Std WER raw':>12}  {'Mean CER raw':>14}")
    for i, spk in enumerate(unique_spk):
        rw = spk_raw_wer[spk]; pw = spk_post_wer[spk]; rc = spk_raw_cer[spk]
        L(f"  {'Speaker '+str(i+1):<12}  {len(rw):4d}  {np.mean(rw):14.4f}  "
          f"{np.mean(pw):14.4f}  {np.std(rw):12.4f}  {np.mean(rc):14.4f}")
    L(f"  Kruskal-Wallis  H = {kw_stat:.4f},  p = {kw_p:.6f}  {sig(kw_p)}")
    if pairwise:
        L("  Pairwise Mann-Whitney U (two-sided):")
        for s1, s2, stat, p in pairwise:
            i1 = unique_spk.index(s1) + 1
            i2 = unique_spk.index(s2) + 1
            L(f"    Speaker {i1} vs Speaker {i2}:  U={stat:.1f},  p={p:.6f}  {sig(p)}")
    L("")

    if HAS_JIWER:
        L("── 5. ERROR TYPE BREAKDOWN (fraction of ref words) ──────────────────")
        L(f"  {'Type':<20}  {'Raw ASR':>12}  {'Post-proc':>12}")
        for label, rv, pv in zip(["Substitutions", "Deletions", "Insertions"],
                                  raw_sdi, post_sdi):
            L(f"  {label:<20}  {rv:12.4f}  {pv:12.4f}")
        L("")

    L("── 6. SENTENCE LENGTH vs WER ─────────────────────────────────────────")
    L(f"  Pearson r = {corr_r:.4f},  p = {corr_p:.6f}  {sig(corr_p)}")
    L("")

    L("=" * 72)
    L("  Plots saved to: " + str(out_dir))
    plots = ["wer_histogram.png", "cer_histogram.png", "speaker_wer_boxplot.png",
             "raw_vs_post_scatter.png", "error_type_breakdown.png",
             "improvement_breakdown.png", "length_vs_wer.png"]
    for p in plots:
        L(f"    {p}")
    L("=" * 72)

    report = "\n".join(lines)
    print(report)

    report_path = out_dir / "analysis_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n✓ Report saved to {report_path}")
    if HAS_MPL:
        print(f"✓ Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
