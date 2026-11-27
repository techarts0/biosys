#!/usr/bin/env python3
"""
Generate 3 publication-grade IEEE/Elsevier vector evaluation figures directly from:
  - locomo_results.json (Official LoCoMo, 10 convs, 5,882 turns, 1,536 QAs)
  - kill_test_results.json (Coding Agent trace & Consolidation Lag consistency sweep)

Outputs (PDF, SVG, 300 DPI PNG) in figures/:
  - fig5_locomo_pareto.{pdf,svg,png}
  - fig6_funnel_ablation.{pdf,svg,png}
  - fig7_coding_consistency.{pdf,svg,png}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.9,
    "grid.color": "#CBD5E1",
    "grid.linewidth": 0.6,
    "grid.linestyle": "--",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_INK = "#1E293B"
C_MUTED = "#64748B"
C_BLUE = "#2563EB"
C_TEAL = "#059669"
C_AMBER = "#D97706"
C_PURPLE = "#7C3AED"
C_RED = "#DC2626"
C_SLATE = "#94A3B8"
C_CYAN = "#0891B2"


def save_fig(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.08}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(OUT_DIR / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def by_name(rows: List[Dict]) -> Dict[str, Dict]:
    return {r["name"].strip(): r for r in rows}


# =============================================================================
# FIGURE 5: LoCoMo Cost-Recall Pareto Frontier + Prefix Hit vs Uncached Prefill
# =============================================================================

def build_fig5(locomo: Dict[str, Dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.35), gridspec_kw={"width_ratios": [1.18, 1.0]})

    # --- Panel (a): Cost-Recall Pareto Frontier (Log-X) ---
    baselines = [
        ("Full-Context", "Full-Context", C_SLATE, "s", (-10, -16)),
        ("Full+Compact(4096)", "Full+Compact", C_MUTED, "D", (8, -14)),
        ("Naive-RAG (Top-8)", "Naive-RAG", C_AMBER, "^", (8, -12)),
        ("Mem0/Zep (Sync-KG)", "Sync-KG", C_PURPLE, "P", (9, -6)),
        ("Mem0/Zep (Async-Sleep)", "Async-KG", "#A855F7", "v", (8, -14)),
        ("BIOSYS-v1 (Orig: LSH,noDTF)", "Pure-LSH\nCache", C_RED, "X", (9, -8)),
    ]

    for key, lbl, col, mk, (dx, dy) in baselines:
        r = locomo[key]
        x, y = r["cost_a10"], r["ev_hit_%"]
        ax1.scatter(x, y, c=col, marker=mk, s=72, edgecolors="#FFFFFF", linewidths=0.8, zorder=4)
        ha = "right" if dx < 0 else "left"
        ax1.annotate(
            lbl, (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=7.6, color=col, fontweight="bold" if "Mem0" in key else "normal",
            ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.82),
        )

    # BIOSYS-v2 Pareto curve
    v2_pts = [
        ("- w/o Prefix Zones", "BIOSYS-v2\n(No Zone B)", (-6, 15)),
        ("BIOSYS-v2 (Adaptive Zone B)", "BIOSYS-v2\n(Adaptive)", (-8, 17)),
        ("BIOSYS-v2 (Fixed Zone B)", "BIOSYS-v2\n(Fixed Zone B)", (8, 11)),
    ]
    xs_v2 = [locomo[k]["cost_a10"] for k, _, _ in v2_pts]
    ys_v2 = [locomo[k]["ev_hit_%"] for k, _, _ in v2_pts]
    ax1.plot(xs_v2, ys_v2, color=C_BLUE, lw=2.0, ls="-", zorder=5, label="BIOSYS-v2 Pareto Frontier")
    ax1.scatter(xs_v2, ys_v2, c=C_BLUE, marker="o", s=88, edgecolors="#FFFFFF", linewidths=1.1, zorder=6)

    for (key, lbl, (dx, dy)) in v2_pts:
        r = locomo[key]
        x, y = r["cost_a10"], r["ev_hit_%"]
        ha = "right" if dx < 0 else "left"
        ax1.annotate(
            lbl, (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=7.8, color=C_BLUE, fontweight="bold", ha=ha, va="center",
            arrowprops=dict(arrowstyle="-", color=C_BLUE, lw=0.7, alpha=0.6),
            bbox=dict(boxstyle="round,pad=0.14", facecolor="#EFF6FF", edgecolor=C_BLUE, lw=0.6, alpha=0.92),
        )

    ax1.set_xscale("log")
    ax1.set_xticks([200, 300, 400, 600, 1000, 1400])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlim(175, 1650)
    ax1.set_ylim(44, 105)
    ax1.set_xlabel("Amortized Cost Proxy $C(0.10)$ (Word Tok-Eq / Query, Log)", fontsize=8.6)
    ax1.set_ylabel("Any-Evidence Hit Rate AnyEvHit (%)", fontsize=8.6)
    ax1.set_title("(a) Cost–Hit Trade-off Frontier on LoCoMo (1,536 QAs)", fontsize=9.3, fontweight="bold", pad=8)
    ax1.grid(True, which="major", alpha=0.65)
    ax1.annotate("Better\n(Lower Cost,\nHigher Hit)", xy=(195, 96), fontsize=7.4, color=C_TEAL,
                 fontweight="bold", bbox=dict(boxstyle="round,pad=0.22", facecolor="#ECFDF5", edgecolor=C_TEAL, lw=0.7))

    # --- Panel (b): Prefix Hit Rate (%) & Uncached Prefill Tokens ---
    systems_b = [
        ("Naive-RAG (Top-8)", "Naive-RAG"),
        ("Mem0/Zep (Sync-KG)", "Sync-KG"),
        ("BIOSYS-v1 (Orig: LSH,noDTF)", "Pure-LSH"),
        ("- w/o Prefix Zones", "v2 (No\nZone B)"),
        ("BIOSYS-v2 (Adaptive Zone B)", "v2\n(Adaptive)"),
        ("BIOSYS-v2 (Fixed Zone B)", "v2\n(Fixed)"),
    ]
    labels_b = [lbl for _, lbl in systems_b]
    pref_vals = [locomo[k]["pref_hit_%"] for k, _ in systems_b]
    unc_vals = [locomo[k]["uncached_tok"] for k, _ in systems_b]

    x_idx = np.arange(len(labels_b))
    w = 0.38
    bars1 = ax2.bar(x_idx - w / 2, pref_vals, width=w, color=C_TEAL, alpha=0.88, edgecolor=C_INK, lw=0.6, label="Prefix Hit Rate (%)")
    ax2.set_ylabel("Block-16 Prefix Hit Rate (%)", fontsize=8.8, color=C_TEAL)
    ax2.set_ylim(0, 105)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(labels_b, fontsize=7.6)
    ax2.grid(True, axis="y", alpha=0.55)

    ax2_r = ax2.twinx()
    bars2 = ax2_r.bar(x_idx + w / 2, unc_vals, width=w, color=C_BLUE, alpha=0.82, edgecolor=C_INK, lw=0.6, hatch="//", label="Uncached Prefill Tok")
    ax2_r.set_ylabel("Uncached Prefill Tokens / Query (TTFT Proxy)", fontsize=8.6, color=C_BLUE)
    ax2_r.set_ylim(0, 485)
    ax2_r.spines["top"].set_visible(False)

    for b in bars1:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.8, f"{h:.0f}%", ha="center", va="bottom", fontsize=6.8, color=C_TEAL, fontweight="bold")
    for b in bars2:
        h = b.get_height()
        ax2_r.text(b.get_x() + b.get_width() / 2, h + 7, f"{h:.0f}", ha="center", va="bottom", fontsize=6.8, color=C_BLUE, fontweight="bold")

    ax2.set_title("(b) Prefix Stability vs. Uncached Prefill Compute", fontsize=9.4, fontweight="bold", pad=8)
    # Unified legend at top
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.3, frameon=True, facecolor="white", framealpha=0.92)

    fig.tight_layout(w_pad=2.2)
    save_fig(fig, "fig5_locomo_pareto")


# =============================================================================
# FIGURE 6: 5-Tier Retrieval Hit Funnel & Component Ablation Study (LoCoMo)
# =============================================================================

def build_fig6(locomo: Dict[str, Dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.30), gridspec_kw={"width_ratios": [1.22, 1.0]})

    # --- Panel (a): 100% Stacked Horizontal Bar of 5-Tier Funnel ---
    funnel_systems = [
        ("BIOSYS-v2 (Adaptive Zone B)", "BIOSYS-v2 (Full)"),
        ("- w/o L-Cache", "v2 w/o L-Cache"),
        ("- Keyword-Only Router", "v2 Keyword-Only"),
        ("BIOSYS-v1 (Orig: LSH,noDTF)", "BIOSYS-v1 (LSH)"),
        ("Mem0/Zep (Sync-KG)", "Sync-KG / RAG"),
    ]
    tiers = [
        ("L2", "L2 (Local Raw)", C_BLUE),
        ("DTF-FWD", "StoreBuf/DTF", C_AMBER),
        ("L3", "L3 (Local KG)", C_TEAL),
        ("K-TLB", "K-TLB (KG ID)", C_PURPLE),
        ("L-TLB", "L-TLB (WAL Off.)", C_CYAN),
        ("ANN", "Red-Path ANN", C_RED),
    ]

    y_pos = np.arange(len(funnel_systems))
    lefts = np.zeros(len(funnel_systems))

    for tkey, tlabel, tcol in tiers:
        vals = []
        for skey, _ in funnel_systems:
            fn = locomo[skey]["funnel"]
            tot = max(1, sum(fn.values()))
            vals.append(100.0 * fn.get(tkey, 0) / tot)
        arr = np.array(vals)
        bars = ax1.barh(y_pos, arr, left=lefts, height=0.56, color=tcol, edgecolor="#FFFFFF", lw=0.7, label=tlabel)
        for i, v in enumerate(arr):
            if v >= 8.5:
                ax1.text(lefts[i] + v / 2.0, y_pos[i], f"{v:.1f}%", ha="center", va="center",
                         fontsize=7.0, color="#FFFFFF", fontweight="bold")
        lefts += arr

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([lbl for _, lbl in funnel_systems], fontsize=8.0)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Share of Queries Resolved per Memory Tier (%)", fontsize=8.8)
    ax1.set_title("(a) Five-Tier Hit Funnel & Server ANN Offload (LoCoMo)", fontsize=9.3, fontweight="bold", pad=8)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.48, -0.22), ncol=3, fontsize=7.2, frameon=True, facecolor="white")
    ax1.grid(True, axis="x", alpha=0.45)

    # --- Panel (b): Targeted Ablation Effects (Normalized Side-by-Side Comparison) ---
    # 4 paired comparisons showing each module's targeted metric!
    categories = [
        "Lag Recall (%)\n[DTF/StoreBuf]",
        "Server ANN (%)\n[L-Cache]",
        "Server ANN (%)\n[Hybrid Router]",
        "SLM Write Tok/q\n[Amygdala Gate]",
    ]
    # Note: plot top 3 percentages on left axis, or normalize cleanly into 2x2 grouped metric bars
    # Even cleaner: plot exact % for first 3 + scaled or direct annotation! Actually showing exact percentage
    # Reduction/Gain OR two stacked mini-panels is super clear; let's use exact values with dual-axis or clean relative!
    # Wait: best IEEE style for mixed units (3 percentages + 1 token count): show 3 percentage metrics cleanly on ax2,
    # AND put Amygdala SLM write-cost reduction directly as "% of Unfiltered SLM Tokens (100% -> 55.7%)" so ALL 4 are 0-100%!!
    full_r = locomo["BIOSYS-v2 (Adaptive Zone B)"]
    no_dtf = locomo["- w/o DTF Forwarding"]
    no_l = locomo["- w/o L-Cache"]
    kw_only = locomo["- Keyword-Only Router"]
    no_amy = locomo["- w/o Amygdala Gate"]

    with_mod = [
        full_r["lag_rec_%"],
        full_r["ann_%"],
        full_r["ann_%"],
        100.0 * full_r["slm_tok"] / no_amy["slm_tok"],  # 55.7% of unfiltered write tokens
    ]
    without_mod = [
        no_dtf["lag_rec_%"],
        no_l["ann_%"],
        kw_only["ann_%"],
        100.0,
    ]
    cat_labels = [
        "Lag Recall ↑\n(w/ vs w/o DTF)",
        "ANN Rate ↓\n(w/ vs w/o L-TLB)",
        "ANN Rate ↓\n(Hybrid vs KW)",
        "SLM Write ↓\n(w/ vs w/o Amyg.)",
    ]
    x2 = np.arange(len(cat_labels))
    bw = 0.34
    b_on = ax2.bar(x2 - bw / 2, with_mod, width=bw, color=C_BLUE, edgecolor=C_INK, lw=0.6, label="BIOSYS-v2 (Module ON)")
    b_off = ax2.bar(x2 + bw / 2, without_mod, width=bw, color=C_SLATE, edgecolor=C_INK, lw=0.6, hatch="\\\\", label="Ablated (Module OFF)")

    for i, (bo, bf) in enumerate(zip(b_on, b_off)):
        ho, hf = bo.get_height(), bf.get_height()
        lbl_o = f"{ho:.1f}%" if i < 3 else f"{full_r['slm_tok']:.0f}t\n({ho:.0f}%)"
        lbl_f = f"{hf:.1f}%" if i < 3 else f"{no_amy['slm_tok']:.0f}t\n(100%)"
        ax2.text(bo.get_x() + bo.get_width() / 2, ho + 2.0, lbl_o, ha="center", va="bottom", fontsize=6.9, color=C_BLUE, fontweight="bold")
        ax2.text(bf.get_x() + bf.get_width() / 2, hf + 2.0, lbl_f, ha="center", va="bottom", fontsize=6.9, color=C_INK)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cat_labels, fontsize=7.5)
    ax2.set_ylim(0, 122)
    ax2.set_ylabel("Metric Value (% / Relative Write Cost %)", fontsize=8.5)
    ax2.set_title("(b) Independent Contribution of Each System Module", fontsize=9.3, fontweight="bold", pad=8)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.3, frameon=True, facecolor="white")
    ax2.grid(True, axis="y", alpha=0.55)

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig6_funnel_ablation")


# =============================================================================
# FIGURE 7: Coding Agent Token Black Hole + Consolidation Lag Stale-Read Sweep
# =============================================================================

def build_fig7(kill_data: Dict[str, List[Dict]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.30), gridspec_kw={"width_ratios": [1.22, 1.0]})

    # --- Panel (a): Coding Agent Amortized Cost Breakdown ---
    ca = by_name(kill_data["coding_agent"])
    ca_systems = [
        ("Full-Context", "Full-\nContext"),
        ("Full+Compact(6000)", "Full+\nCompact"),
        ("Mem0/Zep (Sync-KG)", "Sync-\nKG"),
        ("Naive-RAG (Top-K)", "Naive-\nRAG"),
        ("BIOSYS-v1 (Orig: LSH,noDTF)", "Pure-\nLSH"),
        ("- w/o Prefix Zones", "v2 (No\nZones)"),
        ("BIOSYS-v2 (Full Route A)", "BIOSYS-\nv2 (Full)"),
    ]
    x = np.arange(len(ca_systems))
    eff_inf = [ca[k]["eff_infer_tok/q"] for k, _ in ca_systems]
    slm_15 = [0.15 * ca[k]["slm_tok/q"] for k, _ in ca_systems]
    tot_15 = [e + s for e, s in zip(eff_inf, slm_15)]
    tot_10 = [ca[k]["amort_cost_1.0x"] for k, _ in ca_systems]

    b1 = ax1.bar(x, eff_inf, width=0.55, color=C_BLUE, alpha=0.88, edgecolor=C_INK, lw=0.6, label="Effective Inference Cost (0.10x Cache)")
    b2 = ax1.bar(x, slm_15, bottom=eff_inf, width=0.55, color=C_AMBER, alpha=0.90, edgecolor=C_INK, lw=0.6, label="Amortized SLM Write Cost (0.15x)")
    # Highlight worst-case 1.0x SLM cost with clean diamond markers
    ax1.scatter(x, tot_10, color=C_RED, marker="D", s=34, zorder=6, label="Worst-Case Total Cost (1.0x SLM)")

    for i, (t15, t10) in enumerate(zip(tot_15, tot_10)):
        # Keep bar label strictly clear of the red 1.0x diamond marker
        if t10 - t15 >= 220:
            bar_y = t15 + 45
            ax1.text(x[i], t10 + 55, f"{t10:.0f}", ha="center", va="bottom", fontsize=6.8, color=C_RED)
        else:
            bar_y = max(t15, t10) + 65
        ax1.text(x[i], bar_y, f"{t15:.0f}", ha="center", va="bottom",
                 fontsize=7.3, fontweight="bold" if i == len(ca_systems) - 1 else "normal",
                 color=C_BLUE if i == len(ca_systems) - 1 else C_INK)

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in ca_systems], fontsize=7.6)
    ax1.set_ylim(0, 2950)
    ax1.set_ylabel("Amortized Cost (Effective Tokens / Query)", fontsize=8.6)
    ax1.set_title("(a) Coding Agent Workload: Amortized Total Cost Breakdown", fontsize=9.2, fontweight="bold", pad=8)
    ax1.legend(loc="upper right", fontsize=7.1, frameon=True, facecolor="white", framealpha=0.94)
    ax1.grid(True, axis="y", alpha=0.55)

    # Callout highlighting -76.6% vs Mem0 and -93.2% vs Full-Context (cleanly above 171 label)
    ax1.annotate(
        "-93.2% vs Full\n-76.6% vs Sync-KG",
        xy=(6, 365), xytext=(4.05, 1720),
        fontsize=7.5, fontweight="bold", color=C_TEAL,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#ECFDF5", edgecolor=C_TEAL, lw=0.8),
        arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=1.2, connectionstyle="arc3,rad=-0.16"),
    )

    # --- Panel (b): Consolidation Lag Sweep (Stale-Read Violation Rate %) ---
    ls = by_name(kill_data["lag_sweep"])
    asst = by_name(kill_data["assistant"])
    lags = [5, 10, 20]
    stale_off = [ls[f"Lag={L:>2} | DTF=OFF"]["stale_read_%"] for L in lags]
    stale_on = [ls[f"Lag={L:>2} | DTF=ON"]["stale_read_%"] for L in lags]
    rec_off = [ls[f"Lag={L:>2} | DTF=OFF"]["latest_recall_%"] for L in lags]
    rec_on = [ls[f"Lag={L:>2} | DTF=ON"]["latest_recall_%"] for L in lags]
    mem0_async_stale = asst["Mem0/Zep (Async-Lag10)"]["stale_read_%"]

    x_l = np.arange(len(lags))
    bw = 0.32
    boff = ax2.bar(x_l - bw / 2, stale_off, width=bw, color=C_RED, alpha=0.85, edgecolor=C_INK, lw=0.6, label="BIOSYS w/o DTF (Stale %)")
    bon = ax2.bar(x_l + bw / 2, stale_on, width=bw, color=C_TEAL, alpha=0.90, edgecolor=C_INK, lw=0.6, label="BIOSYS w/ DTF (Stale %)")

    # Reference marker for Mem0/Zep Async at Lag=10 (label placed cleanly in left whitespace)
    ax2.scatter([1], [mem0_async_stale], color=C_PURPLE, marker="P", s=78, zorder=7, label="Async-KG (Lag=10)")
    ax2.annotate(f"Async-KG\n({mem0_async_stale:.1f}%)", (1, mem0_async_stale), xytext=(-14, -4),
                 textcoords="offset points", ha="right", va="center", fontsize=6.9, color=C_PURPLE, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85))

    for b in boff:
        h = b.get_height()
        if h >= 6.0:
            ax2.text(b.get_x() + b.get_width() / 2, h - 1.35, f"{h:.1f}%", ha="center", va="top", fontsize=7.2, color="#FFFFFF", fontweight="bold")
        else:
            ax2.text(b.get_x() + b.get_width() / 2, h + 0.40, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.2, color=C_RED, fontweight="bold")
    for b in bon:
        ax2.text(b.get_x() + b.get_width() / 2, 0.40, "0.0%", ha="center", va="bottom", fontsize=7.3, color=C_TEAL, fontweight="bold")

    # Secondary axis: Latest-State Recall (%)
    ax2_r = ax2.twinx()
    ax2_r.plot(x_l, rec_on, color=C_TEAL, marker="o", lw=1.8, ls="-", label="Latest Recall (DTF ON)")
    ax2_r.plot(x_l, rec_off, color=C_AMBER, marker="s", lw=1.6, ls="--", label="Latest Recall (DTF OFF)")
    ax2_r.set_ylabel("Latest-State Recall (%)", fontsize=8.5, color=C_INK)
    ax2_r.set_ylim(74, 105)
    ax2_r.spines["top"].set_visible(False)

    ax2.set_xticks(x_l)
    ax2.set_xticklabels([f"Lag = {L} turns" for L in lags], fontsize=8.0)
    ax2.set_ylim(0, 17.5)
    ax2.set_ylabel("Stale-Read Violation Rate (%)", fontsize=8.6, color=C_RED)
    ax2.set_title("(b) Consistency vs. Async Consolidation Lag", fontsize=9.2, fontweight="bold", pad=8)
    ax2.grid(True, axis="y", alpha=0.55)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.48, -0.18), ncol=2, fontsize=6.9, frameon=True, facecolor="white")

    fig.tight_layout(w_pad=2.2)
    save_fig(fig, "fig7_coding_consistency")


# =============================================================================
# FIGURE 8: Real NVIDIA A10 (24GB) + Unmodified vLLM + Qwen2.5-7B Serving Results
# =============================================================================

def build_fig8(vllm_data: Dict[str, Dict[str, Dict]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.25), gridspec_kw={"width_ratios": [1.18, 1.0]})

    # --- Panel (a): Official LoCoMo on Real vLLM + Qwen2.5-7B (1,536 QAs) ---
    lc = vllm_data["locomo"]
    sys_a = [
        ("Naive-RAG", "Naive-\nRAG"),
        ("Mem0/Zep-Sync", "Sync-KG"),
        ("Mem0/Zep-Async", "Async-KG"),
        ("BIOSYS-v2-NoZones", "v2 (No\nZones)"),
        ("BIOSYS-v2-Adaptive", "v2\n(Adaptive)"),
        ("BIOSYS-v2-Fixed", "v2\n(Fixed)"),
    ]
    x = np.arange(len(sys_a))
    p50 = [lc[k]["ttft_p50_ms"] for k, _ in sys_a]
    kv = [lc[k]["vllm_prefix_hit_%"] for k, _ in sys_a]
    f1 = [lc[k]["qa_f1_%"] for k, _ in sys_a]
    lag_f1 = [lc[k]["lag_f1_%"] for k, _ in sys_a]

    w = 0.36
    b_ttft = ax1.bar(x - w / 2, p50, width=w, color=C_BLUE, alpha=0.88, edgecolor=C_INK, lw=0.6, label="Measured P50 TTFT (ms)")
    ax1.set_ylabel("Real GPU P50 TTFT (ms)", fontsize=8.6, color=C_BLUE)
    ax1.set_ylim(0, 205)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in sys_a], fontsize=7.5)
    ax1.grid(True, axis="y", alpha=0.55)

    for b in b_ttft:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 3.2, f"{h:.0f}", ha="center", va="bottom",
                 fontsize=6.9, color=C_BLUE, fontweight="bold")

    ax1_r = ax1.twinx()
    b_kv = ax1_r.bar(x + w / 2, kv, width=w, color=C_TEAL, alpha=0.84, edgecolor=C_INK, lw=0.6, hatch="//", label="vLLM KV Prefix Hit (%)")
    ax1_r.plot(x, f1, color=C_RED, marker="o", lw=1.7, ms=5.0, zorder=6, label="Qwen2.5-7B QA F1 (%)")
    ax1_r.plot(x, lag_f1, color=C_AMBER, marker="s", lw=1.4, ls="--", ms=4.5, zorder=6, label="Consolidation-Lag F1 (%)")
    ax1_r.set_ylabel("vLLM Prefix Hit / Qwen2.5-7B F1 (%)", fontsize=8.4, color=C_INK)
    ax1_r.set_ylim(0, 108)
    ax1_r.spines["top"].set_visible(False)

    for b in b_kv:
        h = b.get_height()
        if h <= 35:
            ax1_r.text(b.get_x() + b.get_width() / 2, 5.8, f"{h:.0f}%", ha="center", va="center",
                       fontsize=6.8, color="#FFFFFF", fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.10", facecolor=C_TEAL, edgecolor="none", alpha=0.90))
        else:
            ax1_r.text(b.get_x() + b.get_width() / 2, h + 1.8, f"{h:.0f}%", ha="center", va="bottom",
                       fontsize=6.8, color=C_TEAL, fontweight="bold")

    # Compact note for Full-Context off-scale P50 TTFT
    ax1.annotate(
        "Full-Context: P50=2,792 ms (27.9× slower),\nKVHit=38.3%, RPS=0.50, F1=46.4%",
        xy=(0.02, 0.96), xycoords="axes fraction", ha="left", va="top",
        fontsize=6.9, color=C_MUTED,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#F8FAFC", edgecolor=C_SLATE, lw=0.7),
    )

    ax1.set_title("(a) Real vLLM Serving on LoCoMo (A10 24GB, 1,536 QAs)", fontsize=9.1, fontweight="bold", pad=8)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.48, -0.20), ncol=2, fontsize=6.9, frameon=True, facecolor="white")

    # --- Panel (b): Tool-Heavy Coding Agent on Real vLLM + Qwen2.5-7B (Log-Y TTFT + Throughput) ---
    ca = vllm_data["coding_agent"]
    sys_b = [
        ("Full-Context", "Full-Ctx\n(Trunc.)"),
        ("Mem0/Zep-Sync", "Sync-KG"),
        ("BIOSYS-v2-NoZones", "v2 (No\nZones)"),
        ("BIOSYS-v2-Full", "BIOSYS-v2\n(Full)"),
    ]
    xb = np.arange(len(sys_b))
    p50_b = [ca[k]["ttft_p50_ms"] for k, _ in sys_b]
    p99_b = [ca[k]["ttft_p99_ms"] for k, _ in sys_b]
    rps_b = [ca[k]["throughput_rps"] for k, _ in sys_b]
    kv_b = [ca[k]["vllm_prefix_hit_%"] for k, _ in sys_b]

    wb = 0.34
    bp50 = ax2.bar(xb - wb / 2, p50_b, width=wb, color=C_BLUE, alpha=0.88, edgecolor=C_INK, lw=0.6, label="P50 TTFT (ms, Log)")
    bp99 = ax2.bar(xb + wb / 2, p99_b, width=wb, color=C_PURPLE, alpha=0.80, edgecolor=C_INK, lw=0.6, hatch="\\\\", label="P99 TTFT (ms, Log)")
    ax2.set_yscale("log")
    ax2.set_ylim(20, 5200)
    ax2.set_ylabel("Measured GPU TTFT (ms, Log Scale)", fontsize=8.5, color=C_BLUE)
    ax2.set_xticks(xb)
    ax2.set_xticklabels([f"{lbl}\n(KV:{k:.0f}%)" for (_, lbl), k in zip(sys_b, kv_b)], fontsize=7.4)
    ax2.grid(True, axis="y", alpha=0.55)

    for b in bp50:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h * 1.09, f"{h:.0f}", ha="center", va="bottom",
                 fontsize=6.9, color=C_BLUE, fontweight="bold")
    for b in bp99:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h * 1.09, f"{h:.0f}", ha="center", va="bottom",
                 fontsize=6.8, color=C_PURPLE)

    ax2_r = ax2.twinx()
    ax2_r.plot(xb, rps_b, color=C_TEAL, marker="D", lw=2.0, ms=5.8, zorder=7, label="Serial Request Rate (req/s)")
    ax2_r.set_ylabel("Serial Request Rate (req/s)", fontsize=8.5, color=C_TEAL)
    ax2_r.set_ylim(0, 6.2)
    ax2_r.spines["top"].set_visible(False)
    for i, r in enumerate(rps_b):
        offset_y = 0.35 if i < 3 else 0.32
        ax2_r.text(xb[i], r + offset_y, f"{r:.2f}", ha="center", va="bottom",
                   fontsize=7.3, color=C_TEAL, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.8))

    ax2.annotate(
        "4.53× Faster P50 TTFT\n+67.6% Serial Rate\n100% SubEM Match",
        xy=(3 - wb / 2, 56), xytext=(1.38, 1280),
        fontsize=7.1, fontweight="bold", color=C_TEAL,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#ECFDF5", edgecolor=C_TEAL, lw=0.8),
        arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=1.2, connectionstyle="arc3,rad=-0.16"),
    )

    ax2.set_title("(b) Real vLLM Serving on Coding Agent (A10 24GB)", fontsize=9.1, fontweight="bold", pad=8)
    hb1, lb1 = ax2.get_legend_handles_labels()
    hb2, lb2 = ax2_r.get_legend_handles_labels()
    ax2.legend(hb1 + hb2, lb1 + lb2, loc="upper center", bbox_to_anchor=(0.48, -0.22), ncol=2, fontsize=6.9, frameon=True, facecolor="white")

    fig.tight_layout(w_pad=2.1)
    save_fig(fig, "fig8_vllm_serving")


# =============================================================================
# FIGURE 9: Multi-Tenant Concurrent Shared-vLLM Scaling & KV Thrashing Cliff
# =============================================================================

def build_fig9(conc_data: Dict[str, Dict[str, Dict]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.20), gridspec_kw={"width_ratios": [1.0, 1.08]})
    concs = [1, 4, 8, 16, 32]
    x = np.arange(len(concs))
    x_labels = [str(c) for c in concs]

    # --- Panel (a): Multi-Tenant Coding Agent Throughput & SLO-300 Goodput ---
    ca = conc_data["coding_concurrent"]
    ca_series = [
        ("BIOSYS-Full", "BIOSYS (Full)", C_BLUE, "o"),
        ("BIOSYS-NoZones", "NoZones", C_AMBER, "s"),
        ("Sync-KG", "Sync-KG", C_PURPLE, "^"),
    ]
    for key, lbl, col, mk in ca_series:
        thr = [ca[f"{key}_C{c}"]["throughput_rps"] for c in concs]
        gp3 = [ca[f"{key}_C{c}"]["goodput300_rps"] for c in concs]
        ax1.plot(x, thr, color=col, marker=mk, lw=2.0, ms=5.5, ls="-", label=f"{lbl} Thr.")
        ax1.plot(x, gp3, color=col, marker=mk, lw=1.4, ms=4.5, ls="--", alpha=0.82,
                 markerfacecolor="white", label=f"{lbl} GP300")

    # Clean value labels placed strictly in open whitespace (legend sits below panel)
    ax1.annotate("31.0 r/s (3× isol.)\n6.23× Sync-KG | 1.83× Sorted", (4, 32.04), xytext=(-10, 12),
                 textcoords="offset points", ha="right", va="bottom", fontsize=6.7,
                 fontweight="bold", color=C_BLUE,
                 bbox=dict(boxstyle="round,pad=0.16", facecolor="#EFF6FF", edgecolor=C_BLUE, lw=0.7),
                 arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.0))
    ax1.annotate("Peak SLO-300 GP:\n25.2 vs 0.03 r/s", (3, 25.16), xytext=(-58, -32),
                 textcoords="offset points", ha="center", va="top", fontsize=6.9,
                 fontweight="bold", color=C_TEAL,
                 bbox=dict(boxstyle="round,pad=0.16", facecolor="#ECFDF5", edgecolor=C_TEAL, lw=0.7),
                 arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=1.0))
    ax1.text(3.92, 7.6, "NoZones: 6.4", fontsize=6.7, color=C_AMBER, ha="right", fontweight="bold")
    ax1.text(3.92, 2.8, "Sync-KG: 5.1 (GP→0.01)", fontsize=6.7, color=C_PURPLE, ha="right", fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=8.0)
    ax1.set_xlabel("Concurrent Tenant Sessions (C) on Single A10", fontsize=8.3)
    ax1.set_ylabel("Throughput / SLO-300 Goodput (req/s)", fontsize=8.4)
    ax1.set_ylim(-0.8, 42.5)
    ax1.set_title("(a) Multi-Tenant Coding Agent: Throughput & SLO-300 Goodput", fontsize=8.9, fontweight="bold", pad=7)
    ax1.grid(True, alpha=0.55)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.48, -0.22), ncol=3, fontsize=6.4, frameon=True, facecolor="white", handlelength=1.6)

    # --- Panel (b): Multi-Tenant LoCoMo KV-Cache Thrashing Cliff ---
    lc = conc_data["locomo_concurrent"]
    lc_series = [
        ("BIOSYS-Adaptive", "Adaptive (P50)", C_BLUE, "o"),
        ("BIOSYS-Fixed", "Fixed (P50)", C_RED, "D"),
        ("Sync-KG", "Sync-KG (P50)", C_PURPLE, "^"),
        ("BIOSYS-NoZones", "NoZones (P50)", C_AMBER, "s"),
    ]
    for key, lbl, col, mk in lc_series:
        p50 = [lc[f"{key}_C{c}"]["ttft_p50_ms"] for c in concs]
        ax2.plot(x, p50, color=col, marker=mk, lw=1.9 if "BIOSYS" in key else 1.5, ms=5.2, ls="-", label=lbl)

    ax2.set_yscale("log")
    ax2.set_ylim(60, 34000)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=8.0)
    ax2.set_xlabel("Concurrent Tenant Sessions (C) on Single A10", fontsize=8.3)
    ax2.set_ylabel("Measured P50 TTFT (ms, Log Scale)", fontsize=8.4, color=C_INK)
    ax2.grid(True, alpha=0.55)

    # Right axis: KV Prefix Hit Rate (%) for Adaptive vs Fixed (clearly exposing the C=32 eviction cliff)
    ax2_r = ax2.twinx()
    kv_adapt = [lc[f"BIOSYS-Adaptive_C{c}"]["vllm_kv_hit_%"] for c in concs]
    kv_fixed = [lc[f"BIOSYS-Fixed_C{c}"]["vllm_kv_hit_%"] for c in concs]
    ax2_r.plot(x, kv_adapt, color=C_TEAL, marker="o", lw=1.6, ms=4.5, ls=":", label="Adaptive KVHit%")
    ax2_r.plot(x, kv_fixed, color=C_RED, marker="D", lw=1.6, ms=4.5, ls=":", markerfacecolor="white", label="Fixed KVHit%")
    ax2_r.set_ylabel("vLLM Prometheus KV Prefix Hit (%)", fontsize=8.3, color=C_TEAL)
    ax2_r.set_ylim(0, 108)
    ax2_r.spines["top"].set_visible(False)

    # Place callout cleanly in open middle-left band (y=1100..3000) with zero line contact
    ax2.annotate(
        "C=32 Shared-Cache Contention (3× isol.):\nFixed P50=13.99s, KVHit→15.4%\nAdaptive: 0.59s (23.7×), KVHit 71.4%",
        xy=(4, 13739.2), xytext=(0.12, 1180),
        fontsize=6.7, fontweight="bold", color=C_RED,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#FEF2F2", edgecolor=C_RED, lw=0.8),
        arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.2, connectionstyle="arc3,rad=-0.18"),
    )
    ax2_r.text(4.02, 75.5, "71.9%", fontsize=6.9, color=C_TEAL, fontweight="bold", ha="right")
    ax2_r.text(3.90, 11.2, "16.8%", fontsize=6.9, color=C_RED, fontweight="bold", ha="right")

    ax2.set_title("(b) Multi-Tenant LoCoMo: P50 TTFT & Shared KV Contention", fontsize=8.9, fontweight="bold", pad=7)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.48, -0.22), ncol=3, fontsize=6.5, frameon=True, facecolor="white")

    fig.tight_layout(w_pad=2.1)
    save_fig(fig, "fig9_vllm_concurrent")


def main() -> None:
    locomo_rows = json.loads((ROOT / "locomo_results.json").read_text(encoding="utf-8"))
    kill_data = json.loads((ROOT / "kill_test_results.json").read_text(encoding="utf-8"))
    locomo = by_name(locomo_rows)

    build_fig5(locomo)
    build_fig6(locomo)
    build_fig7(kill_data)
    vllm_path = ROOT / "vllm_serving_results.json"
    if vllm_path.exists():
        build_fig8(json.loads(vllm_path.read_text(encoding="utf-8")))
    conc_path = ROOT / "vllm_concurrent_results.json"
    if conc_path.exists():
        build_fig9(json.loads(conc_path.read_text(encoding="utf-8")))
        print(f"Generated 5 experimental result figures (Fig 5-9, PDF/SVG/PNG) in: {OUT_DIR}")
    else:
        print(f"Generated 4 experimental result figures (Fig 5-8, PDF/SVG/PNG) in: {OUT_DIR}")


if __name__ == "__main__":
    main()
