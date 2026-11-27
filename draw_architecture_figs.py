#!/usr/bin/env python3
"""
Generate 4 vector architecture figures for BIOSYS v2:
  - figures/fig1_architecture.{pdf,svg,png}
  - figures/fig2_coprocessor.{pdf,svg,png}
  - figures/fig3_adapter.{pdf,svg,png}
  - figures/fig4_cascade.{pdf,svg,png}
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_INK = "#1E293B"
C_MUTED = "#475569"

C_BLUE_BG = "#EFF6FF"
C_BLUE_BD = "#2563EB"

C_TEAL_BG = "#ECFDF5"
C_TEAL_BD = "#059669"

C_AMBER_BG = "#FFFBEB"
C_AMBER_BD = "#D97706"

C_PURPLE_BG = "#F5F3FF"
C_PURPLE_BD = "#7C3AED"

C_RED_BG = "#FEF2F2"
C_RED_BD = "#DC2626"

C_SLATE_BG = "#F8FAFC"
C_SLATE_BD = "#64748B"


def save_fig(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.08}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(OUT_DIR / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    fc: str = "#FFFFFF",
    ec: str = C_INK,
    lw: float = 1.3,
    ls: str = "-",
    title_size: float = 9.4,
    sub_size: float = 7.7,
    rounding: float = 0.13,
    zorder: int = 3,
    title_color: str = C_INK,
    sub_color: str = C_MUTED,
    bold: bool = True,
    title_offset: float = 0.17,
    sub_offset: float = 0.18,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder,
    )
    ax.add_patch(patch)
    cx, cy = x + w / 2.0, y + h / 2.0
    if subtitle:
        ax.text(
            cx, cy + h * title_offset, title,
            ha="center", va="center", fontsize=title_size,
            fontweight="bold" if bold else "normal", color=title_color,
            zorder=zorder + 1, linespacing=1.15,
        )
        ax.text(
            cx, cy - h * sub_offset, subtitle,
            ha="center", va="center", fontsize=sub_size,
            color=sub_color, zorder=zorder + 1, linespacing=1.22,
        )
    else:
        ax.text(
            cx, cy, title,
            ha="center", va="center", fontsize=title_size,
            fontweight="bold" if bold else "normal", color=title_color,
            zorder=zorder + 1, linespacing=1.22,
        )
    return patch


def draw_arrow(
    ax: plt.Axes,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    color: str = C_INK,
    lw: float = 1.35,
    style: str = "-|>",
    ls: str = "-",
    rad: float = 0.0,
    zorder: int = 5,
    ms: float = 11.5,
) -> None:
    arr = FancyArrowPatch(
        p0, p1,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(arr)


def draw_ortho_arrow(
    ax: plt.Axes,
    pts: List[Tuple[float, float]],
    color: str = C_INK,
    lw: float = 1.35,
    ls: str = "-",
    style: str = "-|>",
    zorder: int = 5,
    ms: float = 11.5,
) -> None:
    if len(pts) < 2:
        return
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round", solid_joinstyle="round", zorder=zorder)
    draw_arrow(ax, pts[-2], pts[-1], color=color, lw=lw, style=style, ls=ls, zorder=zorder, ms=ms)


def label_pill(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str = C_INK,
    size: float = 7.4,
    bold: bool = False,
    bg: str = "#FFFFFF",
    ec: str = "none",
    ha: str = "center",
    va: str = "center",
    zorder: int = 7,
) -> None:
    ax.text(
        x, y, text,
        ha=ha, va=va, fontsize=size,
        fontweight="bold" if bold else "normal",
        color=color, zorder=zorder, linespacing=1.18,
        bbox=dict(boxstyle="round,pad=0.15", facecolor=bg, edgecolor=ec, lw=0.7),
    )


# =============================================================================
# FIGURE 1: Three-Tier Bio-Systems Macro Architecture
# =============================================================================

def build_fig1() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    cols = [
        (1.05, 2.55, "Cognitive Science (CLS)", C_TEAL_BG, C_TEAL_BD),
        (3.85, 3.35, "BIOSYS v2 Architecture", C_BLUE_BG, C_BLUE_BD),
        (7.45, 2.50, "Computer Systems", C_SLATE_BG, C_SLATE_BD),
    ]
    for cx, cw, title, bg, bd in cols:
        draw_box(ax, cx, 5.55, cw, 0.48, title, fc=bg, ec=bd, lw=1.4, title_size=9.8, rounding=0.10)
        rect = Rectangle((cx, 0.25), cw, 5.20, facecolor="none", edgecolor="#CBD5E1", lw=0.9, zorder=0)
        ax.add_patch(rect)

    tiers = [
        (3.82, 1.48, "Working Memory\n(Client Boundary)", C_BLUE_BG, C_BLUE_BD),
        (2.05, 1.48, "Short-Term Memory\n(Server Adapter)", C_AMBER_BG, C_AMBER_BD),
        (0.32, 1.45, "Long-Term Memory\n(Server Persister)", C_SLATE_BG, C_SLATE_BD),
    ]
    for ty, th, tlabel, tbg, tbd in tiers:
        band = FancyBboxPatch(
            (0.12, ty), 9.96, th,
            boxstyle="round,pad=0.01,rounding_size=0.10",
            facecolor=tbg, edgecolor=tbd, lw=1.1, ls="--", alpha=0.38, zorder=1,
        )
        ax.add_patch(band)
        ax.text(
            0.56, ty + th / 2.0, tlabel,
            ha="center", va="center", rotation=90,
            fontsize=8.3, fontweight="bold", color=tbd, zorder=2, linespacing=1.18,
        )

    # Row 1: Working Memory
    draw_box(ax, 1.24, 4.06, 2.16, 0.98, "Prefrontal Cortex", "Active Working Set &\nExecutive Routing", fc="#FFFFFF", ec=C_TEAL_BD, lw=1.3)
    draw_box(ax, 4.06, 4.02, 2.94, 1.06, "Client Coprocessor", "L1–L3 Caches • Split Semantic TLB\nStore Buffer / DTF • 3-Zone Assembler", fc="#FFFFFF", ec=C_BLUE_BD, lw=1.5)
    draw_box(ax, 7.66, 4.06, 2.08, 0.98, "CPU Cache & MMU", "L1–L3 SRAM • Split TLB\nStore Buffer (~0 ms)", fc="#FFFFFF", ec=C_SLATE_BD, lw=1.3)

    # Row 2: Short-Term Memory (Single-line Hippocampus & Amygdala -> zero crowding!)
    draw_box(ax, 1.24, 2.30, 2.16, 0.98, "Hippocampus & Amygdala", "Fast Episodic Encoding &\nSalience Priority Gating", fc="#FFFFFF", ec=C_TEAL_BD, lw=1.3, title_size=8.8)
    draw_box(ax, 4.06, 2.26, 2.94, 1.06, "Server Adapter", "Fast Path: Crash-Consistent WAL\nSmart Path: Amygdala + Sleep SLM", fc="#FFFFFF", ec=C_AMBER_BD, lw=1.5)
    draw_box(ax, 7.66, 2.30, 2.08, 0.98, "Main Memory", "DRAM Log / Replay Buffers\nMemory Controller", fc="#FFFFFF", ec=C_SLATE_BD, lw=1.3)

    # Row 3: Long-Term Memory
    draw_box(ax, 1.24, 0.56, 2.16, 0.96, "Neocortex", "Consolidated Semantic\nSchema & World Knowledge", fc="#FFFFFF", ec=C_TEAL_BD, lw=1.3)
    draw_box(ax, 4.06, 0.52, 2.94, 1.04, "Server Persister", "Append-Only Log Archive ($O(1)$)\nBi-Temporal Knowledge Graph + ANN", fc="#FFFFFF", ec=C_INK, lw=1.5)
    draw_box(ax, 7.66, 0.56, 2.08, 0.96, "Persistent Storage", "SSD / NVM Archive &\nIndexed Graph Store", fc="#FFFFFF", ec=C_SLATE_BD, lw=1.3)

    # Vertical arrows
    draw_arrow(ax, (2.32, 4.04), (2.32, 3.30), color=C_TEAL_BD, style="<->", lw=1.3)
    draw_arrow(ax, (2.32, 2.28), (2.32, 1.54), color=C_TEAL_BD, style="<->", lw=1.3)
    label_pill(ax, 2.32, 3.67, "Encode / Recall", color=C_TEAL_BD, size=7.1, bg="#F8FAFC")
    label_pill(ax, 2.32, 1.91, "Sleep Replay", color=C_TEAL_BD, size=7.1, bg="#F8FAFC")

    draw_arrow(ax, (8.70, 4.04), (8.70, 3.30), color=C_SLATE_BD, style="<->", lw=1.3)
    draw_arrow(ax, (8.70, 2.28), (8.70, 1.54), color=C_SLATE_BD, style="<->", lw=1.3)
    label_pill(ax, 8.70, 3.67, "Load / Write-Back", color=C_SLATE_BD, size=7.1, bg="#F8FAFC")
    label_pill(ax, 8.70, 1.91, "Flush / Page-In", color=C_SLATE_BD, size=7.1, bg="#F8FAFC")

    draw_arrow(ax, (4.82, 4.00), (4.82, 3.34), color=C_RED_BD, lw=1.7)
    draw_arrow(ax, (6.24, 3.34), (6.24, 4.00), color=C_RED_BD, lw=1.7)
    label_pill(ax, 5.53, 3.67, "DMP: Store / LoadByID / ANN / Watermark", color=C_RED_BD, size=7.3, bold=True, bg="#FFF5F5", ec=C_RED_BD)

    draw_arrow(ax, (4.82, 2.24), (4.82, 1.58), color=C_AMBER_BD, lw=1.6)
    draw_arrow(ax, (6.24, 1.58), (6.24, 2.24), color=C_BLUE_BD, lw=1.6)
    label_pill(ax, 4.82, 1.91, "WAL + Sleep Replay", color=C_AMBER_BD, size=7.0, bold=True, bg="#FFFBEB")
    label_pill(ax, 6.24, 1.91, "$O(1)$ Read / Recovery", color=C_BLUE_BD, size=7.0, bold=True, bg="#EFF6FF")

    for y_mid in (4.55, 2.79, 1.04):
        ax.plot([3.42, 4.04], [y_mid, y_mid], color="#94A3B8", lw=1.0, ls=":", zorder=2)
        ax.plot([7.02, 7.64], [y_mid, y_mid], color="#94A3B8", lw=1.0, ls=":", zorder=2)

    save_fig(fig, "fig1_architecture")


# =============================================================================
# FIGURE 2: Client Memory Coprocessor Microarchitecture
# =============================================================================

def build_fig2() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.8))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    # Top: Unmodified LLM / Agent Serving Engine
    draw_box(
        ax, 0.32, 5.92, 10.56, 0.66,
        "LLM / Agent  (Serving-Agnostic: Unmodified vLLM / SGLang RadixAttention or Cloud Prompt Caching)",
        fc=C_SLATE_BG, ec=C_INK, lw=1.5, title_size=9.4, rounding=0.12,
    )

    # Outer Client Coprocessor Frame
    coproc = FancyBboxPatch(
        (0.32, 0.24), 9.48, 5.20,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor="#FAFCFF", edgecolor=C_BLUE_BD, lw=1.5, ls="--", zorder=1,
    )
    ax.add_patch(coproc)

    # Left vertical strip
    draw_box(ax, 0.44, 0.36, 0.44, 4.96, "", fc=C_BLUE_BG, ec=C_BLUE_BD, lw=1.1, rounding=0.08, zorder=2)
    ax.text(0.66, 2.84, "Client Memory Coprocessor  (~0 ms Local Boundary)", rotation=90,
            ha="center", va="center", fontsize=8.8, fontweight="bold", color=C_BLUE_BD, zorder=3)

    # Context Assembler (Top-Left) with Three-Zone Stable Layout
    asm = FancyBboxPatch(
        (1.12, 3.96), 5.42, 1.34,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor="#FFFFFF", edgecolor=C_TEAL_BD, lw=1.5, zorder=2,
    )
    ax.add_patch(asm)
    ax.text(3.83, 5.12, "Prefix-Cache-Aware Context Assembler (Three-Zone Stable Layout)",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color=C_TEAL_BD, zorder=3)

    draw_box(ax, 1.24, 4.08, 1.48, 0.82, "Zone A", "Immutable System\n& Tools (100% Hit)",
             fc=C_SLATE_BG, ec=C_SLATE_BD, lw=1.1, title_size=8.2, sub_size=6.8, rounding=0.07, zorder=3)
    draw_box(ax, 2.82, 4.08, 1.92, 0.82, "Zone B (Epoch-Pinned)", "Adaptive L3 KG Slices\n(NodeID-Ordered, $\\hat{p} \\geq \\tau$)",
             fc=C_TEAL_BG, ec=C_TEAL_BD, lw=1.2, title_size=7.9, sub_size=6.8, rounding=0.07, zorder=3)
    draw_box(ax, 4.84, 4.08, 1.58, 0.82, "Zone C (Dynamic)", "Time + Delta + L1\n(Zero Prefix Break)",
             fc=C_AMBER_BG, ec=C_AMBER_BD, lw=1.2, title_size=7.9, sub_size=6.8, rounding=0.07, zorder=3)

    # Top-Right: Hybrid Topic Encoder
    draw_box(
        ax, 6.94, 4.00, 2.68, 1.28,
        "Hybrid Topic Encoder",
        "Fine: High-IDF / Entity + Centroids\nCoarse: 2-Probe SimHash\n(Tag-Verified Routing)",
        fc=C_PURPLE_BG, ec=C_PURPLE_BD, lw=1.4, title_size=8.8, sub_size=7.1, rounding=0.10, zorder=3,
    )

    # Left-Lower Cache Stack (cleanly spaced below routing corridor y=3.14..3.94)
    draw_box(
        ax, 1.16, 2.40, 3.24, 0.68,
        "L1-Cache  (Immediate Working Tail)",
        "Recent $W_1$ Turns + Offloaded WAL Tool Handles",
        fc="#FFFFFF", ec=C_BLUE_BD, lw=1.25, title_size=8.2, sub_size=6.9, rounding=0.09,
    )
    draw_box(
        ax, 1.16, 1.42, 3.24, 0.72,
        "L2-Cache  (Session Raw Cache)",
        "Recency-Decay LRU  (~0 ms Local Access)",
        fc="#FFFFFF", ec=C_BLUE_BD, lw=1.25, title_size=8.2, sub_size=6.9, rounding=0.09,
    )
    draw_box(
        ax, 1.16, 0.42, 3.24, 0.74,
        "L3-Cache  (Consolidated Local KG)",
        "Versioned KG Slices + EWMA Heat Tracker",
        fc=C_TEAL_BG, ec=C_TEAL_BD, lw=1.25, title_size=8.2, sub_size=6.9, rounding=0.09,
    )

    # Middle-Center: Store Buffer & DTF Filter
    draw_box(
        ax, 4.82, 1.66, 1.90, 1.42,
        "Store Buffer\n& DTF Filter",
        "Un-consolidated Ring\n(lsn > wm)\nFine-Key Dirty Bits\n(Zero False Sharing)",
        fc=C_AMBER_BG, ec=C_AMBER_BD, lw=1.4, title_size=8.2, sub_size=6.8, rounding=0.10,
        title_offset=0.26, sub_offset=0.18,
    )

    # Right-Lower: Split Semantic TLB
    tlb_frame = FancyBboxPatch(
        (7.06, 0.42), 2.56, 2.68,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=C_PURPLE_BG, edgecolor=C_PURPLE_BD, lw=1.45, zorder=2,
    )
    ax.add_patch(tlb_frame)
    ax.text(8.34, 2.93, "Split Semantic TLB", ha="center", va="center",
            fontsize=8.6, fontweight="bold", color=C_PURPLE_BD, zorder=3)
    draw_box(ax, 7.22, 1.98, 2.24, 0.75, "K-Cache (KG TLB)", "TopicKey → Remote NodeIDs\n(Small Payload Prefetch)",
             fc="#FFFFFF", ec=C_PURPLE_BD, lw=1.15, title_size=7.9, sub_size=6.7, rounding=0.08, zorder=3)
    draw_box(ax, 7.22, 1.06, 2.24, 0.75, "L-Cache (Log TLB)", "TopicKey → WAL Offsets\n($O(1)$ Direct Log Read)",
             fc="#FFFFFF", ec=C_PURPLE_BD, lw=1.15, title_size=7.9, sub_size=6.7, rounding=0.08, zorder=3)
    draw_box(ax, 7.22, 0.52, 2.24, 0.38, "Semantic Tag Verifier",
             fc="#EDE9FE", ec=C_PURPLE_BD, lw=1.0, title_size=7.4, rounding=0.06, zorder=3)

    # === CLEAN NON-CROSSING ARROWS ===
    # Top LLM <-> Coprocessor (labels placed BESIDE arrows -> zero overlap!)
    draw_arrow(ax, (3.83, 5.32), (3.83, 5.90), color=C_TEAL_BD, lw=1.7)
    label_pill(ax, 2.28, 5.64, "Assembled Prefix-Stable Context", color=C_TEAL_BD, size=7.5, bold=True)

    draw_arrow(ax, (8.28, 5.90), (8.28, 5.30), color=C_BLUE_BD, lw=1.7)
    label_pill(ax, 9.65, 5.64, "User Query &\nAgent Response", color=C_BLUE_BD, size=7.4, bold=True)

    # Clean separated middle routing corridor (y = 3.24, 3.48, 3.72):
    # 1. Encoder -> Split-TLB (straight vertical down on right)
    draw_arrow(ax, (8.34, 3.98), (8.34, 3.12), color=C_PURPLE_BD, lw=1.4)
    label_pill(ax, 9.10, 3.56, "Fine/Coarse\nTopicKeys", color=C_PURPLE_BD, size=6.9)

    # 2. Encoder -> Store Buffer & DTF (clean L-bend at y=3.52)
    draw_ortho_arrow(ax, [(7.36, 3.98), (7.36, 3.52), (6.16, 3.52), (6.16, 3.10)], color=C_AMBER_BD, lw=1.35)
    label_pill(ax, 6.86, 3.52, "Fine Keys", color=C_AMBER_BD, size=6.8)

    # 3. Encoder -> L1/L2 (clean high rail at y=3.74 -> down at x=4.60 -> L1)
    draw_ortho_arrow(ax, [(6.92, 4.18), (6.72, 4.18), (6.72, 3.74), (4.58, 3.74), (4.58, 2.74), (4.42, 2.74)], color=C_BLUE_BD, lw=1.2)
    label_pill(ax, 5.06, 3.74, "Raw Turn", color=C_BLUE_BD, size=6.7)

    # 4. L3 -> Zone B (clean left channel x=1.01 -> y=3.50 -> Zone B)
    draw_ortho_arrow(ax, [(1.14, 0.79), (1.01, 0.79), (1.01, 3.48), (3.78, 3.48), (3.78, 4.06)], color=C_TEAL_BD, lw=1.45)
    label_pill(ax, 2.40, 3.48, "Epoch Pin (L3 → Zone B)", color=C_TEAL_BD, size=7.0, bold=True)

    # 5. L1 -> Zone C (clean rail at y=3.26 -> x=5.12 -> Zone C)
    draw_ortho_arrow(ax, [(3.96, 3.10), (3.96, 3.26), (5.16, 3.26), (5.16, 4.06)], color=C_BLUE_BD, lw=1.3)
    label_pill(ax, 4.54, 3.26, "L1 Tail", color=C_BLUE_BD, size=6.7)

    # 6. Store Buffer / DTF -> Zone C (straight vertical at x=5.62, label cleanly at x=5.62, y=3.86)
    draw_arrow(ax, (5.62, 3.10), (5.62, 4.06), color=C_AMBER_BD, lw=1.5)
    label_pill(ax, 6.18, 3.86, "RYOW Delta", color=C_AMBER_BD, size=6.8, bold=True)

    # Lower horizontal interconnects (completely clear!)
    draw_arrow(ax, (4.42, 1.94), (4.80, 1.94), color=C_AMBER_BD, lw=1.3)
    label_pill(ax, 4.61, 1.72, "WB", color=C_AMBER_BD, size=6.5)

    draw_arrow(ax, (6.74, 2.38), (7.04, 2.38), color=C_AMBER_BD, lw=1.35)
    draw_arrow(ax, (7.04, 0.78), (4.42, 0.78), color=C_TEAL_BD, lw=1.45)
    label_pill(ax, 5.75, 0.78, "Prefetch / Backfill L3", color=C_TEAL_BD, size=7.1, bold=True)

    draw_ortho_arrow(ax, [(7.04, 1.32), (5.77, 1.32), (5.77, 1.64)], color=C_AMBER_BD, lw=1.25, ls="--")
    label_pill(ax, 6.38, 1.32, "Watermark Ack", color=C_AMBER_BD, size=6.8)

    # Right DMP Boundary Arrow
    draw_arrow(ax, (9.64, 1.78), (10.94, 1.78), color=C_RED_BD, style="<->", lw=2.0, ms=13)
    label_pill(ax, 10.42, 2.28, "DMP\n(Store / Load\nWatermark)", color=C_RED_BD, size=7.3, bold=True, bg="#FFF5F5", ec=C_RED_BD)

    save_fig(fig, "fig2_coprocessor")


# =============================================================================
# FIGURE 3: Server Dual-Path Adapter & Hybrid Consistency Architecture
# =============================================================================

def build_fig3() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    srv = FancyBboxPatch(
        (1.64, 0.22), 9.54, 5.56,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor="#FAFBFC", edgecolor=C_SLATE_BD, lw=1.3, ls="--", zorder=0,
    )
    ax.add_patch(srv)
    ax.text(6.41, 5.62, "Server-Side Disaggregated Memory Adapter & Persistence Layer",
            ha="center", va="center", fontsize=9.3, fontweight="bold", color=C_INK, zorder=1)

    # Left External Client Coprocessor bar
    draw_box(ax, 0.16, 0.48, 0.88, 4.92, "", fc=C_BLUE_BG, ec=C_BLUE_BD, lw=1.4, rounding=0.10, zorder=2)
    ax.text(0.60, 2.94, "Client Coprocessor\n(L1–L3, Split-TLB, DTF)", rotation=90,
            ha="center", va="center", fontsize=8.4, fontweight="bold", color=C_BLUE_BD, zorder=3, linespacing=1.2)

    # Memory Controller bar
    draw_box(ax, 1.82, 0.48, 1.04, 4.92, "", fc=C_SLATE_BG, ec=C_INK, lw=1.4, rounding=0.10, zorder=2)
    ax.text(2.34, 2.94, "Memory Controller\n(Auth, Router,\nMonotonic LSN &\nWatermark Ack)", rotation=90,
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=C_INK, zorder=3, linespacing=1.22)

    # Persistence Manager bar
    draw_box(ax, 8.60, 0.48, 0.92, 4.92, "", fc=C_SLATE_BG, ec=C_INK, lw=1.4, rounding=0.10, zorder=2)
    ax.text(9.06, 2.94, "Persistence Manager\n(Consistency &\nConcurrency Handler)", rotation=90,
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=C_INK, zorder=3, linespacing=1.22)

    # Far Right Persister Targets
    draw_box(ax, 9.74, 4.02, 1.30, 1.18, "WAL Archive", "Append-Only\n$O(1)$ Offset\nRaw Logs",
             fc=C_BLUE_BG, ec=C_BLUE_BD, lw=1.35, title_size=8.4, sub_size=7.0, rounding=0.10)
    draw_box(ax, 9.74, 1.82, 1.30, 1.56, "Temporal KG\n& ANN Index", "Bi-Temporal\n[valid_from,\nvalid_to)\nNon-Destructive",
             fc=C_TEAL_BG, ec=C_TEAL_BD, lw=1.35, title_size=8.2, sub_size=6.8, rounding=0.10, title_offset=0.28, sub_offset=0.18)

    # === FAST PATH LANE ===
    fast_bg = FancyBboxPatch(
        (3.16, 3.96), 5.18, 1.30,
        boxstyle="round,pad=0.01,rounding_size=0.10",
        facecolor=C_BLUE_BG, edgecolor=C_BLUE_BD, lw=1.0, ls=":", alpha=0.55, zorder=1,
    )
    ax.add_patch(fast_bg)
    ax.text(5.75, 5.12, "Fast Path  (Write-Back WAL  —  Crash Consistency)",
            ha="center", va="center", fontsize=8.1, fontweight="bold", color=C_BLUE_BD, zorder=2)

    draw_box(ax, 3.52, 4.12, 2.60, 0.80, "Log Buffer (FIFO Queue)", "Non-Blocking Ingest (LSN-Ordered)",
             fc="#FFFFFF", ec=C_BLUE_BD, lw=1.3, title_size=8.5, sub_size=7.1, rounding=0.09)

    # === SMART PATH LANE (Title shifted right of x=4.05 arrow -> ZERO crossing!) ===
    smart_bg = FancyBboxPatch(
        (3.16, 1.62), 5.18, 2.02,
        boxstyle="round,pad=0.01,rounding_size=0.10",
        facecolor=C_TEAL_BG, edgecolor=C_TEAL_BD, lw=1.0, ls=":", alpha=0.50, zorder=1,
    )
    ax.add_patch(smart_bg)
    ax.text(6.32, 3.49, "Smart Path  (Async Sleep Consolidation — Eventual Cons.)",
            ha="center", va="center", fontsize=7.9, fontweight="bold", color=C_TEAL_BD, zorder=2)

    draw_box(ax, 3.30, 2.50, 1.56, 0.80, "Amygdala Gate", "Zero-LLM Salience\n$S(x)$ Admission",
             fc="#DCFCE7", ec=C_TEAL_BD, lw=1.45, title_size=8.2, sub_size=6.8, rounding=0.09)
    draw_box(ax, 5.26, 2.50, 2.28, 0.80, "Replay Buffer", "Priority Queue\n(Normal + Fast-Track)",
             fc="#FFFFFF", ec=C_TEAL_BD, lw=1.3, title_size=8.2, sub_size=6.9, rounding=0.09)
    draw_box(ax, 5.12, 1.74, 2.56, 0.48, "SLM Extractor (Sleep Batch)",
             fc="#FFFFFF", ec=C_TEAL_BD, lw=1.3, title_size=8.0, rounding=0.08)

    # Low-salience skip note cleanly placed below Amygdala Gate
    label_pill(ax, 4.08, 1.94, "Skip SLM if $S(x) < \\theta_{\\mathrm{low}}$\n(Log-Only, -44%~93% Write)",
               color=C_MUTED, size=6.6, bg="#FFFFFF", ec="#CBD5E1")

    # === ARROWS ===
    draw_arrow(ax, (1.06, 4.52), (1.80, 4.52), color=C_BLUE_BD, lw=1.6)
    label_pill(ax, 1.43, 4.78, "DMP.Store\n(Async)", color=C_BLUE_BD, size=7.0, bold=True)

    draw_arrow(ax, (2.88, 4.52), (3.50, 4.52), color=C_BLUE_BD, lw=1.5)
    label_pill(ax, 3.19, 4.72, "Write-Back", color=C_BLUE_BD, size=6.8)

    draw_arrow(ax, (6.14, 4.52), (8.58, 4.52), color=C_BLUE_BD, lw=1.6)
    label_pill(ax, 7.36, 4.70, "Direct Append (Crash Cons.)", color=C_BLUE_BD, size=7.1, bold=True)
    draw_arrow(ax, (9.54, 4.52), (9.72, 4.52), color=C_BLUE_BD, lw=1.5)

    # Log Buffer -> Amygdala Gate (passes cleanly at x=3.92, well LEFT of Smart Path title!)
    draw_arrow(ax, (3.92, 4.10), (3.92, 3.32), color=C_TEAL_BD, lw=1.5)
    label_pill(ax, 4.44, 3.78, "Async Fetch", color=C_TEAL_BD, size=7.0)

    draw_arrow(ax, (4.88, 2.90), (5.24, 2.90), color=C_TEAL_BD, lw=1.4)
    draw_arrow(ax, (4.08, 2.48), (4.08, 2.18), color=C_MUTED, lw=1.15, ls="--")

    # Visible vertical arrow Replay Buffer -> SLM Extractor (height=0.26)
    draw_arrow(ax, (6.40, 2.48), (6.40, 2.24), color=C_TEAL_BD, lw=1.45, ms=10.5)

    # SLM Extractor -> Persistence Manager (wide clean gap x=7.70..8.58!)
    draw_arrow(ax, (7.70, 1.98), (8.58, 1.98), color=C_TEAL_BD, lw=1.55)
    label_pill(ax, 8.14, 2.34, "Bi-Temporal\nUpsert\n(Eventual)", color=C_TEAL_BD, size=6.7, bold=True, bg="#ECFDF5")

    draw_arrow(ax, (9.54, 2.60), (9.72, 2.60), color=C_TEAL_BD, lw=1.5)

    # Bottom Read & Watermark Sync Paths
    draw_arrow(ax, (1.06, 1.16), (1.80, 1.16), color=C_INK, style="<->", lw=1.5)
    label_pill(ax, 1.43, 1.42, "DMP.Load\n/ ANN", color=C_INK, size=6.9, bold=True)

    draw_arrow(ax, (2.88, 1.16), (8.58, 1.16), color=C_INK, style="<->", lw=1.4)
    label_pill(ax, 5.73, 1.16, "Sync Read: LoadByID(KG / WAL Offset)   |   Fallback SearchANN()", color=C_INK, size=7.3, bold=True)

    draw_arrow(ax, (8.58, 0.72), (2.88, 0.72), color=C_AMBER_BD, lw=1.5, ls="--")
    draw_arrow(ax, (1.80, 0.72), (1.06, 0.72), color=C_AMBER_BD, lw=1.5, ls="--")
    label_pill(ax, 5.73, 0.72, "Piggybacked Watermark (consolidated_lsn)  →  Clears Client DTF & Store Buffer",
               color=C_AMBER_BD, size=7.2, bold=True, bg="#FFFBEB", ec=C_AMBER_BD)

    save_fig(fig, "fig3_adapter")


# =============================================================================
# FIGURE 4: Confidence-Gated Cascaded Retrieval State Machine
# =============================================================================

def build_fig4() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.9))
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 5.9)
    ax.axis("off")

    # Input: Query & Hybrid Encoder (wider, clean 2-line title!)
    draw_box(ax, 0.14, 2.52, 1.54, 1.00, "Query &\nHybrid Encoder", "Fine & Coarse\nTopicKeys",
             fc=C_SLATE_BG, ec=C_INK, lw=1.4, title_size=8.3, sub_size=6.8, rounding=0.12,
             title_offset=0.20, sub_offset=0.22)

    # === STAGE 1: LOCAL BOUNDARY ===
    st1 = FancyBboxPatch(
        (2.02, 0.36), 3.46, 4.86,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=C_TEAL_BG, edgecolor=C_TEAL_BD, lw=1.45, ls="--", alpha=0.45, zorder=0,
    )
    ax.add_patch(st1)
    ax.text(3.75, 5.04, "Stage 1: Local Boundary (~0 ms, 0 Net Bytes)",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color=C_TEAL_BD, zorder=1)

    draw_box(ax, 2.20, 3.92, 2.86, 0.84, "Store Buffer Co-Scan (RYOW)",
             "If lsn > client_wm: scan un-consolidated ring\n(<0.1 ms, Paraphrase-Safe)",
             fc=C_AMBER_BG, ec=C_AMBER_BD, lw=1.35, title_size=8.0, sub_size=6.6, rounding=0.09)
    draw_box(ax, 2.20, 2.84, 2.86, 0.80, "L2-Cache (Local Raw)",
             "Recent Session Turns  +  Tag Verification",
             fc="#FFFFFF", ec=C_BLUE_BD, lw=1.3, title_size=8.1, sub_size=6.8, rounding=0.09)
    draw_box(ax, 2.20, 1.76, 2.86, 0.80, "L3-Cache (Local KG)",
             "Consolidated Slices  +  Tag Verification",
             fc="#FFFFFF", ec=C_TEAL_BD, lw=1.3, title_size=8.1, sub_size=6.8, rounding=0.09)

    draw_box(ax, 2.20, 0.56, 2.86, 0.86, "Non-Exclusive Local Fusion &\nConfidence Gate",
             "$\\mathrm{top\\_sim} \\geq \\tau_{\\mathrm{local}}$  AND  $\\mathrm{hits} \\geq k_{\\min}$ ?",
             fc="#EFF6FF", ec=C_BLUE_BD, lw=1.45, title_size=7.9, sub_size=7.0, rounding=0.12,
             title_offset=0.20, sub_offset=0.22)

    # === STAGE 2 & 3: OVER DMP BOUNDARY ===
    st2 = FancyBboxPatch(
        (5.96, 0.36), 5.66, 4.86,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor="#FFF7F7", edgecolor=C_RED_BD, lw=1.45, ls="--", alpha=0.45, zorder=0,
    )
    ax.add_patch(st2)
    ax.text(8.79, 5.04, "Stage 2 & 3: Over DMP Boundary (Remote Server)",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color=C_RED_BD, zorder=1)

    draw_box(ax, 6.16, 3.74, 2.18, 0.96, "TLB: K-Cache",
             "$O(1)$ KG NodeIDs\n(Semantic-First Paging)",
             fc=C_PURPLE_BG, ec=C_PURPLE_BD, lw=1.35, title_size=8.3, sub_size=6.8, rounding=0.09)
    draw_box(ax, 9.00, 3.74, 2.16, 0.96, "Temporal KG",
             "DMP.LoadByID(KG)\nSmall Payload (Compact)",
             fc="#FFFFFF", ec=C_TEAL_BD, lw=1.35, title_size=8.3, sub_size=6.8, rounding=0.09)

    draw_box(ax, 6.16, 2.24, 2.18, 0.96, "TLB: L-Cache",
             "$O(1)$ WAL Offsets\n(Compute → I/O Offload)",
             fc=C_PURPLE_BG, ec=C_PURPLE_BD, lw=1.35, title_size=8.3, sub_size=6.8, rounding=0.09)
    draw_box(ax, 9.00, 2.24, 2.16, 0.96, "Log Buffer / Archive",
             "DMP.LoadByID(LOG)\nLarge Payload, Zero ANN",
             fc="#FFFFFF", ec=C_BLUE_BD, lw=1.35, title_size=8.3, sub_size=6.8, rounding=0.09)

    draw_box(ax, 6.50, 0.56, 4.66, 0.96, "Red Path (Last Resort): High-Compute Server ANN",
             "DMP.SearchANN() over Temporal KG (+ WAL if Dirty)\nOnly ~5%–11% Queries Escalate Here  →  Backfill K-TLB & L3",
             fc=C_RED_BG, ec=C_RED_BD, lw=1.6, title_size=8.2, sub_size=6.9, rounding=0.10, title_color=C_RED_BD)

    # === ARROWS ===
    # Query -> parallel fan-out into Store Buffer, L2, L3 (TRUE parallel non-exclusive!)
    ax.plot([1.70, 1.88], [3.02, 3.02], color=C_INK, lw=1.3, zorder=5)
    draw_ortho_arrow(ax, [(1.88, 3.02), (1.88, 4.34), (2.18, 4.34)], color=C_AMBER_BD, lw=1.3)
    draw_ortho_arrow(ax, [(1.88, 3.02), (1.88, 3.24), (2.18, 3.24)], color=C_BLUE_BD, lw=1.3)
    draw_ortho_arrow(ax, [(1.88, 3.02), (1.88, 2.16), (2.18, 2.16)], color=C_TEAL_BD, lw=1.3)

    # Parallel outputs of Store Buffer, L2, L3 -> vertical collector rail (x=5.26) -> Fusion Gate!
    ax.plot([5.08, 5.26], [4.34, 4.34], color=C_AMBER_BD, lw=1.3, zorder=5)
    ax.plot([5.08, 5.26], [3.24, 3.24], color=C_BLUE_BD, lw=1.3, zorder=5)
    ax.plot([5.08, 5.26], [2.16, 2.16], color=C_TEAL_BD, lw=1.3, zorder=5)
    draw_ortho_arrow(ax, [(5.26, 4.34), (5.26, 1.16), (5.08, 1.16)], color=C_BLUE_BD, lw=1.35)

    # Local Hit return pill (bottom left)
    draw_ortho_arrow(ax, [(2.18, 0.99), (0.92, 0.99), (0.92, 0.44)], color=C_TEAL_BD, lw=1.65)
    label_pill(ax, 0.92, 0.24, "Local Hit (~63%)\n→ Context Assembler", color=C_TEAL_BD, size=7.2, bold=True, bg=C_TEAL_BG, ec=C_TEAL_BD)

    # Escalate Gate -> K-Cache over DMP
    draw_ortho_arrow(ax, [(5.08, 0.84), (5.72, 0.84), (5.72, 4.22), (6.14, 4.22)], color=C_PURPLE_BD, lw=1.55)
    label_pill(ax, 5.72, 2.66, "Local Miss\n/ Low Conf.", color=C_PURPLE_BD, size=6.9, bold=True, bg="#FFFFFF", ec=C_PURPLE_BD)

    # K-Cache -> Temporal KG (Hit) & K-Cache -> L-Cache (Miss)
    draw_arrow(ax, (8.36, 4.22), (8.98, 4.22), color=C_TEAL_BD, lw=1.55)
    label_pill(ax, 8.67, 4.42, "Hit", color=C_TEAL_BD, size=7.2, bold=True)

    draw_arrow(ax, (7.25, 3.72), (7.25, 3.22), color=C_RED_BD, lw=1.45)
    label_pill(ax, 7.66, 3.47, "Miss / Low", color=C_RED_BD, size=6.9)

    # L-Cache -> Log Archive (Hit) & L-Cache -> Red Path ANN (Miss)
    draw_arrow(ax, (8.36, 2.72), (8.98, 2.72), color=C_BLUE_BD, lw=1.55)
    label_pill(ax, 8.67, 2.92, "Hit", color=C_BLUE_BD, size=7.2, bold=True)

    draw_arrow(ax, (7.25, 2.22), (7.25, 1.54), color=C_RED_BD, lw=1.65)
    label_pill(ax, 7.98, 1.88, "Miss (Last Resort)", color=C_RED_BD, size=7.0, bold=True, bg=C_RED_BG)

    # Unified Remote Return + Backfill Rail ABOVE Stage 1 & 2 frames -> terminates at Context Assembler box (top-left open space!)
    draw_box(ax, 0.14, 5.14, 1.54, 0.62, "Context\nAssembler", "Zone B/C Merge",
             fc=C_TEAL_BG, ec=C_TEAL_BD, lw=1.4, title_size=8.1, sub_size=6.6, rounding=0.10,
             title_offset=0.16, sub_offset=0.25)
    ax.plot([11.18, 11.40], [4.22, 4.22], color=C_TEAL_BD, lw=1.2, ls="--", zorder=5)
    ax.plot([11.18, 11.40], [2.72, 2.72], color=C_TEAL_BD, lw=1.2, ls="--", zorder=5)
    ax.plot([11.18, 11.40], [1.04, 1.04], color=C_TEAL_BD, lw=1.2, ls="--", zorder=5)
    draw_ortho_arrow(ax, [(11.40, 1.04), (11.40, 5.45), (1.70, 5.45)], color=C_TEAL_BD, lw=1.35, ls="--")
    label_pill(ax, 6.55, 5.45, "Return Payload to Assembler  +  Async Backfill L3 & K-TLB",
               color=C_TEAL_BD, size=7.2, bold=True, bg="#FFFFFF", ec=C_TEAL_BD)

    save_fig(fig, "fig4_cascade")


def main() -> None:
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    print(f"Regenerated 4 polished vector figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
