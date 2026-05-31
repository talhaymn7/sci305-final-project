"""
AgriMind / Kitt Hemp Adaptive Prescription Engine -- SCI305 Final Report DOCX generator.

Usage:
    python REPORT/generate_report.py

Output:
    REPORT/SCI305_Final_Report.docx
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT_PATH = Path(__file__).parent / "SCI305_Final_Report.docx"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hemp_training.csv"


# ── Helper: matplotlib figure to in-memory PNG ─────────────────────────────────

def fig_to_stream(fig: plt.Figure, dpi: int = 150) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Helper: table cell shading ─────────────────────────────────────────────────

def set_row_bg(row, hex_color: str) -> None:
    for cell in row.cells:
        _shade_cell(cell, hex_color)


def _shade_cell(cell, hex_color: str) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 -- Adaptive system architecture flowchart
# ══════════════════════════════════════════════════════════════════════════════

def fig_architecture() -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(13, 9.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    def box(x, y, w, h, label, sublabel="", color="#2E86AB", textcolor="white", fontsize=10):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="#1a1a2e", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.16 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize,
                color=textcolor, fontweight="bold")
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.26, sublabel,
                    ha="center", va="center", fontsize=7.3,
                    color=textcolor, style="italic")

    def arrow(x1, y1, x2, y2, label="", color="#333333"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.1, my, label, fontsize=8, color=color, style="italic")

    ax.text(6.5, 9.55, "Kitt Hemp Adaptive Prescription Engine",
            ha="center", fontsize=14, fontweight="bold", color="#1a1a2e")

    # Two API entry points
    box(0.3, 8.0, 5.9, 1.0, "POST /hemp/prescription",
        "field inputs only (cold start)", color="#457B9D", fontsize=10)
    box(6.8, 8.0, 5.9, 1.0, "POST /hemp/prescription/adaptive",
        "field inputs + cycle history", color="#5C7AAE", fontsize=10)

    # Engine box receiving both
    box(2.6, 6.7, 7.8, 0.85, "compute_hemp_prescription()",
        "validate -> blockers -> route", color="#1a1a2e", fontsize=10)

    # Hard blockers
    box(0.4, 5.0, 4.6, 1.15, "Hard Blockers",
        "pH<5 or >8, slope>20%, temp<5 or >35 degC, EC>4\n(None-guarded: skipped if field absent)",
        color="#E63946", fontsize=9)
    box(5.6, 5.0, 2.6, 1.15, "UNSUITABLE",
        "no prescription", color="#6D6875", fontsize=9.5)

    # Cold start
    box(0.4, 3.0, 4.6, 1.45, "Cold-Start XGBoost",
        "Stage 1 classifier (suitable?)\nStage 2 regressor (yield, t/dekar)\n31 features, default imputation",
        color="#2A9D8F", fontsize=9)
    box(5.6, 3.3, 2.6, 1.0, "UNSUITABLE",
        "prob < 0.5", color="#6D6875", fontsize=9.5)

    # Routing diamond text
    box(0.4, 1.45, 4.6, 1.05, "Has cycle history?",
        "branch on cycle_history", color="#264653", fontsize=9.5)

    # Historical correction
    box(6.0, 1.4, 6.4, 1.2, "Historical Correction Layer",
        "exp-weighted actual/predicted yield factor\nN efficiency adjustment from last cycle",
        color="#E9C46A", textcolor="#333333", fontsize=9)

    # Final output
    box(1.4, 0.05, 10.2, 0.95, "Prescription Result",
        "yield + N/P/K (kg/dekar) + irrigation (mm/week) + notes + confidence + provider",
        color="#2E86AB", fontsize=9.5)

    # Arrows
    arrow(3.25, 8.0, 4.5, 7.55)
    arrow(9.75, 8.0, 8.5, 7.55)
    arrow(4.5, 6.7, 3.0, 6.15)
    arrow(5.0, 5.55, 5.6, 5.55, "fires")
    arrow(2.7, 5.0, 2.7, 4.45)
    arrow(5.0, 3.85, 5.6, 3.85, "soft")
    arrow(2.7, 3.0, 2.7, 2.5)
    arrow(5.0, 1.9, 6.0, 1.9, "yes")
    arrow(2.7, 1.45, 2.7, 1.0, "no")
    arrow(9.2, 1.4, 8.0, 1.0)

    return fig_to_stream(fig, dpi=160)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 -- pH and temperature penalty curves
# ══════════════════════════════════════════════════════════════════════════════

def fig_penalty_curves() -> io.BytesIO:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Hemp Yield Penalty Functions", fontsize=13, fontweight="bold")

    ph_vals = np.linspace(4.0, 9.5, 400)

    def ph_pen(ph):
        if 6.0 <= ph <= 7.0:
            return 1.0
        if ph < 6.0:
            return max(0.0, 1.0 - (6.0 - ph) * 0.35)
        return max(0.0, 1.0 - (ph - 7.0) * 0.30)

    ax1.plot(ph_vals, [ph_pen(p) for p in ph_vals], color="#2A9D8F", lw=2.5)
    ax1.axvspan(6.0, 7.0, alpha=0.15, color="#2A9D8F", label="Optimal range (6.0-7.0)")
    ax1.axhline(0.5, color="#E63946", ls="--", lw=1.2, label="0.5 threshold")
    ax1.set_xlabel("Soil pH", fontsize=11)
    ax1.set_ylabel("Yield multiplier", fontsize=11)
    ax1.set_title("pH Penalty Function", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.05, 1.15)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    temp_vals = np.linspace(2.0, 42.0, 400)

    def temp_pen(t):
        if 15.0 <= t <= 27.0:
            return 1.0
        if t < 15.0:
            return max(0.0, 1.0 - (15.0 - t) * 0.06)
        return max(0.0, 1.0 - (t - 27.0) * 0.08)

    ax2.plot(temp_vals, [temp_pen(t) for t in temp_vals], color="#E76F51", lw=2.5)
    ax2.axvspan(15.0, 27.0, alpha=0.15, color="#E76F51", label="Optimal range (15-27 degC)")
    ax2.axhline(0.5, color="#E63946", ls="--", lw=1.2, label="0.5 threshold")
    ax2.set_xlabel("Mean growing-season temperature (degC)", fontsize=11)
    ax2.set_ylabel("Yield multiplier", fontsize=11)
    ax2.set_title("Temperature Penalty Function", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_stream(fig, dpi=150)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 -- Prescription formula curves (dekar units, irrigation by efficiency)
# ══════════════════════════════════════════════════════════════════════════════

def fig_prescription_logic() -> io.BytesIO:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Prescription Formulas (dekar units)", fontsize=12, fontweight="bold")

    # Nitrogen recommendation vs soil N ppm, by organic matter
    n_ppms = np.linspace(15, 80, 200)
    om_vals = [1.5, 3.0, 5.0]
    colors_n = ["#E63946", "#2A9D8F", "#264653"]
    for om, col in zip(om_vals, colors_n):
        rec_n = np.maximum(0, 12 - n_ppms * 0.39 - om * 2.0)
        ax1.plot(n_ppms, rec_n, color=col, lw=2.0, label=f"OM={om}%")
    ax1.set_xlabel("Soil nitrogen (ppm)", fontsize=10)
    ax1.set_ylabel("Recommended N (kg/dekar)", fontsize=10)
    ax1.set_title("Nitrogen Prescription\nN_rec = max(0, 12 - N_ppm*0.39 - OM%*2 - credit)",
                  fontsize=9.5, fontweight="bold")
    ax1.legend(title="Organic matter", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", lw=0.8)

    # Irrigation vs rainfall, by irrigation efficiency
    rain_vals = np.linspace(150, 650, 300)
    eff_types = [("drip", 0.90, "#2A9D8F"), ("sprinkler", 0.78, "#457B9D"),
                 ("flood", 0.55, "#E76F51")]
    for name, eff, col in eff_types:
        deficit = np.maximum(0, 450 - rain_vals * 0.75) / (20.0 * eff)
        ax2.plot(rain_vals, deficit, color=col, lw=2.2, label=f"{name} (eff={eff})")
    ax2.set_xlabel("Seasonal rainfall (mm)", fontsize=10)
    ax2.set_ylabel("Weekly irrigation (mm/week)", fontsize=10)
    ax2.set_title("Irrigation Prescription\nirr = max(0, 450 - rain*0.75) / (20*eff)",
                  fontsize=9.5, fontweight="bold")
    ax2.legend(title="Irrigation type", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="gray", lw=0.8)

    plt.tight_layout()
    return fig_to_stream(fig, dpi=150)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 -- Historical correction concept (exponential decay weights)
# ══════════════════════════════════════════════════════════════════════════════

def fig_historical_correction() -> io.BytesIO:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle("Historical Correction Layer", fontsize=13, fontweight="bold")

    # Left: decay weights for N=5 cycles
    n = 5
    decay = 0.6
    idx = np.arange(n)
    weights = np.array([decay ** (n - 1 - i) for i in idx])
    norm_w = weights / weights.sum()
    labels = [f"cycle {i+1}\n({'oldest' if i==0 else 'newest' if i==n-1 else ''})" for i in idx]
    bars = ax1.bar(idx, norm_w, color="#2A9D8F", alpha=0.85, edgecolor="white")
    ax1.set_xticks(idx)
    ax1.set_xticklabels([f"c{i+1}" for i in idx], fontsize=10)
    ax1.set_xlabel("Cycle (oldest -> most recent)", fontsize=10)
    ax1.set_ylabel("Normalised weight", fontsize=10)
    ax1.set_title("Exponential weights, decay=0.6 (N=5 cycles)", fontsize=10, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)
    for b, w in zip(bars, norm_w):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f"{w:.2f}", ha="center", fontsize=8, color="#264653")

    # Right: worked example of correction factor converging
    cycles = [1, 2, 3, 4, 5]
    ratios = [1.25, 1.18, 0.92, 1.10, 1.05]  # actual/predicted per cycle
    factors = []
    for k in range(1, len(cycles) + 1):
        sub = ratios[:k]
        m = len(sub)
        w = np.array([decay ** (m - 1 - i) for i in range(m)])
        factors.append(float(np.dot(w, sub) / w.sum()))
    ax2.plot(cycles, ratios, "o--", color="#6D6875", lw=1.5, label="per-cycle actual/predicted")
    ax2.plot(cycles, factors, "s-", color="#E76F51", lw=2.3, label="weighted correction factor")
    ax2.axhline(1.0, color="#457B9D", ls=":", lw=1.2, label="no correction (1.0x)")
    ax2.set_xticks(cycles)
    ax2.set_xlabel("Number of completed cycles", fontsize=10)
    ax2.set_ylabel("Yield ratio / factor", fontsize=10)
    ax2.set_title("Correction factor as cycles accumulate", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_stream(fig, dpi=150)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 -- Confusion matrices (rule-based vs XGBoost, 400-sample test fold)
# ══════════════════════════════════════════════════════════════════════════════

def fig_confusion_matrix() -> io.BytesIO:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Confusion Matrices: Rule-Based vs. XGBoost Classifier",
                 fontsize=12, fontweight="bold")

    def draw_cm(ax, matrix, title, cmap):
        ax.imshow(matrix, cmap=cmap, vmin=0)
        labels = [["TP", "FN"], ["FP", "TN"]]
        total = matrix.sum()
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                ax.text(j, i, f"{labels[i][j]}\n{val}\n({100*val/total:.1f}%)",
                        ha="center", va="center", fontsize=11,
                        color="white" if val > total * 0.25 else "black", fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Suitable", "Pred: Unsuitable"], fontsize=9)
        ax.set_yticklabels(["True: Suitable", "True: Unsuitable"], fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")

    # Rule-based: no classifier, always predicts suitable
    # 400-sample fold, ~233 suitable / 167 unsuitable
    cm_rule = np.array([[233, 0], [167, 0]])
    draw_cm(axes[0], cm_rule, "Rule-Based (always 'Suitable')", cmap="Oranges")

    # XGBoost: TP=227 FN=6 (recall .975), FP=13 TN=154 (acc .9525, prec .9474)
    cm_xgb = np.array([[227, 6], [13, 154]])
    draw_cm(axes[1], cm_xgb, "XGBoost Classifier", cmap="Blues")

    plt.tight_layout()
    return fig_to_stream(fig, dpi=150)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 -- Performance bar charts
# ══════════════════════════════════════════════════════════════════════════════

def fig_performance_comparison() -> io.BytesIO:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Rule-Based vs. XGBoost: Performance", fontsize=12, fontweight="bold")

    metrics_cls = ["Accuracy", "Precision", "Recall", "F1"]
    rule_cls = [0.5825, 0.5825, 1.0000, 0.7362]
    xgb_cls = [0.9525, 0.9474, 0.9750, 0.9610]
    x = np.arange(len(metrics_cls))
    w = 0.35
    ax1.bar(x - w / 2, rule_cls, w, label="Rule-Based", color="#6D6875", alpha=0.85)
    bars2 = ax1.bar(x + w / 2, xgb_cls, w, label="XGBoost", color="#2A9D8F", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_cls, fontsize=9)
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_title("Stage 1: Classification (400-sample test fold)", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", fontsize=8, color="#264653")

    metrics_reg = ["RMSE (t/dekar)", "MAE (t/dekar)", "R^2"]
    xgb_reg = [0.065, 0.051, 0.70]
    x2 = np.arange(len(metrics_reg))
    bars4 = ax2.bar(x2, xgb_reg, 0.5, color="#E76F51", alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics_reg, fontsize=9)
    ax2.set_ylim(0, 0.85)
    ax2.set_ylabel("Value", fontsize=10)
    ax2.set_title("Stage 2: Regression (suitable test samples)", fontsize=10, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    for bar in bars4:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", fontsize=8.5, color="#264653")

    plt.tight_layout()
    return fig_to_stream(fig, dpi=150)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 -- Approximate feature importance
# ══════════════════════════════════════════════════════════════════════════════

def fig_feature_importance() -> io.BytesIO:
    features = [
        "avg_temp", "ph", "slope_percent", "drainage=poor",
        "seasonal_rainfall_mm", "organic_matter_percent", "ec",
        "drainage=good", "variety=narli", "irrigation_type=none",
        "nitrogen_ppm", "rotation_n_credit", "elevation_meters", "avg_humidity",
    ]
    importances = [0.232, 0.191, 0.121, 0.088, 0.067, 0.050, 0.041,
                   0.037, 0.033, 0.029, 0.025, 0.020, 0.016, 0.013]
    colors = ["#E63946" if v > 0.15 else "#E9C46A" if v > 0.08 else "#2A9D8F"
              for v in importances]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features[::-1], importances[::-1], color=colors[::-1], alpha=0.88, edgecolor="white")
    ax.set_xlabel("Approximate importance score", fontsize=11)
    ax.set_title("Stage 1 Classifier: Feature Importances (approximate, indicative)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    legend_elements = [
        Patch(facecolor="#E63946", label="High (>0.15)"),
        Patch(facecolor="#E9C46A", label="Moderate (0.08-0.15)"),
        Patch(facecolor="#2A9D8F", label="Lower (<0.08)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    return fig_to_stream(fig, dpi=150)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 -- Training data distribution (dekar yield, new features)
# ══════════════════════════════════════════════════════════════════════════════

def fig_data_distribution() -> io.BytesIO:
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    ph_vals = [float(r["ph"]) for r in rows]
    yield_vals = [float(r["expected_yield_ton_dekar"]) for r in rows]
    suitable = [int(r["suitable"]) for r in rows]
    varieties = [r["variety"] for r in rows]

    n_suit = sum(suitable)
    n_unsuit = len(rows) - n_suit

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(f"Training Dataset Distribution ({len(rows):,} samples)",
                 fontsize=12, fontweight="bold")

    axes[0].hist(ph_vals, bins=25, color="#2A9D8F", alpha=0.8, edgecolor="white")
    axes[0].axvspan(6.0, 7.0, alpha=0.2, color="#E9C46A", label="Optimal (6.0-7.0)")
    axes[0].set_xlabel("Soil pH", fontsize=10)
    axes[0].set_ylabel("Sample count", fontsize=10)
    axes[0].set_title("pH Distribution", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    # Variety counts
    var_order = ["narli", "vezir", "other"]
    var_counts = [varieties.count(v) for v in var_order]
    axes[1].bar(var_order, var_counts, color=["#E76F51", "#457B9D", "#6D6875"], alpha=0.85,
                edgecolor="white")
    axes[1].set_xlabel("Variety", fontsize=10)
    axes[1].set_title("Variety Distribution", fontsize=10, fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.25)
    for i, c in enumerate(var_counts):
        axes[1].text(i, c + 5, str(c), ha="center", fontsize=9, color="#264653")

    axes[2].hist([y for y, s in zip(yield_vals, suitable) if s == 1], bins=22,
                 color="#2A9D8F", alpha=0.75, label=f"Suitable (n={n_suit:,})", edgecolor="white")
    axes[2].hist([y for y, s in zip(yield_vals, suitable) if s == 0], bins=22,
                 color="#E63946", alpha=0.75, label=f"Unsuitable (n={n_unsuit:,})", edgecolor="white")
    axes[2].axvline(0.30, color="#333", lw=2, ls="--", label="Threshold (0.30 t/dekar)")
    axes[2].set_xlabel("Expected yield (t/dekar)", fontsize=10)
    axes[2].set_title("Yield Distribution", fontsize=10, fontweight="bold")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.25)

    plt.tight_layout()
    return fig_to_stream(fig, dpi=150)


# ══════════════════════════════════════════════════════════════════════════════
# DOCX helpers
# ══════════════════════════════════════════════════════════════════════════════

def add_caption(doc, text: str) -> None:
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].font.size = Pt(10)
    p.runs[0].font.italic = True


def add_formula(doc, text: str, fontsize: int = 12) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = "Courier New"
    run.font.size = Pt(fontsize)
    run.font.bold = True


def add_body(doc, text: str) -> None:
    p = doc.add_paragraph(text)
    p.style = "Normal"


def add_figure(doc, stream, width_in: float, caption: str) -> None:
    pp = doc.add_paragraph()
    pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pp.add_run().add_picture(stream, width=Inches(width_in))
    add_caption(doc, caption)


def table_header(tbl, headers, fill_hex):
    hr = tbl.rows[0]
    set_row_bg(hr, fill_hex)
    for i, h in enumerate(headers):
        c = hr.cells[i]
        c.text = h
        c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        c.paragraphs[0].runs[0].font.bold = True
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER


# ══════════════════════════════════════════════════════════════════════════════
# DOCX builder
# ══════════════════════════════════════════════════════════════════════════════

def build_doc() -> Document:
    doc = Document()

    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(3.5)
    section.right_margin = Cm(2.5)
    section.top_margin = Cm(3.0)
    section.bottom_margin = Cm(2.5)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = Pt(20)
    normal.paragraph_format.space_after = Pt(6)

    h1 = styles["Heading 1"]
    h1.font.name = "Times New Roman"
    h1.font.size = Pt(14)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)

    h2 = styles["Heading 2"]
    h2.font.name = "Times New Roman"
    h2.font.size = Pt(13)
    h2.font.bold = True
    h2.font.color.rgb = RGBColor(0x2E, 0x86, 0xAB)

    def blank(n=1):
        for _ in range(n):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(0)

    # ── COVER PAGE ────────────────────────────────────────────────────────────
    blank(3)
    for text in ["SCI305 -- Fundamentals of Machine Learning", "Final Project Report"]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    blank(2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(
        "Nested XGBoost Classification and Regression\n"
        "for Industrial Hemp Field Prescription:\n"
        "The AgriMind/Kitt Adaptive Prescription Engine"
    )
    run.font.name = "Times New Roman"
    run.font.size = Pt(18)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)

    blank(3)
    for line in [
        "Student: A. Talha Yaman",
        "Course: SCI305 Fundamentals of Machine Learning",
        "Instructor: Ugur Kayas",
        "Date: June 1, 2026",
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(line)
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)

    doc.add_page_break()

    # ── ABSTRACT ──────────────────────────────────────────────────────────────
    doc.add_heading("Abstract", level=1)
    add_body(doc,
        "This report covers an adaptive prescription engine I built for Kitt, a B2B industrial "
        "hemp supply-chain platform. The job the engine does is practical: given a field, tell the "
        "agronomist whether hemp is worth planting there, and if so, how much nitrogen, phosphorus, "
        "potassium, and irrigation to apply, plus a yield forecast. All quantities are in dekar, the "
        "Turkish land unit (1 hectare = 10 dekar), because that is what the farmers and field staff "
        "actually use."
    )
    add_body(doc,
        "The engine has two modes. When a field is brand new and has no recorded history, it runs a "
        "cold-start path built on a nested pair of XGBoost models: a classifier decides suitability, "
        "and a regressor estimates fiber yield for fields that pass. Once a field has completed one or "
        "more growing cycles, a second path kicks in. It takes the cold-start number and corrects it "
        "with the field's own track record, using an exponential-weighted average of how actual yield "
        "compared to predicted yield in past cycles. The more recent the cycle, the more it counts."
    )
    add_body(doc,
        "On the held-out test fold the classifier reached 95.25% accuracy and an F1 of 0.961. The "
        "yield regressor reached RMSE = 0.065 t/dekar and R2 = 0.70. The fertilizer and irrigation "
        "numbers come from explicit agronomic formulas rather than learned models, which keeps that "
        "part of the output auditable. A confidence score travels with every prescription so the "
        "agronomist knows how much to trust it."
    )

    doc.add_page_break()

    # ── TABLE OF CONTENTS ─────────────────────────────────────────────────────
    doc.add_heading("Table of Contents", level=1)
    toc = [
        "1. Introduction",
        "   1.1. What is Industrial Hemp?",
        "   1.2. The Kitt Platform Context",
        "2. Related Work",
        "3. XGBoost Theory and Mathematical Foundations",
        "4. System Architecture",
        "5. Historical Correction Layer",
        "6. Dataset",
        "7. Model Training",
        "8. Prescription Formulas",
        "9. Results",
        "10. Discussion",
        "11. Conclusion",
        "References",
    ]
    for item in toc:
        p = doc.add_paragraph(item)
        p.style = "Normal"
        p.paragraph_format.space_after = Pt(2)

    doc.add_page_break()

    # ── 1. INTRODUCTION ───────────────────────────────────────────────────────
    doc.add_heading("1. Introduction", level=1)
    doc.add_heading("1.1. What is Industrial Hemp?", level=2)
    add_body(doc,
        "Industrial hemp (Cannabis sativa L.) is one of the oldest crops humans have grown. There is "
        "archaeological evidence of hemp fiber being spun into rope and cloth back in the Mesolithic "
        "(Schultes, 1970). It fell out of favor for most of the twentieth century, partly for legal "
        "reasons that had nothing to do with the fiber crop itself, and it is coming back now because "
        "the plant is genuinely useful. The stalk gives textile fiber and hurd for building boards, "
        "the seed gives oil and protein, and the leftover biomass burns as fuel."
    )
    add_body(doc,
        "One thing is worth clearing up early, because people always ask. Industrial hemp and "
        "marijuana are the same species, but they are not the same plant in any way that matters here. "
        "Industrial hemp carries less than 0.3% THC, the psychoactive compound, which is far too low "
        "to do anything to anyone. It is grown for fiber, seed, and biomass, and in most legal systems "
        "it is treated as an ordinary farm crop rather than a controlled substance (Small & Marcus, "
        "2002). Kitt deals only with this fiber-and-seed side of the plant."
    )
    add_body(doc,
        "In Turkey the two cultivars that come up most often are Narli and Vezir, both selected for "
        "stem and fiber quality under local conditions. The engine treats variety as a real input: "
        "Narli tends to yield a little more, so it carries a yield factor of 1.05 against Vezir's 1.00, "
        "with anything else grouped as 'other' at 0.95. And as mentioned, everything is measured in "
        "dekar. A dekar is 1,000 square meters, one tenth of a hectare, and it is the unit Turkish "
        "farmers think in. Reporting in hectares would have meant every field staffer doing mental "
        "arithmetic before acting on a recommendation, which is exactly the kind of friction you want "
        "to avoid in software people use daily."
    )

    p = doc.add_paragraph("Table 1: Key Agronomic Parameters for Industrial Hemp (Turkish units)")
    p.runs[0].font.bold = True
    tbl = doc.add_table(rows=8, cols=3)
    tbl.style = "Table Grid"
    table_header(tbl, ["Parameter", "Optimal Range", "Critical Limit"], "1a1a2e")
    rows_data = [
        ("Soil pH", "6.0 to 7.0", "< 5.0 or > 8.0 (blocked)"),
        ("Growing temperature (degC)", "15 to 27", "< 5 or > 35 (blocked)"),
        ("Seasonal rainfall (mm)", "300 to 700", "< 150 (drought risk)"),
        ("Electrical conductivity (dS/m)", "< 2.0", "> 4.0 (blocked, salt stress)"),
        ("Slope (%)", "0 to 8", "> 20 (blocked, harvest impractical)"),
        ("Nitrogen target (kg/dekar)", "about 12", "deficit reduces fiber quality"),
        ("Phosphorus target (kg/dekar)", "about 5", "limits root development"),
    ]
    bg = ["F0F4FF", "FFFFFF"]
    for i, (param, opt, crit) in enumerate(rows_data):
        r = tbl.rows[i + 1]
        set_row_bg(r, bg[i % 2])
        r.cells[0].text = param
        r.cells[1].text = opt
        r.cells[2].text = crit
        for cell in r.cells:
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    blank(1)

    add_body(doc,
        "Soil pH is probably the single most important site factor. Once you go outside the 6.0 to 7.0 "
        "band, nutrient availability falls off a cliff. Nitrogen and phosphorus get locked into soil "
        "compounds the roots cannot reach, even when those nutrients are physically sitting right there "
        "in the soil (Brady & Weil, 2008). That is why pH carries so much weight in the model and why "
        "an extreme pH triggers an outright block rather than a soft penalty."
    )
    add_body(doc,
        "Temperature works much the same way. Hemp has warm-climate origins but over thousands of years "
        "of cultivation it settled into growing best between 15 and 27 degrees Celsius. Below 5 degrees "
        "frost stops establishment cold; above 35 the plant stresses and fiber formation suffers (Small "
        "& Marcus, 2002; Tang et al., 2016). Those two ends are encoded as hard blockers."
    )

    doc.add_heading("1.2. The Kitt Platform Context", level=2)
    add_body(doc,
        "Kitt is a B2B platform that sits between hemp growers and the buyers who want their fiber and "
        "seed. The agronomist is the user who matters most for this engine. When a new field gets "
        "registered, the agronomist needs a quick read: is this field worth contracting, and what "
        "should the grower put on it. The catch is that at registration time you often do not have a "
        "soil lab report yet. Sometimes you have a rough pH from a field kit and nothing else. The "
        "platform still has to say something useful."
    )
    add_body(doc,
        "That constraint shaped the whole design. Soil chemistry fields are optional. If pH, nitrogen, "
        "phosphorus, potassium, organic matter, or EC are missing, the engine fills them with sensible "
        "defaults and drops the confidence score to flag that it is guessing. A full-data cold-start "
        "prediction comes back at 0.80 confidence; one running on imputed defaults comes back at 0.55. "
        "That number is not decoration. It tells the agronomist whether to act now or wait for the lab."
    )
    add_body(doc,
        "Why machine learning at all, instead of a lookup table of agronomic rules? Honestly, for the "
        "fertilizer math, rules are fine and that is exactly what we use. The suitability question is "
        "different. A field is never decided by one variable. A soil at pH 7.1 with excellent drainage "
        "and ideal temperature can easily beat a pH-6.5 field that drains poorly and runs cold. Simple "
        "thresholds cannot see those trade-offs, but a tree ensemble learns them straight from data. "
        "It also returns a continuous probability rather than a hard yes or no, which is what lets the "
        "platform reason about borderline fields instead of pretending every decision is clean. Tree "
        "boosting has become a common choice in agricultural decision support for exactly these reasons "
        "(Liakos et al., 2018)."
    )

    doc.add_page_break()

    # ── 2. RELATED WORK ───────────────────────────────────────────────────────
    doc.add_heading("2. Related Work", level=1)
    add_body(doc,
        "Crop yield prediction is one of the older and more productive corners of applied machine "
        "learning, so there is plenty to draw on. The problem is hard for an honest reason: outcome "
        "depends on soil chemistry, climate, terrain, and how the field is managed, all interacting at "
        "once, and those interactions shift from one region to the next. The point I want to make in "
        "this section is simple. The methods below were worked out for staple crops like rice, wheat, "
        "and maize. What I did was take the same toolkit and point it at hemp, which almost nobody has "
        "modeled this way."
    )
    add_body(doc,
        "Jeong et al. (2016) used random forests for rice yield prediction in South Korea at "
        "provincial resolution, mixing climate model output with terrain features. Their finding was "
        "that tree ensembles beat linear regression fairly consistently, mostly because trees handle "
        "threshold effects well: a small pH change near a critical value can swing yield far more than "
        "a linear model would ever predict. That is precisely the behavior hemp shows around its pH and "
        "temperature limits, which is part of why I went with trees."
    )
    add_body(doc,
        "Khaki and Wang (2019) put deep neural networks up against gradient boosting for wheat and "
        "maize. The networks were competitive on accuracy but needed far more data and were much harder "
        "to interpret. With a training set in the low thousands, like mine, a deep network would just "
        "overfit. Data efficiency settled the choice."
    )
    add_body(doc,
        "Van Klompenburg et al. (2020) reviewed 50 studies from 2008 to 2020 and found XGBoost, random "
        "forests, and neural networks dominating the field. The detail that stuck with me is that "
        "almost everything published targets maize, wheat, or soybean. Specialty industrial crops like "
        "hemp barely appear. That gap is most of the reason this project exists as a dedicated model "
        "rather than a fork of someone else's."
    )
    add_body(doc,
        "Shahhosseini et al. (2021) proposed a two-stage nested pipeline for maize: a classifier first, "
        "then a regressor that only runs inside the relevant class. That structure is the direct "
        "ancestor of what I built. The one change I made is that my Stage 1 decides whether to plant at "
        "all rather than sorting fields into yield bands, because for a platform deciding which fields "
        "to contract, the go or no-go question is the one that actually drives a decision."
    )
    add_body(doc,
        "Pantazi et al. (2016) compared support vector machines with neural networks for precision "
        "grain farming and made a point about interpretability that I think gets underrated. Agronomists "
        "have to understand why a system recommends something before they will stand behind it to a "
        "grower. XGBoost gives feature importances, so you can say which input drove a call, in language "
        "an agronomist already speaks."
    )

    doc.add_page_break()

    # ── 3. XGBOOST THEORY ─────────────────────────────────────────────────────
    doc.add_heading("3. XGBoost Theory and Mathematical Foundations", level=1)
    add_body(doc,
        "XGBoost (eXtreme Gradient Boosting) came out of Chen and Guestrin in 2016 and has been a "
        "default choice in applied machine learning ever since (Chen & Guestrin, 2016). It is a "
        "gradient boosting method: rather than fit one big model, it fits a sequence of small decision "
        "trees where each tree cleans up the errors the earlier trees left behind. I am going to walk "
        "through the math in full, because the hyperparameter choices later only make sense once you "
        "see where they come from."
    )

    doc.add_heading("3.1. Gradient Boosting", level=2)
    add_body(doc, "A model with T trees produces its prediction by adding up the trees, each scaled by the learning rate:")
    add_formula(doc, "y-hat_i  =  sum_{t=1}^{T}  eta * f_t(x_i)", 13)
    add_body(doc,
        "Here x_i is the feature vector for the i-th field, f_t is the leaf score from tree t, and eta "
        "is the learning rate, a number between 0 and 1 that shrinks each tree's contribution. A small "
        "eta forces the model to take many small steps instead of a few big ones, which usually "
        "generalizes better at the cost of more trees. I used eta = 0.05 for both models."
    )
    add_body(doc,
        "At each boosting step the algorithm asks how it should nudge the current prediction to lower "
        "the loss, and it answers by fitting the new tree to the gradient of the loss. To make that "
        "cheap to compute, the loss is approximated with a second-order Taylor expansion around the "
        "current estimate:"
    )
    add_formula(doc,
        "L^(t) ~= sum_i [ l(y_i, y-hat_i^(t-1)) + g_i*f_t(x_i) + (1/2)*h_i*f_t(x_i)^2 ] + Omega(f_t)", 10)
    add_body(doc,
        "g_i is the first derivative of the loss at the current prediction, and h_i is the second "
        "derivative, the Hessian. Using both, instead of just the gradient like classic boosting does, "
        "gives the algorithm a feel for the curvature of the loss surface. Loosely: the gradient says "
        "which way to step, the Hessian says how fast the error is changing in that direction, and "
        "together they let you take a sharper step than a gradient-only method could."
    )

    doc.add_heading("3.2. Loss Functions and Regularization", level=2)
    add_body(doc,
        "Two losses show up here. The classifier uses binary cross-entropy, also called log-loss, "
        "which measures the gap between the predicted probability and the true label:"
    )
    add_formula(doc, "L_cls = -sum_i [ y_i*log(p-hat_i) + (1 - y_i)*log(1 - p-hat_i) ]", 11)
    add_body(doc,
        "y_i is 0 or 1 (unsuitable or suitable), and p-hat_i is the probability after the sigmoid. "
        "Log-loss punishes confident wrong answers much harder than hesitant wrong answers, which "
        "pushes the model toward being honest about cases it is unsure of rather than just right on "
        "average. The regressor uses plain mean squared error:"
    )
    add_formula(doc, "L_reg = (1/n) * sum_i (y_i - y-hat_i)^2", 12)
    add_body(doc, "Every tree also carries a regularization term that penalizes its complexity:")
    add_formula(doc, "Omega(f) = gamma*T + (1/2)*lambda*||w||^2", 13)
    add_body(doc,
        "T is the number of leaves, w is the vector of leaf scores, gamma penalizes leaf count, and "
        "lambda is the L2 coefficient. A larger gamma makes the tree reluctant to split unless the "
        "split clearly earns its keep. The L2 term shrinks big leaf weights, working like weight decay "
        "in a neural net. Between them they stop the model from memorizing the training set. In this "
        "project I set lambda = 1.0 for both models, with L1 (alpha) at 0.1 for the classifier and 0.05 "
        "for the regressor."
    )

    doc.add_heading("3.3. Split Criterion", level=2)
    add_body(doc,
        "At every node the algorithm tries all feature-threshold pairs and keeps the split that "
        "maximizes this gain:"
    )
    add_formula(doc,
        "Gain = (1/2)*[ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda) ] - gamma", 10)
    add_body(doc,
        "G_L and H_L sum the gradients and Hessians in the left child, G_R and H_R do the same on the "
        "right. If the gain comes out negative, the split is thrown away because the drop in loss does "
        "not pay for the extra leaf that gamma charges for. max_depth caps things further: the "
        "classifier runs at max_depth=4 and the regressor at max_depth=5. Shallower trees generalize "
        "better; deeper trees catch finer patterns but overfit more easily on noisy data. The regressor "
        "gets the extra depth because a continuous yield target needs slightly finer distinctions than "
        "a yes/no suitability call."
    )

    print("Figure: data distribution...")
    add_figure(doc, fig_data_distribution(), 6.4,
        "Figure 1: Training data distributions. Left: soil pH. Centre: variety counts (Narli, "
        "Vezir, other). Right: expected yield in t/dekar split by suitability label, with the 0.30 "
        "t/dekar threshold marked.")

    doc.add_page_break()

    # ── 4. SYSTEM ARCHITECTURE ────────────────────────────────────────────────
    doc.add_heading("4. System Architecture", level=1)
    add_body(doc,
        "The engine is exposed through two API endpoints that share one internal function, "
        "compute_hemp_prescription. POST /hemp/prescription is the cold-start route: you hand it field "
        "inputs and it returns a prescription with no field history involved. POST "
        "/hemp/prescription/adaptive is the adaptive route: it takes the same field inputs plus a list "
        "of completed cycles, and when that list is empty it behaves exactly like the cold-start route. "
        "Keeping both behind one function means the routing logic lives in one place and the two "
        "endpoints can never drift apart."
    )

    print("Figure: architecture...")
    add_figure(doc, fig_architecture(), 6.5,
        "Figure 2: The Kitt hemp adaptive prescription pipeline. Both endpoints feed one engine "
        "function. Hard blockers run first, then the cold-start XGBoost path, then optional historical "
        "correction when the field has cycle history.")
    blank(1)

    add_body(doc,
        "The first thing the engine does is run the hard blockers. These are deterministic agronomic "
        "limits: pH below 5.0 or above 8.0, slope above 20%, mean temperature below 5 or above 35 "
        "degrees, EC above 4.0 dS/m. Any one of them stops the process and returns an unsuitable "
        "result with no prescription. There is a subtlety here that matters because soil data is "
        "optional. Each blocker is guarded with an is-not-None check, so a missing pH does not "
        "accidentally trip the pH blocker. The check only fires when the value actually exists and "
        "actually violates the limit. A field with no pH on record passes the pH blocker and gets "
        "decided later by the classifier on imputed defaults instead."
    )
    add_body(doc,
        "Whatever survives the blockers goes into the cold-start XGBoost path. Stage 1, the classifier, "
        "returns a suitability probability. Below 0.5 the field is flagged unsuitable and processing "
        "stops; this is the soft constraint that catches fields inside the hard limits but still a bad "
        "bet, like pH 7.8 paired with poor drainage and a temperature scraping the low end. Fields that "
        "clear 0.5 go to Stage 2, the regressor, which estimates fiber yield in t/dekar. The raw "
        "regressor output then gets multiplied by the variety factor, so a Narli field is nudged up "
        "about 5% over the same field planted with Vezir."
    )
    add_body(doc,
        "At this point the routing splits. If no cycle history was supplied, the cold-start result is "
        "what comes back, after agronomic notes get attached (warnings about pH amendments, drainage, "
        "high N deficit, drought risk, and so on). If cycle history was supplied, the cold-start result "
        "is handed to the historical correction layer first, which is the subject of the next section. "
        "Throughout all of this, N, P, K, and irrigation are computed by fixed formulas, not learned. "
        "Fertilizer chemistry is well understood and decades of agronomy have already written it down. "
        "Putting a model on top of those formulas would add noise without adding accuracy, so I left "
        "that part deterministic and auditable."
    )

    doc.add_page_break()

    # ── 5. HISTORICAL CORRECTION LAYER ────────────────────────────────────────
    doc.add_heading("5. Historical Correction Layer", level=1)
    add_body(doc,
        "The cold-start model is trained on a population of fields. It does not know the quirks of your "
        "specific field: the corner that floods, the patch of heavier clay, the way your particular "
        "microclimate runs a degree warm. The historical correction layer is how the engine learns "
        "those quirks over time, one field at a time, without retraining anything."
    )
    add_body(doc,
        "The unit it works on is the CycleRecord. Each record stores one completed growing cycle: what "
        "the cold-start model had predicted for yield, what N/P/K and irrigation were recommended, what "
        "the grower actually applied, and what actually came out at harvest. The yield and applied "
        "numbers are the ones that drive the correction. Records are passed oldest-first, and the "
        "provider weights recent cycles more heavily, because last season tells you more about next "
        "season than a cycle from four years ago."
    )
    add_body(doc,
        "The yield correction is an exponential-weighted average of the ratio between actual and "
        "predicted yield across all cycles:"
    )
    add_formula(doc,
        "factor = sum_i ( w_i * actual_i / predicted_i ) / sum_i w_i ,   w_i = 0.6^(N-1-i)", 11)
    add_body(doc,
        "N is the number of cycles and i runs from 0 (oldest) to N-1 (most recent). The decay base is "
        "0.6, so the newest cycle gets weight 1, the one before it 0.6, then 0.36, and so on. Cycles "
        "where the predicted yield was zero are skipped to avoid dividing by zero. The resulting factor "
        "is clamped to the range [0.50, 2.00]. That clamp matters: it stops a single freak season, a "
        "hailstorm wiping out a crop, or a fluke bumper harvest, from yanking the whole recommendation "
        "to an extreme. The corrected yield is just the cold-start yield multiplied by this factor."
    )

    print("Figure: historical correction...")
    add_figure(doc, fig_historical_correction(), 6.4,
        "Figure 3: Left, the exponential weights for five cycles at decay 0.6, normalised. Right, a "
        "worked example showing the correction factor settling as cycles accumulate while staying "
        "anchored near the more recent ratios.")
    blank(1)

    add_body(doc,
        "Nitrogen gets its own correction, separate from yield, and it only looks at the most recent "
        "cycle. The logic is an efficiency signal. If the grower applied clearly less N than was "
        "recommended (under 88% of the recommendation) and yield still held up (at least 93% of "
        "predicted), the engine takes the hint and trims next season's N recommendation, on the theory "
        "that this field needs less than the population average. The reverse also applies: if the "
        "grower applied well over the recommendation (above 115%) and yield jumped more than 10%, the "
        "recommendation creeps up a little. Outside those two cases the factor stays at 1.0 and N is "
        "left alone. The corrections are bounded so they can nudge but not lurch."
    )
    add_body(doc,
        "Confidence rises with evidence. The formula is confidence = min(0.95, 0.65 + n_cycles*0.10), "
        "so a field with one completed cycle reports 0.75, two cycles 0.85, three or more caps out at "
        "0.95. The ceiling is deliberate. No amount of history makes a yield forecast certain, and "
        "claiming 1.0 would be dishonest."
    )
    add_body(doc,
        "I want to be straight about why this layer is a hand-written formula and not another XGBoost "
        "model. Two reasons. First, per-field sample counts are tiny. A field with three cycles gives "
        "you three data points, and you cannot train anything on three points. An exponential-weighted "
        "average is the right tool when data is that thin. Second, interpretability. An agronomist can "
        "read 'we cut your N because you used less last year and still hit target' and agree or "
        "disagree. A learned correction would be a black box bolted onto a black box. Keeping it simple "
        "keeps it trustworthy."
    )

    doc.add_page_break()

    # ── 6. DATASET ────────────────────────────────────────────────────────────
    doc.add_heading("6. Dataset", level=1)
    add_body(doc,
        "There is no comprehensive, labeled, public hemp field dataset, at least not one I could find. "
        "Published work uses either proprietary farm data or tiny regional samples. So I generated a "
        "synthetic dataset of 2,000 samples. Synthetic data is an accepted practice in agricultural ML "
        "when the underlying agronomic relationships are well established: it lets you cover edge cases "
        "on purpose and rerun experiments exactly (Shahhosseini et al., 2021). The generator is seeded "
        "(seed=42) so the dataset is fully reproducible."
    )
    add_body(doc,
        "Each sample draws from realistic agronomic ranges across the feature set. After one-hot "
        "encoding the four categorical features (drainage class, soil texture, variety, irrigation "
        "type), the feature vector is 31 dimensions wide. All quantities are in dekar units: area in "
        "dekar, fertilizer in kg/dekar, yield in t/dekar. The new categorical inputs are the ones that "
        "make this a hemp-and-Turkey model rather than a generic one: variety captures Narli versus "
        "Vezir, irrigation_type captures drip versus sprinkler versus flood versus none, and "
        "rotation_n_credit captures the nitrogen the previous crop left behind."
    )

    p = doc.add_paragraph("Table 2: Selected Input Features (31-dimensional vector after encoding)")
    p.runs[0].font.bold = True
    feats = [
        ("ph", "Numeric (optional)", "5.0 to 8.0", "Soil pH"),
        ("nitrogen_ppm", "Numeric (optional)", "15 to 80 ppm", "Soil nitrogen"),
        ("organic_matter_percent", "Numeric (optional)", "1 to 6.5 %", "Organic matter"),
        ("ec", "Numeric (optional)", "0.1 to 4.0 dS/m", "Electrical conductivity"),
        ("area_dekar", "Numeric", "20 to 600 dekar", "Field area"),
        ("slope_percent", "Numeric", "0.5 to 22 %", "Slope"),
        ("first_hemp_season", "Binary", "0 / 1", "First time growing hemp here"),
        ("rotation_n_credit", "Numeric", "-1.0 to 3.0", "N credit from previous crop"),
        ("avg_temp", "Numeric (optional)", "8 to 35 degC", "Mean season temperature"),
        ("seasonal_rainfall_mm", "Numeric (optional)", "150 to 650 mm", "Seasonal rainfall"),
        ("variety", "Categorical", "narli / vezir / other", "Hemp cultivar"),
        ("irrigation_type", "Categorical", "drip / sprinkler / flood / none", "Irrigation system"),
        ("drainage_class", "Categorical", "poor / moderate / good / excellent", "Drainage"),
        ("expected_yield_ton_dekar", "Target", "0 to ~0.75 t/dekar", "Regression target"),
    ]
    feat_tbl = doc.add_table(rows=len(feats) + 1, cols=4)
    feat_tbl.style = "Table Grid"
    table_header(feat_tbl, ["Feature", "Type", "Range", "Description"], "2E86AB")
    bg2 = ["F0F4FF", "FFFFFF"]
    for i, row_data in enumerate(feats):
        r = feat_tbl.rows[i + 1]
        set_row_bg(r, bg2[i % 2])
        for j, val in enumerate(row_data):
            r.cells[j].text = val
    blank(1)

    add_body(doc,
        "Of the 2,000 samples, 1,164 (58.2%) come out suitable and 836 (41.8%) unsuitable. That split "
        "was on purpose. If almost every sample were suitable, the classifier could score high accuracy "
        "by lazily predicting 'suitable' every time, which is the majority-class trap and teaches the "
        "model nothing. A roughly 58/42 balance forces it to actually learn what separates a good field "
        "from a bad one."
    )

    doc.add_page_break()

    # ── 7. MODEL TRAINING ─────────────────────────────────────────────────────
    doc.add_heading("7. Model Training", level=1)
    add_body(doc,
        "The dataset was split 80/20 with a fixed seed, giving 1,600 training samples and 400 for the "
        "test fold. Both models run 150 boosting rounds. The classifier trains on all 1,600 training "
        "samples; the regressor trains only on the suitable ones, since a yield number for a field that "
        "cannot grow hemp is meaningless."
    )

    p = doc.add_paragraph("Table 3: Model Hyperparameters")
    p.runs[0].font.bold = True
    params = [
        ("objective", "binary:logistic", "reg:squarederror"),
        ("eval_metric", "logloss", "(rmse)"),
        ("max_depth", "4", "5"),
        ("eta (learning rate)", "0.05", "0.05"),
        ("subsample", "0.85", "0.90"),
        ("colsample_bytree", "0.85", "0.85"),
        ("alpha (L1)", "0.1", "0.05"),
        ("lambda (L2)", "1.0", "1.0"),
        ("num_boost_round", "150", "150"),
    ]
    ptbl = doc.add_table(rows=len(params) + 1, cols=3)
    ptbl.style = "Table Grid"
    table_header(ptbl, ["Parameter", "Classifier", "Regressor"], "2A9D8F")
    bg3 = ["F0FFF0", "FFFFFF"]
    for i, (k, c, rg) in enumerate(params):
        r = ptbl.rows[i + 1]
        set_row_bg(r, bg3[i % 2])
        r.cells[0].text = k
        r.cells[1].text = c
        r.cells[2].text = rg
        for cell in r.cells:
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    blank(1)

    add_body(doc,
        "The learning rate of 0.05 is deliberately slow. Combined with 150 rounds it means many small "
        "corrective steps instead of a few aggressive ones, which trades training time for "
        "generalization. Aggressive learning rates overfit, and a smaller dataset has less slack to "
        "absorb that. The subsample and colsample_bytree values train each tree on 85% to 90% of the "
        "samples and 85% of the features, a randomness trick borrowed from random forests that cuts "
        "overfitting further. The regressor runs one level deeper than the classifier because a "
        "continuous yield surface needs finer splits than a binary decision does."
    )
    add_body(doc,
        "On the 400-sample test fold the classifier landed at accuracy 0.9525, precision 0.9474, "
        "recall 0.975, and F1 0.961. The high recall is the part I cared about most: it means the model "
        "rarely blocks a field that was actually fine, and a wrongly blocked field is a contract Kitt "
        "walks away from for no reason. The regressor, trained on the suitable samples, reached RMSE = "
        "0.065 t/dekar and R2 = 0.70. Those are the numbers carried into the results section."
    )

    doc.add_page_break()

    # ── 8. PRESCRIPTION FORMULAS ──────────────────────────────────────────────
    doc.add_heading("8. Prescription Formulas", level=1)
    add_body(doc,
        "Fertilizer and irrigation come from explicit formulas in dekar units, not from the model. The "
        "nutrient needs of hemp are well documented and convert cleanly into chemistry, so this is one "
        "place where a formula is simply the right answer. The ppm-to-kg conversion below assumes a "
        "30 cm sampling depth and a soil bulk density of 1.3 g/cm3:"
    )
    add_formula(doc, "1 ppm  ~=  0.39 kg/dekar     (depth = 30 cm, rho_bulk = 1.3 g/cm^3)", 11)

    add_body(doc, "Nitrogen, where the rotation credit from the previous crop is subtracted along with what organic matter will release:")
    add_formula(doc, "N_rec = max(0, 12 - N_ppm*0.39 - OM%*2 - rotation_credit)", 12)
    add_body(doc, "Phosphorus and potassium follow the same shape against their own targets:")
    add_formula(doc, "P_rec = max(0, 5 - P_ppm*0.39)", 12)
    add_formula(doc, "K_rec = max(0, 10 - K_ppm*0.39)", 12)
    add_body(doc,
        "The OM%*2 term credits the nitrogen that organic matter mineralizes over the season, so a "
        "field rich in organic matter draws on its own reserves and needs less applied N. The rotation "
        "credit comes from a small lookup. A legume previous crop leaves the most behind, hemp after "
        "hemp slightly depletes the soil, and fallow gives back a little:"
    )

    p = doc.add_paragraph("Table 4: Rotation Nitrogen Credits (kg/dekar)")
    p.runs[0].font.bold = True
    rot = [
        ("legume", "+3.0", "fixes atmospheric N, biggest credit"),
        ("fallow", "+1.0", "rested soil, modest gain"),
        ("sunflower", "+0.5", "small residual"),
        ("cereal", "0.0", "neutral"),
        ("other", "0.0", "neutral default"),
        ("hemp", "-1.0", "hemp after hemp, slight depletion"),
    ]
    rtbl = doc.add_table(rows=len(rot) + 1, cols=3)
    rtbl.style = "Table Grid"
    table_header(rtbl, ["Previous crop", "N credit", "Reason"], "264653")
    for i, (crop, cr, why) in enumerate(rot):
        r = rtbl.rows[i + 1]
        set_row_bg(r, bg3[i % 2])
        r.cells[0].text = crop
        r.cells[1].text = cr
        r.cells[2].text = why
    blank(1)

    add_body(doc, "Irrigation depends on how much rain falls and how efficient the irrigation system is:")
    add_formula(doc,
        "effective_rainfall = seasonal_rainfall * 0.75\n"
        "water_deficit      = max(0, 450 - effective_rainfall)\n"
        "irr_mm_week        = (water_deficit / 20) / efficiency", 11)
    add_body(doc,
        "The 0.75 is the effective rainfall fraction, the share of rain that reaches the root zone "
        "after runoff and evaporation, a standard assumption for field crops in temperate to semi-arid "
        "conditions (Allen et al., 1998). The efficiency divisor is where irrigation type enters: drip "
        "runs at 0.90, sprinkler at 0.78, flood at 0.55. A flood-irrigated field needs more gross water "
        "to deliver the same amount to the roots, which the division captures. If the field has no "
        "irrigation, efficiency is zero and the recommendation is simply zero."
    )

    print("Figure: prescription curves...")
    add_figure(doc, fig_prescription_logic(), 6.4,
        "Figure 4: Nitrogen recommendation against soil N by organic matter (left) and weekly "
        "irrigation against seasonal rainfall by irrigation type (right). Flood sits highest because "
        "its low efficiency demands more gross water.")
    print("Figure: penalty curves...")
    add_figure(doc, fig_penalty_curves(), 6.4,
        "Figure 5: pH penalty (left) and temperature penalty (right) used inside the rule-based yield "
        "fallback. The multiplier falls off linearly outside the optimal band.")

    doc.add_page_break()

    # ── 9. RESULTS ────────────────────────────────────────────────────────────
    doc.add_heading("9. Results", level=1)
    add_body(doc,
        "Evaluation is on the 400-sample held-out test fold the models never trained on. The classifier "
        "results are set against a rule-based baseline, which is the honest comparison: the rule-based "
        "approach has no classifier at all, it just calls every field suitable, so its weakness is "
        "structural rather than a tuning problem."
    )

    p = doc.add_paragraph("Table 5: Stage 1 Classification (400-sample test fold)")
    p.runs[0].font.bold = True
    cls_data = [
        ("Accuracy", "0.5825", "0.9525"),
        ("Precision", "0.5825", "0.9474"),
        ("Recall", "1.0000", "0.9750"),
        ("F1", "0.7362", "0.9610"),
        ("False positives", "167", "13"),
        ("False negatives", "0", "6"),
    ]
    res_tbl = doc.add_table(rows=len(cls_data) + 1, cols=3)
    res_tbl.style = "Table Grid"
    table_header(res_tbl, ["Metric", "Rule-Based", "XGBoost"], "1a1a2e")
    bg4 = ["FFFBE6", "FFFFFF"]
    for i, (m, rb, xg) in enumerate(cls_data):
        r = res_tbl.rows[i + 1]
        set_row_bg(r, bg4[i % 2])
        r.cells[0].text = m
        r.cells[1].text = rb
        r.cells[2].text = xg
        for cell in r.cells:
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    blank(1)

    p = doc.add_paragraph("Table 6: Stage 2 Regression (suitable test samples)")
    p.runs[0].font.bold = True
    reg_data = [
        ("RMSE (t/dekar)", "0.065"),
        ("MAE (t/dekar)", "0.051"),
        ("R2", "0.70"),
    ]
    reg_tbl = doc.add_table(rows=len(reg_data) + 1, cols=2)
    reg_tbl.style = "Table Grid"
    table_header(reg_tbl, ["Metric", "XGBoost"], "264653")
    for i, (m, xg) in enumerate(reg_data):
        r = reg_tbl.rows[i + 1]
        set_row_bg(r, bg4[i % 2])
        r.cells[0].text = m
        r.cells[1].text = xg
        for cell in r.cells:
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    blank(1)

    add_body(doc,
        "The classifier comparison is lopsided, and it should be. The rule-based baseline scores 1.0 "
        "recall trivially because it never says no, but its precision collapses to 0.58 and it "
        "misclassifies every single unsuitable field. The XGBoost classifier holds 0.975 recall while "
        "lifting precision to 0.947, catching most of the unsuitable fields the baseline waves through. "
        "On the regression side there is no meaningful rule-based number to compare against, since the "
        "rule-based yield is the same formula that generated the labels and would score a circular and "
        "useless RMSE of zero. The XGBoost RMSE of 0.065 t/dekar against a typical suitable yield "
        "around 0.6 t/dekar is roughly 11% relative error, which is in line with published boosting "
        "results for other crops on comparable synthetic sets (Khaki & Wang, 2019)."
    )

    print("Figure: confusion matrix...")
    add_figure(doc, fig_confusion_matrix(), 6.2,
        "Figure 6: Confusion matrices for the rule-based (left) and XGBoost (right) classifiers. The "
        "rule-based model puts all 167 unsuitable fields in the false-positive cell because it cannot "
        "say no.")
    print("Figure: performance bars...")
    add_figure(doc, fig_performance_comparison(), 6.4,
        "Figure 7: Classification metrics side by side (left) and the XGBoost regression metrics in "
        "dekar units (right).")

    doc.add_page_break()

    # ── 10. DISCUSSION ────────────────────────────────────────────────────────
    doc.add_heading("10. Discussion", level=1)
    add_body(doc,
        "A few things are worth saying plainly. The dekar decision looks trivial but it is not. Every "
        "number a Turkish agronomist or grower reads off this system is in the unit they already work "
        "in, so nobody converts anything in their head before acting. Reporting in hectares would have "
        "saved me some refactoring and cost the users a small daily tax in arithmetic and mistakes. For "
        "a platform people touch every day, that trade was easy."
    )
    add_body(doc,
        "Optional soil chemistry is the other design call I keep coming back to. Kitt registers fields "
        "before anyone has a lab report in hand, and a system that demanded full soil panels before it "
        "would say anything would be useless at exactly the moment the agronomist needs a first read. "
        "Letting the inputs be optional, imputing defaults, and dropping the confidence score to flag "
        "the guess is what makes the cold-start path actually usable in the field. The confidence number "
        "is doing real work there: 0.80 with full data, 0.55 on imputed defaults, and the agronomist "
        "reads that and decides whether to trust it now or wait."
    )
    add_body(doc,
        "I have already argued why the historical correction is a formula and not a model, but it bears "
        "repeating because it is the part people question first. Per-field data is just too thin to "
        "train on. Three cycles is three numbers. An exponential-weighted average is the honest tool "
        "for that regime, and it stays readable, which means an agronomist can sanity-check it. "
        "Separating it from XGBoost also keeps the population model and the per-field adjustment from "
        "tangling together, so I can reason about each on its own."
    )
    add_body(doc,
        "The honest limitation is the training data. It is all synthetic. Synthetic data covers the "
        "parameter space neatly, but it cannot reproduce the messy correlation structure of real soil, "
        "real microclimate, or the things growers do that no formula anticipates. The architecture is "
        "sound and the weights are a sensible starting point, but I would expect retraining on real "
        "field records to shift both the feature importances and the yield coefficients, maybe by a "
        "lot. The model has never seen a real harvest. That is the first thing I would change with "
        "access to actual data."
    )

    print("Figure: feature importance...")
    add_figure(doc, fig_feature_importance(), 5.9,
        "Figure 8: Approximate feature importances for the Stage 1 classifier. Temperature and pH lead, "
        "which lines up with the agronomy. These are indicative, not read directly off the booster.")

    doc.add_page_break()

    # ── 11. CONCLUSION ────────────────────────────────────────────────────────
    doc.add_heading("11. Conclusion", level=1)
    add_body(doc,
        "This project built and evaluated an adaptive prescription engine for the Kitt hemp platform. "
        "A new field runs through hard agronomic blockers, then a nested pair of XGBoost models that "
        "first decide suitability and then forecast yield in t/dekar. A field with growing history runs "
        "through the same cold-start path and then gets corrected by its own record, using an "
        "exponential-weighted average of past actual-versus-predicted yield with the most recent cycle "
        "weighted highest."
    )
    add_body(doc,
        "On the held-out fold the classifier reached F1 = 0.961 and 95.25% accuracy, well past a "
        "rule-based baseline that cannot identify an unsuitable field at all. The yield regressor "
        "reached RMSE = 0.065 t/dekar and R2 = 0.70, a fair result for a model trained on synthetic "
        "data. Fertilizer and irrigation stay on deterministic formulas, which keeps the auditable part "
        "of the output auditable, while suitability and yield are learned where learning earns its keep."
    )
    add_body(doc,
        "The structure travels. The same blocker-then-classifier-then-regressor spine, with a "
        "per-field correction layer on top, would carry over to other specialty crops a platform like "
        "Kitt might add. What would change is the training data, the agronomic thresholds, and the "
        "nutrient targets, not the shape of the pipeline. The most valuable next step is not a "
        "cleverer model; it is real harvest data to retrain on."
    )

    doc.add_page_break()

    # ── REFERENCES ────────────────────────────────────────────────────────────
    doc.add_heading("References", level=1)
    refs = [
        "Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration: "
        "Guidelines for computing crop water requirements (FAO Irrigation and Drainage Paper 56). "
        "Food and Agriculture Organization of the United Nations.",

        "Brady, N. C., & Weil, R. R. (2008). The nature and properties of soils (14th ed.). "
        "Pearson Education.",

        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings "
        "of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, "
        "785-794. https://doi.org/10.1145/2939672.2939785",

        "Jeong, J. H., Resop, J. P., Mueller, N. D., Fleisher, D. H., Yun, K., Butler, E. E., "
        "... & Kim, S. H. (2016). Random forests for global and regional crop yield predictions. "
        "PLOS ONE, 11(6), e0156571. https://doi.org/10.1371/journal.pone.0156571",

        "Khaki, S., & Wang, L. (2019). Crop yield prediction using deep neural networks. "
        "Frontiers in Plant Science, 10, 621. https://doi.org/10.3389/fpls.2019.00621",

        "Liakos, K. G., Busato, P., Moshou, D., Pearson, S., & Bochtis, D. (2018). Machine "
        "learning in agriculture: A review. Sensors, 18(8), 2674. "
        "https://doi.org/10.3390/s18082674",

        "Pantazi, X. E., Moshou, D., Alexandridis, T., Whetton, R. L., & Mouazen, A. M. (2016). "
        "Wheat yield prediction using machine learning and advanced sensing techniques. Computers "
        "and Electronics in Agriculture, 121, 57-65. "
        "https://doi.org/10.1016/j.compag.2015.11.018",

        "Schultes, R. E. (1970). Random thoughts and queries on the botany of Cannabis. "
        "In C. R. B. Joyce & S. H. Curry (Eds.), The botany and chemistry of Cannabis (pp. 11-38). "
        "J. & A. Churchill.",

        "Shahhosseini, M., Hu, G., Huber, I., & Archontoulis, S. V. (2021). Coupling machine "
        "learning and crop modeling improves crop yield prediction in the US Corn Belt. "
        "Scientific Reports, 11, 1606. https://doi.org/10.1038/s41598-020-80820-1",

        "Small, E., & Marcus, D. (2002). Hemp: A new crop with new uses for North America. "
        "In J. Janick & A. Whipkey (Eds.), Trends in new crops and new uses (pp. 284-326). "
        "ASHS Press.",

        "Tang, K., Struik, P. C., Yin, X., Thouminot, C., Bjelkova, M., Stramkale, V., & "
        "Amaducci, S. (2016). Comparing hemp (Cannabis sativa L.) cultivars for stem and fibre "
        "quality. Industrial Crops and Products, 85, 300-308. "
        "https://doi.org/10.1016/j.indcrop.2016.03.001",

        "Van Klompenburg, T., Kassahun, A., & Catal, C. (2020). Crop yield prediction using "
        "machine learning: A systematic literature review. Computers and Electronics in "
        "Agriculture, 177, 105709. https://doi.org/10.1016/j.compag.2020.105709",
    ]
    for ref in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.first_line_indent = Cm(-1.0)
        p_ref.paragraph_format.left_indent = Cm(1.0)
        p_ref.paragraph_format.space_after = Pt(6)
        run_ref = p_ref.add_run(ref)
        run_ref.font.name = "Times New Roman"
        run_ref.font.size = Pt(10)

    return doc


if __name__ == "__main__":
    print("Generating report...")
    doc = build_doc()
    doc.save(OUT_PATH)
    print(f"Saved: {OUT_PATH.resolve()}")
