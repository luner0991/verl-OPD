import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── 加载数据 ──────────────────────────────────────────────────────────────────
with open("/home/liuxinyuan/anchorKD/openreasoning_gp_nogp_ppl.json", encoding="utf-8") as f:
    data = json.load(f)

ppls = np.array([item["teacher_output_ppl"] for item in data if item.get("teacher_output_ppl") is not None])
n = len(ppls)

# ── 统计指标 ──────────────────────────────────────────────────────────────────
mean_ppl   = ppls.mean()
median_ppl = np.median(ppls)
std_ppl    = ppls.std()
p5, p25, p75, p95 = np.percentile(ppls, [5, 25, 75, 95])
skewness   = stats.skew(ppls)
kurt       = stats.kurtosis(ppls)

# 分段统计
bins_custom = [0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 100]
labels_custom = ["<1.5","1.5-2","2-2.5","2.5-3","3-3.5","3.5-4","4-5","5-6",">6"]
counts = []
for lo, hi in zip(bins_custom[:-1], bins_custom[1:]):
    cnt = ((ppls >= lo) & (ppls < hi)).sum()
    counts.append(cnt)

# ── 绘图 ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
fig = plt.figure(figsize=(16, 12))
fig.suptitle("PPL Distribution Analysis — Qwen3-0.6B on teacher_output\n"
             f"(N={n:,}  |  Mean={mean_ppl:.3f}  |  Median={median_ppl:.3f}  |  Std={std_ppl:.3f})",
             fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

# ── 1. 主直方图 + KDE ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
sns.histplot(ppls, bins=80, kde=True, color="#4C72B0", alpha=0.7,
             line_kws={"linewidth": 2}, ax=ax1)
ax1.axvline(mean_ppl,   color="#DD4949", linewidth=1.8, linestyle="--", label=f"Mean={mean_ppl:.3f}")
ax1.axvline(median_ppl, color="#E88A1A", linewidth=1.8, linestyle="-.",  label=f"Median={median_ppl:.3f}")
ax1.axvspan(p25, p75, alpha=0.12, color="green", label=f"IQR [{p25:.2f}, {p75:.2f}]")
ax1.set_xlabel("PPL"); ax1.set_ylabel("Count")
ax1.set_title("Histogram + KDE")
ax1.legend(fontsize=10)

# ── 2. 累积分布 CDF ───────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
sorted_ppl = np.sort(ppls)
cdf = np.arange(1, n + 1) / n
ax2.plot(sorted_ppl, cdf, color="#4C72B0", linewidth=2)
for pct, val, col in [(0.25, p25, "#55A868"), (0.50, median_ppl, "#E88A1A"),
                       (0.75, p75, "#55A868"), (0.95, p95, "#DD4949")]:
    ax2.axhline(pct, color=col, linewidth=0.8, linestyle=":")
    ax2.axvline(val, color=col, linewidth=0.8, linestyle=":")
    ax2.annotate(f"P{int(pct*100)}={val:.2f}", xy=(val, pct),
                 xytext=(val + 0.05, pct - 0.06), fontsize=8, color=col)
ax2.set_xlabel("PPL"); ax2.set_ylabel("Cumulative Probability")
ax2.set_title("CDF"); ax2.grid(alpha=0.3)

# ── 3. Box plot ───────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
bp = ax3.boxplot(ppls, vert=True, patch_artist=True, widths=0.5,
                 boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                 medianprops=dict(color="#E88A1A", linewidth=2),
                 flierprops=dict(marker="o", markersize=2, alpha=0.3, color="#DD4949"))
ax3.set_ylabel("PPL"); ax3.set_title("Box Plot")
ax3.set_xticks([]); ax3.grid(axis="y", alpha=0.3)

# ── 4. 分段条形图 ─────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
bar_colors = ["#2ecc71" if c / n > 0.10 else "#3498db" if c / n > 0.03 else "#e74c3c"
              for c in counts]
bars = ax4.bar(labels_custom, counts, color=bar_colors, edgecolor="white", linewidth=0.8)
for bar, cnt in zip(bars, counts):
    pct = cnt / n * 100
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
             f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
ax4.set_xlabel("PPL Range"); ax4.set_ylabel("Count")
ax4.set_title("Segment Distribution")
ax4.tick_params(axis="x", rotation=30)
ax4.grid(axis="y", alpha=0.3)

# ── 5. 统计摘要文本 ───────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
summary = (
    f"{'Statistic':<18} {'Value':>8}\n"
    f"{'─'*28}\n"
    f"{'Samples':<18} {n:>8,}\n"
    f"{'Min':<18} {ppls.min():>8.4f}\n"
    f"{'Max':<18} {ppls.max():>8.4f}\n"
    f"{'Mean':<18} {mean_ppl:>8.4f}\n"
    f"{'Median':<18} {median_ppl:>8.4f}\n"
    f"{'Std':<18} {std_ppl:>8.4f}\n"
    f"{'P5':<18} {p5:>8.4f}\n"
    f"{'P25 (Q1)':<18} {p25:>8.4f}\n"
    f"{'P75 (Q3)':<18} {p75:>8.4f}\n"
    f"{'P95':<18} {p95:>8.4f}\n"
    f"{'IQR':<18} {p75-p25:>8.4f}\n"
    f"{'Skewness':<18} {skewness:>8.4f}\n"
    f"{'Kurtosis':<18} {kurt:>8.4f}\n"
    f"{'─'*28}\n"
    f"{'PPL < 2.0':<18} {(ppls < 2.0).sum():>8,} ({(ppls < 2.0).mean()*100:.1f}%)\n"
    f"{'PPL < 3.0':<18} {(ppls < 3.0).sum():>8,} ({(ppls < 3.0).mean()*100:.1f}%)\n"
    f"{'PPL > 5.0':<18} {(ppls > 5.0).sum():>8,} ({(ppls > 5.0).mean()*100:.1f}%)\n"
)
ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
         fontsize=10, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="#f0f4ff", alpha=0.8))
ax5.set_title("Summary Statistics")

out = "/home/liuxinyuan/anchorKD/ppl_nogp_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# 打印分析结论
print("\n===== 分布分析 =====")
print(f"均值 {mean_ppl:.3f} > 中位数 {median_ppl:.3f}，分布右偏（Skewness={skewness:.3f}）")
print(f"50% 的样本 PPL 集中在 [{p25:.2f}, {p75:.2f}]（IQR={p75-p25:.3f}）")
print(f"PPL < 3.0 的样本占 {(ppls < 3.0).mean()*100:.1f}%（模型对这些文本较熟悉）")
print(f"PPL > 5.0 的样本占 {(ppls > 5.0).mean()*100:.1f}%（高困惑度，模型不熟悉）")
