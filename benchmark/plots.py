"""Charts for the benchmark (matplotlib, saved as PNG files)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker                      # write files; no window needed
import matplotlib.pyplot as plt

COLORS = {"baseline": "#8c8c8c", "agent_full": "#1f77b4"}
LABELS = {"baseline": "Plain LLM", "agent_full": "Full agent"}
CAT_LABELS = {
    "calculator": "Calculator", "datetime": "Date/time", "time_difference": "Time diff.",
    "weather": "Weather", "web_search": "Web search", "multi_tool": "Multi-tool",
    "knowledge": "Knowledge",
}


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_overall(summary, cmp, conds, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for i, c in enumerate(conds):
        s = summary[(c, "overall")]
        ax.bar(i, s["accuracy"] * 100, color=COLORS[c], width=0.6,
               yerr=[[ (s["accuracy"] - s["ci_low"]) * 100], [(s["ci_high"] - s["accuracy"]) * 100]],
               capsize=6)
        ax.text(i, s["accuracy"] * 100 + 6, f"{s['accuracy']*100:.1f}%\n(n={s['decided']})",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(conds)), [LABELS[c] for c in conds])
    ax.set_ylim(0, 125)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Overall accuracy (95% Wilson CI)\nMcNemar exact p = {cmp['overall']['p_exact']:.4f} "
                 f"({_stars(cmp['overall']['p_exact'])})", fontsize=10)
    return _save(fig, out_dir, "1_overall_accuracy.png")


def plot_by_category(summary, cmp, conds, cats, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.38
    for j, c in enumerate(conds):
        xs = [i + (j - 0.5) * width for i in range(len(cats))]
        vals = [summary[(c, cat)]["accuracy"] * 100 for cat in cats]
        lo = [(summary[(c, cat)]["accuracy"] - summary[(c, cat)]["ci_low"]) * 100 for cat in cats]
        hi = [(summary[(c, cat)]["ci_high"] - summary[(c, cat)]["accuracy"]) * 100 for cat in cats]
        ax.bar(xs, vals, width, label=LABELS[c], color=COLORS[c], yerr=[lo, hi], capsize=3)
    for i, cat in enumerate(cats):                       # significance marker above each pair
        p = cmp[cat]["p_holm"]
        ax.text(i, 108, _stars(p), ha="center", fontsize=10)
    ax.set_xticks(range(len(cats)), [CAT_LABELS.get(c, c) for c in cats])
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy per tool category (markers: McNemar, Holm-adjusted; ns = not significant)",
                 fontsize=10)
    ax.legend(loc="lower right")
    return _save(fig, out_dir, "2_accuracy_by_category.png")


def plot_cost(summary, conds, cats, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    width = 0.38
    for j, c in enumerate(conds):
        xs = [i + (j - 0.5) * width for i in range(len(cats))]
        ax1.bar(xs, [summary[(c, cat)]["mean_latency_s"] for cat in cats], width,
                label=LABELS[c], color=COLORS[c])
        ax2.bar(xs, [summary[(c, cat)]["mean_llm_calls"] for cat in cats], width,
                label=LABELS[c], color=COLORS[c])
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(cats)), [CAT_LABELS.get(c, c) for c in cats], rotation=25, ha="right")
        ax.legend()
    ax1.set_yscale("log")
    ax1.set_ylabel("Mean latency per question (s, log scale)")
    ax1.set_title("Latency")
    ax2.set_ylabel("Mean model calls per question")
    ax2.set_title("Model calls")
    return _save(fig, out_dir, "3_cost_latency_calls.png")


def plot_discordant(cmp, cats, conds, out_dir):
    a, b = conds
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ys = range(len(cats))
    only_a = [-cmp[c]["only_a"] for c in cats]
    only_b = [cmp[c]["only_b"] for c in cats]
    ax.barh(ys, only_a, color=COLORS[a], label=f"Only {LABELS[a]} correct")
    ax.barh(ys, only_b, color=COLORS[b], label=f"Only {LABELS[b]} correct")
    lim = max([abs(v) for v in only_a + only_b] + [1]) + 1.5
    for y, c in zip(ys, cats):                         # p-value at the right edge, off the bars
        ax.text(lim - 0.1, y, f"p={cmp[c]['p_exact']:.3f}", ha="right", va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(list(ys), [CAT_LABELS.get(c, c) for c in cats])
    ax.invert_yaxis()
    ax.set_xlabel("Number of questions where only one method was right")
    ax.set_title("Discordant pairs per category (the data McNemar's test uses)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(int(v))}"))
    return _save(fig, out_dir, "4_discordant_pairs.png")


def make_all(summary, cmp, conds, cats, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [plot_overall(summary, cmp, conds, out_dir),
             plot_by_category(summary, cmp, conds, cats, out_dir),
             plot_cost(summary, conds, cats, out_dir)]
    if len(conds) == 2:
        paths.append(plot_discordant(cmp, cats, conds, out_dir))
    return paths