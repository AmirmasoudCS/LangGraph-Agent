"""Score the raw results, print the tables, run McNemar's test, save CSVs and charts.

Usage (from the project root):
    python -m benchmark.analyze
    python -m benchmark.analyze --no-plots
"""
import argparse
import csv
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from benchmark import metrics, plots

HERE = Path(__file__).parent
DATASET = HERE / "dataset.json"
RESULTS = HERE / "results"
RAW = RESULTS / "raw.jsonl"
REVIEWS = RESULTS / "reviews.json"

CONDS = ["baseline", "agent_full"]
CATEGORIES = ["calculator", "datetime", "time_difference", "weather",
              "web_search", "multi_tool", "knowledge"]

console = Console()


def load():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    raw = {}
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            raw[(row["id"], row["condition"])] = row         # a later run replaces an earlier one
    reviews = json.loads(REVIEWS.read_text(encoding="utf-8")) if REVIEWS.exists() else {}
    return dataset, [r for r in raw.values() if r["condition"] in CONDS], reviews


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(x):
    return f"{x * 100:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if not RAW.exists():
        console.print("[red]No results yet. Run: python -m benchmark.run_benchmark[/red]")
        return

    dataset, raw, reviews = load()
    scored = metrics.score_rows(dataset, raw, reviews)
    cats = [c for c in CATEGORIES if any(r["category"] == c for r in scored)]

    # ---- summaries: overall and per category, per condition ----
    summary = {}
    for cond in CONDS:
        summary[(cond, "overall")] = metrics.summarize([r for r in scored if r["condition"] == cond])
        for cat in cats:
            summary[(cond, cat)] = metrics.summarize(
                [r for r in scored if r["condition"] == cond and r["category"] == cat])

    # ---- McNemar: overall and per category (Holm-corrected across categories) ----
    cmp = {"overall": metrics.paired(scored, *CONDS)}
    for cat in cats:
        cmp[cat] = metrics.paired([r for r in scored if r["category"] == cat], *CONDS)
    adjusted = metrics.holm([cmp[c]["p_exact"] for c in cats])
    for cat, p in zip(cats, adjusted):
        cmp[cat]["p_holm"] = p
    cmp["overall"]["p_holm"] = cmp["overall"]["p_exact"]

    # ---- print: overall ----
    t = Table(title="Overall", title_justify="left", border_style="cyan")
    for col in ["Method", "Accuracy", "95% CI", "Right/Decided", "Undecided",
                "Mean time", "Model calls", "Tool calls", "Errors"]:
        t.add_column(col)
    for cond in CONDS:
        s = summary[(cond, "overall")]
        t.add_row(plots.LABELS[cond], pct(s["accuracy"]), f"{pct(s['ci_low'])} to {pct(s['ci_high'])}",
                  f"{s['correct']}/{s['decided']}", str(s["undecided"]), f"{s['mean_latency_s']:.1f}s",
                  f"{s['mean_llm_calls']:.1f}", f"{s['mean_tool_calls']:.1f}", pct(s["error_rate"]))
    console.print(t)

    # ---- print: per category ----
    t = Table(title="Per tool category", title_justify="left", border_style="cyan")
    for col in ["Category", "Plain LLM", "Full agent", "Only LLM right", "Only agent right",
                "p (exact)", "p (Holm)", "Agent time", "Agent used tool"]:
        t.add_column(col)
    for cat in cats:
        a, b = summary[("baseline", cat)], summary[("agent_full", cat)]
        c = cmp[cat]
        tool_use = "-" if b["tool_use_rate"] is None else pct(b["tool_use_rate"])
        t.add_row(cat, f"{pct(a['accuracy'])} ({a['correct']}/{a['decided']})",
                  f"{pct(b['accuracy'])} ({b['correct']}/{b['decided']})",
                  str(c["only_a"]), str(c["only_b"]), f"{c['p_exact']:.4f}", f"{c['p_holm']:.4f}",
                  f"{b['mean_latency_s']:.1f}s", tool_use)
    console.print(t)

    o = cmp["overall"]
    console.print(
        f"\n[bold]McNemar overall[/bold] on {o['pairs']} paired questions: "
        f"both right {o['both_right']}, only plain LLM right {o['only_a']}, "
        f"only agent right {o['only_b']}, both wrong {o['both_wrong']}.\n"
        f"Exact p = {o['p_exact']:.4f}  |  chi-square (continuity corrected) = {o['chi2']:.2f}, "
        f"p = {o['p_chi2']:.4f}"
    )
    undecided = sum(1 for r in scored if r["correct"] is None)
    if undecided:
        console.print(f"[yellow]{undecided} runs are still undecided and excluded from the tests. "
                      f"Label them with: python -m benchmark.review[/yellow]")

    # ---- save CSVs ----
    write_csv(RESULTS / "scored.csv", [{**r, "tools_used": ",".join(r["tools_used"]),
                                        "expected_tools": ",".join(r["expected_tools"])} for r in scored])
    rows = [{"condition": c, "category": cat, **s} for (c, cat), s in summary.items()]
    write_csv(RESULTS / "summary.csv", rows)
    write_csv(RESULTS / "mcnemar.csv", [{"category": k, **v} for k, v in cmp.items()])
    console.print(f"\n[dim]Saved scored.csv, summary.csv and mcnemar.csv in {RESULTS}[/dim]")

    if not args.no_plots:
        paths = plots.make_all(summary, cmp, CONDS, cats, RESULTS / "figures")
        for p in paths:
            console.print(f"[dim]chart:[/dim] {p}")


if __name__ == "__main__":
    main()