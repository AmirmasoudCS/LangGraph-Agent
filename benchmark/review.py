"""Label the answers the automatic checks could not decide.

Usage (from the project root):
    python -m benchmark.review

For each undecided answer you see the question, the answer, the hint from the
automatic check and the live search results the benchmark fetched. Press
y (correct), n (wrong), s (skip) or q (save and quit). Labels are saved to
benchmark/results/reviews.json, and `python -m benchmark.analyze` uses them.
"""
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from benchmark import metrics

HERE = Path(__file__).parent
DATASET = HERE / "dataset.json"
RAW = HERE / "results" / "raw.jsonl"
REVIEWS = HERE / "results" / "reviews.json"

console = Console()


def find_snippets(ref):
    """Return the search snippets from a reference, including inside 'all' parts."""
    if not ref:
        return ""
    if ref.get("snippets"):
        return ref["snippets"]
    return "\n".join(find_snippets(p) for p in ref.get("parts", []))


def main():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    raw = {}
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            raw[(row["id"], row["condition"])] = row
    reviews = json.loads(REVIEWS.read_text(encoding="utf-8")) if REVIEWS.exists() else {}

    by_id = {e["id"]: e for e in dataset}
    todo = []
    for (qid, cond), row in raw.items():
        if f"{qid}|{cond}" in reviews or qid not in by_id:
            continue
        verdict, note = metrics.evaluate(by_id[qid], row)
        if verdict is None:
            todo.append((qid, cond, row, note))

    if not todo:
        console.print("[green]Nothing to review.[/green]")
        return

    console.print(f"[bold]{len(todo)} answers to review.[/bold] y = correct, n = wrong, s = skip, q = quit\n")
    try:
        for i, (qid, cond, row, note) in enumerate(todo, 1):
            snippets = find_snippets(row.get("reference"))
            console.print(Panel(
                f"[bold]Question:[/bold] {row['question']}\n\n"
                f"[bold]{cond} answer:[/bold]\n{row['answer']}\n\n"
                f"[dim]check note: {note}[/dim]\n\n"
                f"[bold]Search results fetched for this question:[/bold]\n{snippets[:900] or '(none)'}",
                title=f"{i}/{len(todo)}  {qid}", title_align="left", border_style="yellow"))
            while True:
                choice = console.input("Correct? [y/n/s/q] ").strip().lower()
                if choice in ("y", "n", "s", "q"):
                    break
            if choice == "q":
                break
            if choice in ("y", "n"):
                reviews[f"{qid}|{cond}"] = (choice == "y")
                REVIEWS.write_text(json.dumps(reviews, indent=1), encoding="utf-8")
    except (KeyboardInterrupt, EOFError):
        pass
    console.print(f"\n[green]Saved {len(reviews)} labels.[/green] Run: python -m benchmark.analyze")


if __name__ == "__main__":
    main()