"""Run the benchmark: every question under every selected condition.

Usage (from the project root):
    python -m benchmark.run_benchmark                      # baseline vs agent_full
    python -m benchmark.run_benchmark --limit 5            # quick trial
    python -m benchmark.run_benchmark --categories weather calculator
    python -m benchmark.run_benchmark --ids calc-01 wx-02
    python -m benchmark.run_benchmark --fresh              # discard old results

Results are appended to benchmark/results/raw.jsonl after every run, and runs
that are already in the file are skipped, so you can stop (Ctrl+C) and resume.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from benchmark import references
from benchmark.runners import CONDITIONS
from src.ollama_server import ensure_server

HERE = Path(__file__).parent
DATASET = HERE / "dataset.json"
RAW = HERE / "results" / "raw.jsonl"

console = Console()


def load_done() -> set:
    done = set()
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["id"], row["condition"]))
    return done


def main():
    parser = argparse.ArgumentParser(description="Run the agent benchmark.")
    parser.add_argument("--conditions", nargs="+", default=["baseline", "agent_full"],
                        choices=list(CONDITIONS))
    parser.add_argument("--categories", nargs="+", help="only these categories")
    parser.add_argument("--ids", nargs="+", help="only these question ids")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--fresh", action="store_true", help="delete previous results first")
    args = parser.parse_args()

    entries = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.categories:
        entries = [e for e in entries if e["category"] in args.categories]
    if args.ids:
        entries = [e for e in entries if e["id"] in args.ids]
    if args.limit:
        entries = entries[:args.limit]

    RAW.parent.mkdir(exist_ok=True)
    if args.fresh and RAW.exists():
        RAW.unlink()
    done = load_done()

    ensure_server()
    console.print("[dim]warming up the model (not recorded)...[/dim]")
    CONDITIONS["baseline"]("Say OK.")

    total = len(entries) * len(args.conditions)
    console.print(f"[bold]{len(entries)} questions x {len(args.conditions)} conditions "
                  f"= {total} runs[/bold] ({len(done)} already done)\n")

    try:
        for entry in entries:
            todo = [c for c in args.conditions if (entry["id"], c) not in done]
            if not todo:
                continue
            # One live reference per question, fetched right before the runs,
            # so both conditions are scored against the same ground truth.
            ref = references.fetch(entry["check"], entry["question"])

            for cond in todo:
                started = datetime.now(timezone.utc)
                result = CONDITIONS[cond](entry["question"])
                finished = datetime.now(timezone.utc)
                row = {
                    "id": entry["id"], "category": entry["category"],
                    "question": entry["question"], **result.to_dict(),
                    "reference": ref,
                    "started_at": started.isoformat(), "finished_at": finished.isoformat(),
                }
                with RAW.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

                status = "[red]error[/red]" if result.error else "[green]ok[/green]"
                preview = result.answer.replace("\n", " ")[:70]
                console.print(f"{status} [bold yellow]{entry['id']}[/bold yellow] "
                              f"[cyan]{cond}[/cyan] {result.latency_s}s "
                              f"[dim]llm={result.llm_calls} tools={result.tool_calls}[/dim] {preview}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped. Run the same command again to resume.[/yellow]")
        return

    console.print("\n[bold green]Done.[/bold green] Next: python -m benchmark.analyze")


if __name__ == "__main__":
    main()