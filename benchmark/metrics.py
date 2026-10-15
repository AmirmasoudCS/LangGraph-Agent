"""Automatic scoring, aggregate metrics and significance tests.

Every checker returns (verdict, note) where verdict is
  True   the answer is correct
  False  the answer is wrong
  None   undecided: needs human review (see benchmark/review.py)
"""
import math
import re
import statistics
import unicodedata
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# =====================================================================
# Text helpers
# =====================================================================
NUM_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUM_RE.finditer(text.replace("\u2212", "-")):
        try:
            out.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return out


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def contains(text: str, keywords: list[str], mode: str = "any") -> bool:
    """Whole-word, case- and accent-insensitive keyword match."""
    t = _normalize(text)
    hits = [re.search(rf"\b{re.escape(_normalize(k))}\b", t) is not None for k in keywords]
    return all(hits) if mode == "all" else any(hits)


# ---------- times ----------
TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(?:([ap])\.?m\b\.?)?", re.I)
HOUR_AMPM_RE = re.compile(r"\b(\d{1,2})\s*([ap])\.?m\b", re.I)
UTC_RE = re.compile(r"(?:UTC|GMT)\s*[+\-\u2212]\s*\d{1,2}(?::?\d{2})?", re.I)


def times_in(text: str) -> set[int]:
    """All clock times mentioned in the text, as minutes since midnight.
    A time without AM/PM yields both readings (3:30 -> 03:30 and 15:30)."""
    text = UTC_RE.sub(" ", text)          # "UTC+09:00" is an offset, not a time
    found: set[int] = set()

    def add(h: int, m: int, ap: str):
        if m > 59 or h > 24:
            return
        if ap:
            if 1 <= h <= 12:
                found.add(((h % 12) + (12 if ap.lower() == "p" else 0)) * 60 + m)
        elif 1 <= h <= 12:
            found.add((h % 12) * 60 + m)
            found.add((h % 12 + 12) * 60 + m)
        else:
            found.add((h % 24) * 60 + m)

    for hh, mm, ap in TIME_RE.findall(text):
        add(int(hh), int(mm), ap)
    for hh, ap in HOUR_AMPM_RE.findall(text):
        add(int(hh), 0, ap)
    return found


# ---------- weather values ----------
F_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°|º|degrees?)?\s*(?:F\b|Fahrenheit\b)", re.I)
C_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*(?:°|º|degrees?\b|deg\b|C\b|Celsius\b)", re.I)
UNIT_RE = {
    "humidity": re.compile(r"(\d+(?:\.\d+)?)\s*%"),
    "wind_kmh": re.compile(
        r"(\d+(?:[.,]\d+)?)\s*(?:km\s*/\s*h|kph|kmh|km per hour|kilomet\w+ per hour)", re.I),
}
DEFAULT_TOL = {"temperature_c": 1.5, "humidity": 6.0, "wind_kmh": 4.0}


def extract_field(field: str, text: str) -> list[float]:
    """Values with the right unit (°C, %, km/h). Fahrenheit is converted.
    If the answer has no unit-qualified number, fall back to every number."""
    text = text.replace("\u2212", "-")
    if field == "temperature_c":
        f_vals = [(float(x) - 32) * 5 / 9 for x in F_RE.findall(text)]
        c_vals = [float(x.replace(",", ".")) for x in C_RE.findall(F_RE.sub(" ", text))]
        vals = c_vals + f_vals
    else:
        vals = [float(x.replace(",", ".")) for x in UNIT_RE[field].findall(text)]
    return vals or numbers_in(text)


# ---------- hours ----------
def hours_candidates(text: str) -> set[float]:
    cands = set(numbers_in(text))
    for h, m in re.findall(
            r"(\d+)\s*(?:h|hours?)\b\s*(?:and\s*)?(\d+)\s*(?:m|min|minutes?)\b", text, re.I):
        cands.add(int(h) + int(m) / 60)
    for m in re.findall(r"(\d+(?:\.\d+)?)\s*(?:min|minutes?)\b", text, re.I):
        cands.add(float(m) / 60)
    return cands


# =====================================================================
# Checkers
# =====================================================================
def check_numeric(spec, answer, question):
    exp = float(spec["expected"])
    rel, abs_tol = spec.get("rel_tol", 1e-9), spec.get("abs_tol", 0.0)
    qnums = set(numbers_in(question))
    cands = numbers_in(answer)
    if exp not in qnums:                   # ignore operands repeated from the question
        cands = [c for c in cands if c not in qnums]
    if not cands:
        return False, "no number in answer"
    ok = any(math.isclose(c, exp, rel_tol=rel, abs_tol=abs_tol) for c in cands)
    return ok, f"expected {exp:g}, answer numbers {cands[:6]}"


def check_contains(spec, answer):
    ok = contains(answer, spec["keywords"], spec.get("match", "any"))
    return ok, f"keywords {spec['keywords']}"


def check_time(spec, answer, window):
    start, end = window
    if start is None or end is None:
        return None, "no timing window"
    tz = ZoneInfo(spec["tz"])
    tol = spec.get("tol_min", 3)
    expected, t = set(), start
    while t <= end + timedelta(minutes=1):          # every minute the run could have read the clock
        local = t.astimezone(tz)
        expected.add(local.hour * 60 + local.minute)
        t += timedelta(minutes=1)
    cands = times_in(answer)
    if not cands:
        return False, "no clock time in answer"

    def dist(a, b):
        d = abs(a - b) % 1440
        return min(d, 1440 - d)

    ok = any(dist(c, e) <= tol for c in cands for e in expected)
    return ok, f"tz {spec['tz']}, answer times {sorted(cands)[:4]} (minutes since midnight)"


def check_weather(spec, answer, ref):
    if not ref or "error" in ref:
        return None, f"no weather reference ({(ref or {}).get('error', 'missing')})"
    field = spec.get("field", "temperature_c")
    scale = spec.get("scale", 1)
    truth = ref[field] * scale
    tol = spec.get("tol", DEFAULT_TOL[field])
    cands = extract_field(field, answer) if scale == 1 else numbers_in(answer)
    ok = any(abs(c - truth) <= tol for c in cands)
    return ok, f"{field}: API {truth:g} (±{tol}), answer {cands[:5]}"


def check_time_difference(spec, answer, ref):
    if not ref or "error" in ref:
        return None, "no time-difference reference"
    truth = abs(ref["hours"])
    ok = any(abs(c - truth) <= 0.02 for c in hours_candidates(answer))
    return ok, f"expected {truth:g} hours"


STOP = {"The", "This", "That", "These", "Based", "According", "Here", "However", "There", "Its"}


def support_hint(answer: str, snippets: str) -> str:
    tokens = set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b|\b\d{3,}\b", answer)) - STOP
    if not tokens or not snippets:
        return ""
    found = [t for t in tokens if re.search(rf"\b{re.escape(t)}\b", snippets, re.I)]
    return f"{len(found)}/{len(tokens)} salient terms also appear in the search results"


def check_search(spec, answer, ref):
    kws = spec.get("keywords", [])
    snippets = (ref or {}).get("snippets", "")
    if not kws:
        return None, "needs human review. " + support_hint(answer, snippets)
    mode = spec.get("match", "any")
    if not contains(answer, kws, mode):
        return False, f"keywords {kws} missing from answer"
    if not snippets:
        return True, "keywords found (no search reference available)"
    if contains(snippets, kws, mode):
        return True, "keywords found in answer and confirmed by search results"
    return None, "answer has the keywords but the search results do not confirm them; review"


def check(spec, answer, ref, window, question):
    kind = spec["type"]
    if kind == "numeric":
        return check_numeric(spec, answer, question)
    if kind == "contains":
        return check_contains(spec, answer)
    if kind == "time":
        return check_time(spec, answer, window)
    if kind == "weather":
        return check_weather(spec, answer, ref)
    if kind == "time_difference":
        return check_time_difference(spec, answer, ref)
    if kind == "search":
        return check_search(spec, answer, ref)
    if kind == "all":
        parts = (ref or {}).get("parts", [None] * len(spec["checks"]))
        results = [check(s, answer, parts[i], window, question) for i, s in enumerate(spec["checks"])]
        verdicts = [r[0] for r in results]
        note = " | ".join(r[1] for r in results)
        if False in verdicts:
            return False, note
        if None in verdicts:
            return None, note
        return True, note
    raise ValueError(f"unknown check type: {kind}")


def _parse(ts):
    return datetime.fromisoformat(ts) if ts else None


def evaluate(entry: dict, row: dict):
    """Score one raw result row against its dataset entry."""
    answer = (row.get("answer") or "").strip()
    if row.get("error") or not answer:
        return False, f"no answer ({row.get('error') or 'empty'})"
    window = (_parse(row.get("started_at")), _parse(row.get("finished_at")))
    return check(entry["check"], answer, row.get("reference"), window, entry["question"])


def score_rows(dataset: list[dict], raw: list[dict], reviews: dict) -> list[dict]:
    """Attach correctness to every raw row. Human reviews fill in undecided rows."""
    by_id = {e["id"]: e for e in dataset}
    scored = []
    for row in raw:
        entry = by_id.get(row["id"])
        if entry is None:
            continue
        verdict, note = evaluate(entry, row)
        source = "auto"
        key = f"{row['id']}|{row['condition']}"
        if verdict is None and key in reviews:
            verdict, source = reviews[key], "human"
        scored.append({
            "id": row["id"], "category": entry["category"], "condition": row["condition"],
            "correct": verdict, "source": source, "note": note,
            "latency_s": row.get("latency_s", 0.0), "llm_calls": row.get("llm_calls", 0),
            "tool_calls": row.get("tool_calls", 0),
            "input_tokens": row.get("input_tokens", 0), "output_tokens": row.get("output_tokens", 0),
            "revisions": row.get("revisions", 0), "error": row.get("error", ""),
            "tools_used": row.get("tools_used", []), "expected_tools": entry.get("expected_tools", []),
            "question": entry["question"], "answer": (row.get("answer") or "").strip(),
        })
    return scored


# =====================================================================
# Aggregation
# =====================================================================
def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # clamp so the interval always contains p (avoids tiny negative error bars)
    return min(p, max(0.0, center - half)), max(p, min(1.0, center + half))


def summarize(rows: list[dict]) -> dict:
    decided = [r for r in rows if r["correct"] is not None]
    n, k = len(decided), sum(1 for r in decided if r["correct"])
    lo, hi = wilson(k, n)

    def mean(key):
        return statistics.mean(r[key] for r in rows) if rows else 0.0

    needing_tools = [r for r in rows if r["expected_tools"] and r["tools_used"] is not None
                     and r["condition"] != "baseline"]
    tool_use = (sum(set(r["expected_tools"]) <= set(r["tools_used"]) for r in needing_tools)
                / len(needing_tools)) if needing_tools else None
    return {
        "runs": len(rows), "decided": n, "undecided": len(rows) - n, "correct": k,
        "accuracy": k / n if n else 0.0, "ci_low": lo, "ci_high": hi,
        "mean_latency_s": mean("latency_s"),
        "median_latency_s": statistics.median(r["latency_s"] for r in rows) if rows else 0.0,
        "mean_llm_calls": mean("llm_calls"), "mean_tool_calls": mean("tool_calls"),
        "mean_tokens": mean("input_tokens") + mean("output_tokens"),
        "mean_revisions": mean("revisions"),
        "error_rate": sum(1 for r in rows if r["error"]) / len(rows) if rows else 0.0,
        "tool_use_rate": tool_use,
    }


# =====================================================================
# McNemar's test (paired binary outcomes)
# =====================================================================
def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value (binomial test on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def mcnemar_chi2(b: int, c: int):
    """Chi-square statistic with continuity correction, and its p-value (1 d.o.f.)."""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / n
    return stat, math.erfc(math.sqrt(stat / 2))


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def paired(rows: list[dict], cond_a: str, cond_b: str) -> dict:
    """Pair the two conditions question by question and run McNemar.
    Only questions decided (True/False) under BOTH conditions are used.
    b = A right, B wrong.   c = A wrong, B right."""
    a = {r["id"]: r["correct"] for r in rows if r["condition"] == cond_a}
    b_ = {r["id"]: r["correct"] for r in rows if r["condition"] == cond_b}
    ids = [i for i in a if i in b_ and a[i] is not None and b_[i] is not None]
    both_right = sum(a[i] and b_[i] for i in ids)
    only_a = sum(a[i] and not b_[i] for i in ids)
    only_b = sum((not a[i]) and b_[i] for i in ids)
    both_wrong = sum((not a[i]) and (not b_[i]) for i in ids)
    n = len(ids)
    stat, p_chi = mcnemar_chi2(only_a, only_b)
    return {
        "pairs": n, "both_right": both_right, "only_a": only_a, "only_b": only_b,
        "both_wrong": both_wrong,
        "acc_a": (both_right + only_a) / n if n else 0.0,
        "acc_b": (both_right + only_b) / n if n else 0.0,
        "p_exact": mcnemar_exact(only_a, only_b), "chi2": stat, "p_chi2": p_chi,
    }