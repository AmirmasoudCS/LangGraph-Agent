"""Extend benchmark/dataset.json with more questions (idempotent: existing ids are kept).

Usage (from the project root):
    python -m benchmark.extend_dataset

Existing questions are never changed, so results already in raw.jsonl stay valid.
Expected values are computed by Python, and generation is seeded, so the
output is reproducible.
"""
import json
import random
from pathlib import Path
from zoneinfo import ZoneInfo

PATH = Path(__file__).parent / "dataset.json"
data = json.loads(PATH.read_text(encoding="utf-8"))
have = {d["id"] for d in data}
rng = random.Random(42)


def add(id, category, question, check, tools):
    if id not in have:
        data.append({"id": id, "category": category, "question": question,
                     "expected_tools": tools, "check": check})
        have.add(id)


def num(expected, abs_tol=0.0):
    return {"type": "numeric", "expected": expected, "abs_tol": abs_tol}


# ---------------------------------------------------------------- calculator (harder)
n = 10
def calc(question, value, abs_tol=0.0):
    global n
    add(f"calc-{n:02d}", "calculator", question, num(value, abs_tol), ["calculator"])
    n += 1

for _ in range(5):                                      # big multiplications
    a, b = rng.randint(10000, 99999), rng.randint(1000, 9999)
    calc(f"What is {a} * {b}?", a * b)
a, b, c, d = 48213, 7391, 625, 18
calc(f"Calculate ({a} + {b}) * {c} - {d}.", (a + b) * c - d)
a, b, c, d = 847, 653, 391, 782
calc(f"Calculate {a} * {b} + {c} * {d}.", a * b + c * d)
a, b, c, d = 9123, 4567, 38, 21
calc(f"Calculate ({a} - {b}) * ({c} + {d}).", (a - b) * (c + d))
a, b = 1357, 2468
calc(f"Calculate {a} ** 2 + {b} ** 2.", a ** 2 + b ** 2)
a, b, c = 7654, 321, 17
calc(f"Calculate ({a} * {b}) / {c}. Round to 2 decimal places.", round(a * b / c, 2), 0.0051)
a, b, c = 99991, 37, 41
calc(f"Calculate ({a} + {b} * {c}) / 7. Round to 2 decimal places.", round((a + b * c) / 7, 2), 0.0051)
a, b, c = 3456, 789, 12
calc(f"Calculate {a} * {b} - {c} ** 5.", a * b - c ** 5)
for base, exp in [(13, 11), (7, 15), (23, 8), (17, 10)]:
    calc(f"What is {base} to the power of {exp}?", base ** exp)
for pct, val in [(23.7, 8450), (8.5, 12480), (62.5, 3472)]:
    calc(f"What is {pct}% of {val}?", round(pct / 100 * val, 4), 0.01)
for a, b in [(7919, 313), (123456, 789)]:
    calc(f"What is {a} divided by {b}? Round to 3 decimal places.", round(a / b, 3), 0.0006)

# ---------------------------------------------------------------- datetime
more_cities = [("Mumbai", "Asia/Kolkata"), ("Los Angeles", "America/Los_Angeles"), ("Paris", "Europe/Paris"),
               ("Cairo", "Africa/Cairo"), ("Johannesburg", "Africa/Johannesburg"), ("Singapore", "Asia/Singapore"),
               ("Moscow", "Europe/Moscow"), ("Mexico City", "America/Mexico_City"), ("Kathmandu", "Asia/Kathmandu"),
               ("Adelaide", "Australia/Adelaide"), ("Auckland", "Pacific/Auckland"), ("Anchorage", "America/Anchorage"),
               ("Honolulu", "Pacific/Honolulu"), ("Tehran", "Asia/Tehran")]
for i, (city, tz) in enumerate(more_cities, 7):
    add(f"time-{i:02d}", "datetime", f"What is the current local time in {city}?",
        {"type": "time", "tz": tz, "tol_min": 3}, ["get_current_datetime"])

# ---------------------------------------------------------------- time_difference (tricky zones)
pairs = [("Adelaide", "Australia/Adelaide", "Kathmandu", "Asia/Kathmandu"),
         ("Tehran", "Asia/Tehran", "Kabul", "Asia/Kabul"),
         ("Auckland", "Pacific/Auckland", "London", "Europe/London"),
         ("Anchorage", "America/Anchorage", "Honolulu", "Pacific/Honolulu"),
         ("Cairo", "Africa/Cairo", "Moscow", "Europe/Moscow"),
         ("Denver", "America/Denver", "Berlin", "Europe/Berlin"),
         ("Mumbai", "Asia/Kolkata", "Sydney", "Australia/Sydney"),
         ("Singapore", "Asia/Singapore", "Los Angeles", "America/Los_Angeles"),
         ("Paris", "Europe/Paris", "New York", "America/New_York"),
         ("Johannesburg", "Africa/Johannesburg", "Tokyo", "Asia/Tokyo"),
         ("Mexico City", "America/Mexico_City", "Madrid", "Europe/Madrid"),
         ("Dubai", "Asia/Dubai", "Toronto", "America/Toronto"),
         ("Istanbul", "Europe/Istanbul", "Vancouver", "America/Vancouver"),
         ("Bangkok", "Asia/Bangkok", "Adelaide", "Australia/Adelaide"),
         ("Lagos", "Africa/Lagos", "Auckland", "Pacific/Auckland")]
for i, (a, za, b, zb) in enumerate(pairs, 6):
    add(f"diff-{i:02d}", "time_difference",
        f"How many hours apart are the current local times in {a} and {b}? Answer in hours.",
        {"type": "time_difference", "tz_a": za, "tz_b": zb}, ["time_difference"])

# ---------------------------------------------------------------- weather
W = [("Oslo", "temperature_c", "the current temperature in Oslo in degrees Celsius", 1.5),
     ("Nairobi", "temperature_c", "the current temperature in Nairobi in degrees Celsius", 1.5),
     ("Seoul", "temperature_c", "the current temperature in Seoul in degrees Celsius", 1.5),
     ("Madrid", "temperature_c", "the current temperature in Madrid in degrees Celsius", 1.5),
     ("Cape Town", "temperature_c", "the current temperature in Cape Town in degrees Celsius", 1.5),
     ("Singapore", "temperature_c", "the current temperature in Singapore in degrees Celsius", 1.5),
     ("Berlin", "humidity", "the current relative humidity in Berlin in percent", 6),
     ("Mumbai", "humidity", "the current relative humidity in Mumbai in percent", 6),
     ("Oslo", "humidity", "the current relative humidity in Oslo in percent", 6),
     ("Vienna", "humidity", "the current relative humidity in Vienna in percent", 6),
     ("Sydney", "humidity", "the current relative humidity in Sydney in percent", 6),
     ("Toronto", "humidity", "the current relative humidity in Toronto in percent", 6),
     ("Reykjavik", "wind_kmh", "the current wind speed in Reykjavik in km/h", 4),
     ("Chicago", "wind_kmh", "the current wind speed in Chicago in km/h", 4),
     ("Istanbul", "wind_kmh", "the current wind speed in Istanbul in km/h", 4),
     ("Baku", "wind_kmh", "the current wind speed in Baku in km/h", 4),
     ("Wellington", "wind_kmh", "the current wind speed in Wellington in km/h", 4),
     ("Dubai", "wind_kmh", "the current wind speed in Dubai in km/h", 4)]
for i, (city, field, phrase, tol) in enumerate(W, 7):
    add(f"wx-{i:02d}", "weather", f"What is {phrase}?",
        {"type": "weather", "city": city, "field": field, "tol": tol}, ["get_weather"])

# ---------------------------------------------------------------- web_search
S = [("Who is the current CEO of Alphabet?", ["Pichai"], "Alphabet CEO"),
     ("Who is the current CEO of Amazon?", ["Jassy"], "Amazon CEO"),
     ("Who is the current CEO of Meta?", ["Zuckerberg"], "Meta CEO"),
     ("Who is the current CEO of Tesla?", ["Musk"], "Tesla CEO"),
     ("Who is the current Director-General of the World Health Organization?", ["Tedros"], "WHO Director-General"),
     ("Who is the current Managing Director of the International Monetary Fund?", ["Georgieva"], "IMF Managing Director"),
     ("Who is the current President of the World Bank?", ["Banga"], "World Bank president"),
     ("Who is the current Chancellor of Germany?", ["Merz"], "Chancellor of Germany"),
     ("Who is the current President of France?", ["Macron"], "President of France"),
     ("Who is the current Pope?", ["Leo"], "current Pope")]
for i, (q, kws, query) in enumerate(S, 9):
    add(f"web-{i:02d}", "web_search", q,
        {"type": "search", "keywords": kws, "match": "any", "query": query}, ["web_search"])

# ---------------------------------------------------------------- multi_tool
m = 7
def multi(question, check, tools):
    global m
    add(f"multi-{m:02d}", "multi_tool", question, check, tools)
    m += 1

TIME = lambda tz: {"type": "time", "tz": tz, "tol_min": 3}
for city, tz, pct, val in [("Paris", "Europe/Paris", 18, 2750), ("Singapore", "Asia/Singapore", 35, 1240),
                           ("Los Angeles", "America/Los_Angeles", 7.5, 9600)]:
    multi(f"What is the current local time in {city}, and what is {pct}% of {val}?",
          {"type": "all", "checks": [TIME(tz), num(round(pct / 100 * val, 4), 0.01)]},
          ["get_current_datetime", "calculator"])
for city, k in [("Oslo", 4), ("Madrid", 2)]:
    multi(f"What is the current temperature in {city} in degrees Celsius, and what is that temperature multiplied by {k}?",
          {"type": "weather", "city": city, "field": "temperature_c", "scale": k, "tol": float(k)},
          ["get_weather", "calculator"])
for city, tz in [("Seoul", "Asia/Seoul"), ("Cairo", "Africa/Cairo")]:
    multi(f"What is the current temperature in {city} in degrees Celsius, and what time is it there?",
          {"type": "all", "checks": [{"type": "weather", "city": city, "field": "temperature_c", "tol": 1.5}, TIME(tz)]},
          ["get_weather", "get_current_datetime"])
for (a, za, b, zb), (x, y) in zip([("London", "Europe/London", "Sydney", "Australia/Sydney"),
                                   ("Delhi", "Asia/Kolkata", "Chicago", "America/Chicago")],
                                  [(347, 29), (812, 63)]):
    multi(f"How many hours apart are the current local times in {a} and {b}, and what is {x} * {y}?",
          {"type": "all", "checks": [{"type": "time_difference", "tz_a": za, "tz_b": zb}, num(x * y)]},
          ["time_difference", "calculator"])
for company, kw, q, (x, y) in [("Amazon", "Jassy", "Amazon", (3375, 15)), ("Nvidia", "Huang", "Nvidia", (2640, 24))]:
    multi(f"Who is the current CEO of {company}, and what is {x} divided by {y}?",
          {"type": "all", "checks": [{"type": "search", "keywords": [kw], "match": "any", "query": f"{q} CEO"}, num(x / y, 0.01)]},
          ["web_search", "calculator"])
multi("Which is warmer right now, Oslo or Nairobi? Give both temperatures in degrees Celsius.",
      {"type": "all", "checks": [{"type": "weather", "city": "Oslo", "field": "temperature_c", "tol": 1.5},
                                 {"type": "weather", "city": "Nairobi", "field": "temperature_c", "tol": 1.5}]},
      ["get_weather"])

# ---------------------------------------------------------------- knowledge (control)
K = [("What is the chemical symbol for sodium?", {"type": "contains", "keywords": ["Na"], "match": "any"}),
     ("What is the capital of Australia?", {"type": "contains", "keywords": ["Canberra"], "match": "any"}),
     ("Who wrote the novel Nineteen Eighty-Four?", {"type": "contains", "keywords": ["Orwell"], "match": "any"}),
     ("What is the smallest prime number?", num(2)),
     ("Which planet is known as the Red Planet?", {"type": "contains", "keywords": ["Mars"], "match": "any"}),
     ("Who developed the theory of general relativity?", {"type": "contains", "keywords": ["Einstein"], "match": "any"})]
for i, (q, chk) in enumerate(K, 7):
    add(f"know-{i:02d}", "knowledge", q, chk, [])

# ---------------------------------------------------------------- validate and save
for d in data:
    c = d["check"]
    for part in (c["checks"] if c["type"] == "all" else [c]):
        for key in ("tz", "tz_a", "tz_b"):
            if key in part:
                ZoneInfo(part[key])                      # raises if the zone name is wrong
assert len({d["id"] for d in data}) == len(data), "duplicate ids"
PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
from collections import Counter
print(len(data), dict(Counter(d["category"] for d in data)))