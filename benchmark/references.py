"""Ground truth for questions whose answer changes over time.

The benchmark fetches these itself, directly from the APIs, once per question
right before the conditions run. Both conditions are then scored against the
same reference, so the comparison stays paired and fair.

  weather          current values from Open-Meteo
  time_difference  UTC offsets computed with zoneinfo
  search           top DuckDuckGo snippets for a reference query
  all              a list of sub-checks, each with its own reference
Other check types (numeric, contains, time) need no live reference.
"""
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
from ddgs import DDGS

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather(city: str) -> dict:
    geo = requests.get(GEO_URL, params={"name": city, "count": 1}, timeout=10).json()
    place = geo["results"][0]
    cur = requests.get(
        FORECAST_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        },
        timeout=10,
    ).json()["current"]
    return {
        "city": place["name"],
        "temperature_c": cur["temperature_2m"],
        "humidity": cur["relative_humidity_2m"],
        "wind_kmh": cur["wind_speed_10m"],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_time_difference(tz_a: str, tz_b: str) -> dict:
    now = datetime.now(timezone.utc)
    off_a = now.astimezone(ZoneInfo(tz_a)).utcoffset()
    off_b = now.astimezone(ZoneInfo(tz_b)).utcoffset()
    return {"hours": (off_b - off_a).total_seconds() / 3600}   # positive: b is ahead of a


def fetch_search(query: str) -> dict:
    last_error = None
    for attempt in range(3):                      # DuckDuckGo sometimes rate-limits
        try:
            results = DDGS().text(query, max_results=6)
            text = "\n".join(f"{r['title']}. {r['body']}" for r in results)
            if text.strip():
                return {"query": query, "snippets": text}
        except Exception as e:
            last_error = e
        time.sleep(2 * (attempt + 1))
    return {"error": f"search failed: {last_error}"}


def fetch(spec: dict, question: str = "") -> dict:
    """Return the live reference for a check spec (an empty dict if none is needed)."""
    kind = spec["type"]
    try:
        if kind == "weather":
            return fetch_weather(spec["city"])
        if kind == "time_difference":
            return fetch_time_difference(spec["tz_a"], spec["tz_b"])
        if kind == "search":
            return fetch_search(spec.get("query", question))
        if kind == "all":
            return {"parts": [fetch(s, question) for s in spec["checks"]]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {}