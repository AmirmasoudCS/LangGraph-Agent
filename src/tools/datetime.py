from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from langchain_core.tools import tool


def _describe(dt: datetime, label: str) -> str:
    offset = dt.strftime("%z")                       # e.g. "+0900"
    offset = f"UTC{offset[:3]}:{offset[3:]}"         # -> "UTC+09:00"
    return f"{label}: {dt.strftime('%A, %Y-%m-%d %H:%M')} ({offset})"


def _timezone_for(location: str):
    """Return (ZoneInfo, label) for a city name or IANA timezone, or (None, error)."""
    name = location.strip()

    # 1. Direct IANA names such as "Asia/Tokyo" or "UTC"
    if "/" in name or name.upper() == "UTC":
        try:
            key = name.replace(" ", "_")
            return ZoneInfo(key), key
        except ZoneInfoNotFoundError:
            pass  # fall through and try it as a city name

    # 2. City name -> geocoding API, which includes the timezone
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1}, timeout=10,
        ).json()
    except Exception as e:
        return None, f"Error looking up '{location}': {e}"

    if not geo.get("results"):
        return None, f"Location '{location}' not found."
    place = geo["results"][0]
    label = f"{place['name']}, {place.get('country', '')}".strip(", ")
    try:
        return ZoneInfo(place["timezone"]), label
    except (KeyError, ZoneInfoNotFoundError):
        return None, f"No timezone found for '{location}'."


@tool
def get_current_datetime(location: str = "") -> str:
    """Get the current date and time. Leave 'location' empty for the user's own
    local time. For anywhere else, pass a city or a timezone name, e.g. 'Tokyo',
    'New York', or 'America/Los_Angeles'. Use this for any question about the
    current time or date, in any place."""
    if location.strip().lower() in ("", "local", "here", "my location"):
        return _describe(datetime.now().astimezone(), "Local time")

    tz, label = _timezone_for(location)
    if tz is None:
        return label                                  # label holds the error message
    return _describe(datetime.now(tz), label)