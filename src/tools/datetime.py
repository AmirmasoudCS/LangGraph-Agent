from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from langchain_core.tools import tool


def _describe(dt: datetime, label: str) -> str:
    offset = dt.strftime("%z")                       # e.g. "+0900"
    offset = f"UTC{offset[:3]}:{offset[3:]}"         # -> "UTC+09:00"
    return f"{label}: {dt.strftime('%A, %Y-%m-%d %H:%M')} ({offset})"


def _timezone_for(location: str):
    """Return (tzinfo, label) for a city name or IANA timezone, or (None, error)."""
    name = location.strip()

    # 0. Empty / "local" means the user's own timezone
    if name.lower() in ("", "local", "here", "my location"):
        return datetime.now().astimezone().tzinfo, "Local"

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
    tz, label = _timezone_for(location)
    if tz is None:
        return label                                  # label holds the error message
    if label == "Local":
        return _describe(datetime.now().astimezone(), "Local time")
    return _describe(datetime.now(tz), label)


@tool
def time_difference(location_a: str, location_b: str) -> str:
    """Compute the time difference between two places, e.g. 'Baku' and
    'New York'. Pass a city or timezone name for each; use an empty string for
    the user's own local time. Use this for questions like 'how many hours
    ahead is X of Y' or 'what time is it in X when it is 3 PM in Y'.
    Never calculate time differences yourself."""
    tz_a, label_a = _timezone_for(location_a)
    if tz_a is None:
        return label_a
    tz_b, label_b = _timezone_for(location_b)
    if tz_b is None:
        return label_b

    now_a = datetime.now(tz_a)
    now_b = datetime.now(tz_b)
    # Current UTC offsets, so daylight saving time is taken into account
    minutes = int((now_b.utcoffset() - now_a.utcoffset()).total_seconds() // 60)

    if minutes == 0:
        relation = f"{label_a} and {label_b} are in the same time zone right now."
    else:
        hours, mins = divmod(abs(minutes), 60)
        amount = f"{hours}h {mins}m" if mins else f"{hours}h"
        direction = "ahead of" if minutes > 0 else "behind"
        relation = f"{label_b} is {amount} {direction} {label_a}."

    return (
        f"{relation}\n"
        f"{_describe(now_a, label_a)}\n"
        f"{_describe(now_b, label_b)}"
    )