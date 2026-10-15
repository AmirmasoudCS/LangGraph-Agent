import requests
from langchain_core.tools import tool

# WMO weather codes used by Open-Meteo, simplified
WEATHER_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "rain showers", 82: "heavy showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather AND the forecast for the next 12 hours
    (in 3-hour steps) for a city, e.g. 'Baku'. Use this for questions about
    now, later today, this evening, or tonight."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=10,
        ).json()
        if not geo.get("results"):
            return f"City '{city}' not found."
        place = geo["results"][0]

        data = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                "hourly": "temperature_2m,precipitation_probability,weather_code",
                "forecast_days": 2,
                "timezone": "auto",
            }, timeout=10,
        ).json()

        cur = data["current"]
        lines = [
            f"{place['name']}, {place.get('country', '')} (local time {cur['time'].replace('T', ' ')})",
            f"NOW: {cur['temperature_2m']}°C, humidity {cur['relative_humidity_2m']}%, "
            f"wind {cur['wind_speed_10m']} km/h",
            "FORECAST:",
        ]

        hourly = data["hourly"]
        # first hourly slot that is after the current time
        start = next((i for i, t in enumerate(hourly["time"]) if t > cur["time"]), 0)
        for i in range(start, min(start + 12, len(hourly["time"])), 3):
            t = hourly["time"][i][11:16]          # "HH:MM"
            desc = WEATHER_CODES.get(hourly["weather_code"][i], "unknown")
            lines.append(
                f"  {t}: {hourly['temperature_2m'][i]}°C, {desc}, "
                f"{hourly['precipitation_probability'][i]}% chance of precipitation"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"