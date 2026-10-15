import requests
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city, e.g. 'Baku'."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=10,
        ).json()
        if not geo.get("results"):
            return f"City '{city}' not found."
        place = geo["results"][0]
        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            }, timeout=10,
        ).json()["current"]
        return (f"{place['name']}, {place.get('country', '')}: "
                f"{w['temperature_2m']}°C, humidity {w['relative_humidity_2m']}%, "
                f"wind {w['wind_speed_10m']} km/h")
    except Exception as e:
        return f"Error: {e}"