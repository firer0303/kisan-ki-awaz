"""
Kisan Ki Awaz - Weather Service
=================================
Integration layer for weather data from Pakistan Meteorological
Department (PMD) or OpenWeatherMap.
"""
from typing import Dict, Optional

import requests
from loguru import logger

from config import settings


class WeatherService:
    """
    Provides weather data and agricultural risk alerts.
    Designed for future integration with PMD API.
    """

    # Default locations for Pakistani agricultural regions
    PAKISTAN_REGIONS = {
        "punjab_central": {"name": "Faisalabad", "lat": 31.42, "lon": 73.08},
        "punjab_south": {"name": "Multan", "lat": 30.20, "lon": 71.53},
        "sindh_central": {"name": "Hyderabad", "lat": 25.39, "lon": 68.37},
        "kpk": {"name": "Peshawar", "lat": 34.01, "lon": 71.58},
        "balochistan": {"name": "Quetta", "lat": 30.18, "lon": 66.97},
    }

    def __init__(self):
        self.provider = settings.weather.active_provider
        logger.info(f"Weather service initialized (provider: {self.provider})")

    def get_weather(self, region_key: str = "punjab_central") -> Dict:
        """
        Get current weather and agricultural advisories for a region.

        Returns:
            Dict with keys:
            - temperature_c: float
            - humidity_pct: float
            - condition: str
            - wind_kph: float
            - agricultural_advisory: str
            - risk_alerts: list
            - is_demo: bool
            - source: str
        """
        region = self.PAKISTAN_REGIONS.get(region_key, self.PAKISTAN_REGIONS["punjab_central"])

        if self.provider == "openweathermap" and settings.weather.openweathermap_key:
            return self._fetch_openweathermap(region)

        return self._demo_weather(region)

    def _fetch_openweathermap(self, region: Dict) -> Dict:
        """Fetch from OpenWeatherMap API."""
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": region["lat"],
                "lon": region["lon"],
                "appid": settings.weather.openweathermap_key,
                "units": "metric",
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            return {
                "temperature_c": data["main"]["temp"],
                "humidity_pct": data["main"]["humidity"],
                "condition": data["weather"][0]["description"],
                "wind_kph": data["wind"]["speed"] * 3.6,
                "agricultural_advisory": self._generate_advisory(
                    data["main"]["temp"], data["main"]["humidity"]
                ),
                "risk_alerts": self._generate_risk_alerts(
                    data["main"]["temp"], data["main"]["humidity"],
                    data["weather"][0]["description"]
                ),
                "is_demo": False,
                "source": "OpenWeatherMap",
                "location": region["name"],
            }
        except Exception as e:
            logger.error(f"OpenWeatherMap API error: {e}")
            return self._demo_weather(region)

    def _demo_weather(self, region: Dict) -> Dict:
        """Return demo weather data for demonstration purposes."""
        # Simulate typical Pakistan agricultural weather
        import random
        random.seed(hash(region["name"]))

        temp = random.uniform(22, 38)
        humidity = random.uniform(30, 75)
        conditions = ["Clear", "Partly Cloudy", "Cloudy", "Light Rain", "Humid"]
        condition = random.choice(conditions)

        return {
            "temperature_c": round(temp, 1),
            "humidity_pct": round(humidity, 1),
            "condition": condition,
            "wind_kph": round(random.uniform(5, 25), 1),
            "agricultural_advisory": self._generate_advisory(temp, humidity),
            "risk_alerts": self._generate_risk_alerts(temp, humidity, condition),
            "is_demo": True,
            "source": "Demo (configure weather API in .env)",
            "location": region["name"],
        }

    def _generate_advisory(self, temp: float, humidity: float) -> str:
        """Generate agricultural advisory based on weather conditions."""
        advisories = []
        if temp > 35:
            advisories.append("High temperature: Increase irrigation frequency. Consider mulching to reduce soil temperature.")
        if temp < 10:
            advisories.append("Cold weather: Protect sensitive crops. Delay sowing of summer crops.")
        if humidity > 70:
            advisories.append("High humidity: Watch for fungal diseases. Avoid overhead irrigation.")
        if humidity < 30:
            advisories.append("Low humidity: Increase irrigation. Watch for pest outbreaks (whitefly, mites).")
        if not advisories:
            advisories.append("Moderate conditions: Follow regular crop management schedule.")
        return " | ".join(advisories)

    def _generate_risk_alerts(self, temp: float, humidity: float, condition: str) -> list:
        """Generate weather risk alerts for agriculture."""
        alerts = []
        if temp > 40:
            alerts.append({"level": "high", "message": "Extreme heat: Crop stress likely. Apply emergency irrigation."})
        if humidity > 80 and temp > 20:
            alerts.append({"level": "high", "message": "Disease risk HIGH: Conditions favorable for fungal diseases."})
        if "rain" in condition.lower():
            alerts.append({"level": "medium", "message": "Rain expected: Delay pesticide/fungicide application."})
        if temp > 30 and humidity > 60:
            alerts.append({"level": "medium", "message": "Pest risk: Conditions favorable for pest multiplication."})
        return alerts

    def get_available_regions(self) -> Dict:
        """Return available agricultural regions."""
        return dict(self.PAKISTAN_REGIONS)
