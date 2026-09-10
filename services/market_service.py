"""
Kisan Ki Awaz - Market Data Service
======================================
Integration layer for agricultural market prices.
Designed for future connection to official price APIs
(Pakistan Bureau of Statistics, AMIS, provincial agriculture depts).
"""
from typing import Dict, List

from loguru import logger

from config import settings


class MarketService:
    """
    Provides agricultural market price data.
    Currently uses demo/reference data. Designed so a live API
    can be connected by replacing the _fetch_* methods.
    """

    def __init__(self):
        self.provider = settings.market.active_provider
        logger.info(f"Market service initialized (provider: {self.provider})")

    def get_crop_prices(self, crop: str) -> Dict:
        """
        Get market price information for a crop.

        Returns:
            Dict with keys:
            - crop: str
            - prices: List[Dict] (market name, price, unit, date)
            - trend: str ("up", "down", "stable")
            - source: str
            - is_demo: bool
            - disclaimer: str
        """
        if self.provider == "api":
            return self._fetch_api_prices(crop)

        return self._demo_prices(crop)

    def _fetch_api_prices(self, crop: str) -> Dict:
        """
        Fetch prices from configured market data API.
        Replace this implementation with actual API calls.
        """
        try:
            import requests
            url = settings.market.api_url
            params = {"crop": crop, "api_key": settings.market.api_key}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return {
                "crop": crop,
                "prices": data.get("prices", []),
                "trend": data.get("trend", "stable"),
                "source": "Market Data API",
                "is_demo": False,
                "disclaimer": "",
            }
        except Exception as e:
            logger.error(f"Market API error: {e}")
            return self._demo_prices(crop)

    def _demo_prices(self, crop: str) -> Dict:
        """
        Return reference/demo price data.
        IMPORTANT: These are NOT live prices. They are for demonstration only.
        Farmers should verify current prices from official sources.
        """
        # Reference price ranges based on PBS data (historical, NOT current)
        reference = {
            "wheat": {
                "prices": [
                    {"market": "Lahore Mandi", "price": "Rs. 3,800-4,200", "unit": "40kg", "date": "Reference 2023"},
                    {"market": "Multan Mandi", "price": "Rs. 3,750-4,100", "unit": "40kg", "date": "Reference 2023"},
                    {"market": "Support Price", "price": "Rs. 3,900", "unit": "40kg", "date": "Govt. 2023"},
                ],
                "trend": "stable",
            },
            "rice": {
                "prices": [
                    {"market": "Sheikhupura", "price": "Rs. 120-140", "unit": "kg (Super Basmati)", "date": "Reference 2023"},
                    {"market": "Karachi", "price": "Rs. 60-80", "unit": "kg (IRRI-6)", "date": "Reference 2023"},
                ],
                "trend": "up",
            },
            "cotton": {
                "prices": [
                    {"market": "Multan", "price": "Rs. 8,500-9,500", "unit": "40kg (seed cotton)", "date": "Reference 2023"},
                    {"market": "Faisalabad", "price": "Rs. 8,400-9,400", "unit": "40kg", "date": "Reference 2023"},
                ],
                "trend": "stable",
            },
            "sugarcane": {
                "prices": [
                    {"market": "Sugar Mills", "price": "Rs. 300-350", "unit": "40kg", "date": "Reference 2023"},
                    {"market": "Support Price", "price": "Rs. 300", "unit": "40kg", "date": "Govt. 2023"},
                ],
                "trend": "stable",
            },
        }

        crop_lower = crop.lower()
        data = reference.get(crop_lower, {
            "prices": [{"market": "N/A", "price": "Contact local mandi", "unit": "N/A", "date": "N/A"}],
            "trend": "unknown",
        })

        return {
            "crop": crop,
            "prices": data["prices"],
            "trend": data["trend"],
            "source": "Demo / Reference Data",
            "is_demo": True,
            "disclaimer": (
                "⚠️ These are REFERENCE prices only (2023 data). "
                "Current market prices may differ significantly. "
                "Please verify from your local mandi, AMIS, or "
                "provincial Agriculture Department before making decisions."
            ),
        }

    def get_market_trends(self, crop: str) -> Dict:
        """Get market trend analysis for a crop."""
        return {
            "crop": crop,
            "short_term": "Market price data integration pending. Connect live API for real trends.",
            "long_term": "Based on historical data, agricultural commodity prices in Pakistan have shown general upward trend.",
            "is_demo": True,
        }
