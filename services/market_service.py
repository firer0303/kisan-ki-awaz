"""
Kisan Ki Awaz - Market Data Service
======================================
Live agricultural market-price integration with AMIS Punjab,
with a safe reference-data fallback.
"""
from typing import Dict, List
from loguru import logger
from config import settings


class MarketService:
    """Provides live agricultural market prices when available."""

    AMIS_COMMODITY_IDS = {
        "wheat": 1,
    }

    def __init__(self):
        self.provider = settings.market.active_provider
        logger.info(f"Market service initialized (provider: {self.provider})")

    def get_crop_prices(self, crop: str) -> Dict:
        crop_key = (crop or "").strip().lower()

        # Use explicitly configured market API first when present.
        if self.provider == "api":
            return self._fetch_api_prices(crop)

        # Otherwise use the official AMIS Punjab wholesale/mandi feed.
        if crop_key in self.AMIS_COMMODITY_IDS:
            live = self._fetch_amis_prices(crop_key)
            if live.get("prices"):
                return live

        return self._demo_prices(crop)

    def _fetch_amis_prices(self, crop: str) -> Dict:
        """Fetch the latest available wholesale/mandi prices from AMIS Punjab."""
        import re
        import requests
        from bs4 import BeautifulSoup

        commodity_id = self.AMIS_COMMODITY_IDS.get(crop)
        url = f"https://www.amis.pk/ViewPrices.aspx?commodityId={commodity_id}&searchType=0"

        try:
            response = requests.get(
                url,
                headers={"User-Agent": "KisanKiAwaz/1.0"},
                timeout=8,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            page_text = soup.get_text(" ", strip=True)
            date_match = re.search(r"Dated:\s*([0-9]{2}-[0-9]{2}-[0-9]{4})", page_text)
            report_date = date_match.group(1) if date_match else "Latest available"

            prices = []
            for row in soup.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
                if not cells:
                    continue

                # Expected AMIS row: serial, market, graph, min, max, FQP, quantity.
                if len(cells) < 6:
                    continue

                market = ""
                for value in cells:
                    cleaned = value.strip()
                    if cleaned and cleaned.lower() not in {"graph", "-"} and not re.fullmatch(r"\d+", cleaned):
                        market = cleaned
                        break

                numeric = []
                for value in cells:
                    value = value.replace(",", "").strip()
                    if re.fullmatch(r"\d+(?:\.\d+)?", value):
                        numeric.append(value)

                # For AMIS the useful fields after the market are Min, Max, FQP.
                if market and len(numeric) >= 3:
                    try:
                        # Keep plausible price values; ignore serial numbers.
                        vals = [float(x) for x in numeric[-3:]]
                        if max(vals) > 0:
                            prices.append({
                                "market": market,
                                "price": f"Rs. {vals[0]:,.0f}-{vals[1]:,.0f}",
                                "average": f"Rs. {vals[2]:,.0f}",
                                "unit": "100kg",
                                "date": report_date,
                            })
                    except Exception:
                        continue

            # De-duplicate rows and keep a compact list for the farmer.
            unique = []
            seen = set()
            for item in prices:
                key = item["market"].lower()
                if key in seen:
                    continue
                seen.add(key)
                unique.append(item)

            return {
                "crop": crop,
                "prices": unique[:12],
                "trend": "unknown",
                "source": "AMIS Punjab",
                "source_url": url,
                "is_demo": False,
                "disclaimer": (
                    "یہ AMIS پنجاب کی تازہ دستیاب منڈی قیمت ہے۔ "
                    "آپ کی مقامی منڈی میں ریٹ کچھ مختلف ہو سکتا ہے۔"
                ),
            }

        except Exception as e:
            logger.warning(f"AMIS live price fetch failed: {e}")
            return {
                "crop": crop,
                "prices": [],
                "trend": "unknown",
                "source": "AMIS Punjab",
                "source_url": url,
                "is_demo": False,
                "disclaimer": "",
            }

    def _fetch_api_prices(self, crop: str) -> Dict:
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
        """Historical/reference fallback only; never label this as live."""
        reference = {
            "wheat": {
                "prices": [
                    {"market": "Lahore Mandi", "price": "Rs. 3,800-4,200", "unit": "40kg", "date": "Reference 2023"},
                    {"market": "Multan Mandi", "price": "Rs. 3,750-4,100", "unit": "40kg", "date": "Reference 2023"},
                ],
                "trend": "stable",
            },
            "rice": {
                "prices": [
                    {"market": "Sheikhupura", "price": "Rs. 120-140", "unit": "kg (Super Basmati)", "date": "Reference 2023"},
                ],
                "trend": "up",
            },
            "cotton": {
                "prices": [
                    {"market": "Multan", "price": "Rs. 8,500-9,500", "unit": "40kg", "date": "Reference 2023"},
                ],
                "trend": "stable",
            },
            "sugarcane": {
                "prices": [
                    {"market": "Sugar Mills", "price": "Rs. 300-350", "unit": "40kg", "date": "Reference 2023"},
                ],
                "trend": "stable",
            },
        }

        data = reference.get((crop or "").lower(), {
            "prices": [],
            "trend": "unknown",
        })

        return {
            "crop": crop,
            "prices": data["prices"],
            "trend": data["trend"],
            "source": "Reference Data",
            "source_url": "",
            "is_demo": True,
            "disclaimer": (
                "موجودہ لائیو ریٹ دستیاب نہیں ہو سکا۔ یہ صرف پرانا حوالہ جاتی ڈیٹا ہے۔ "
                "اپنی مقامی منڈی سے موجودہ ریٹ ضرور چیک کریں۔"
            ),
        }

    def get_market_trends(self, crop: str) -> Dict:
        return {
            "crop": crop,
            "short_term": "Live market trend depends on the latest mandi feed.",
            "long_term": "Historical agricultural prices can vary by crop and region.",
            "is_demo": True,
        }
