"""
Open-Meteo APIで竹富島（西桟橋）の雲量を取得する。

- APIキー不要の完全無料API
- 15分キャッシュでレート制限を意識する
- 取得失敗時は None を返し、UIでは「--」表示
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import httpx

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

# 西桟橋（竹富島）
LAT = 24.3237
LON = 124.0893

# Open-Meteo: 無料、キー不要、観測時間帯の雲量を時間別で返す
# cloud_cover = 低層・中層・上層を統合した0-100%の雲量
WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT}&longitude={LON}"
    "&hourly=cloud_cover,visibility,temperature_2m"
    "&timezone=Asia%2FTokyo"
    "&forecast_days=2"
)

CACHE_MINUTES = 15
_cache: dict = {"data": None, "fetched_at": None}


async def _fetch_raw() -> dict | None:
    headers = {"User-Agent": "StarlinkNishi/3.0 (weather)"}
    try:
        async with httpx.AsyncClient(timeout=20.0, headers=headers, follow_redirects=True) as client:
            resp = await client.get(WEATHER_URL)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning("Open-Meteo取得失敗: %s", e)
        return None


def _summarize_cloud(cloud_pct: float) -> tuple[str, str]:
    """雲量(0-100)から日本語の状態と短いコメントを返す。"""
    if cloud_pct < 15:
        return ("快晴", "空は澄んでいます")
    if cloud_pct < 35:
        return ("晴れ", "星が見やすい空です")
    if cloud_pct < 65:
        return ("晴れ時々曇り", "雲の切れ間から見える可能性があります")
    if cloud_pct < 85:
        return ("曇り", "雲が多く、見えにくいです")
    return ("厚い雲", "星は見えにくいでしょう")


def _extract_window_values(
    raw: dict,
    obs_start: datetime,
    obs_end: datetime,
) -> tuple[list[float], list[float], list[float]]:
    """Open-Meteoレスポンスから観測窓内のcloud/visibility/tempを抽出する。"""
    hourly = raw.get("hourly", {})
    times: list[str] = hourly.get("time", [])
    clouds: list[float] = hourly.get("cloud_cover", [])
    vis: list[float] = hourly.get("visibility", [])
    temps: list[float] = hourly.get("temperature_2m", [])

    start_key = obs_start.strftime("%Y-%m-%dT%H:00")
    end_key = obs_end.strftime("%Y-%m-%dT%H:00")

    matched_clouds: list[float] = []
    matched_vis: list[float] = []
    matched_temps: list[float] = []
    for i, t in enumerate(times):
        if not (start_key <= t <= end_key):
            continue
        if i < len(clouds) and clouds[i] is not None:
            matched_clouds.append(float(clouds[i]))
        if i < len(vis) and vis[i] is not None:
            matched_vis.append(float(vis[i]))
        if i < len(temps) and temps[i] is not None:
            matched_temps.append(float(temps[i]))

    return matched_clouds, matched_vis, matched_temps


async def get_weather_for_window(
    obs_start: datetime,
    obs_end: datetime,
) -> dict | None:
    """観測時間帯（obs_start〜obs_end）の雲量を取得する。

    戻り値: {
      "cloud_pct": 平均雲量 0-100,
      "cloud_pct_min": 最小雲量,
      "cloud_pct_max": 最大雲量,
      "state_jp": "快晴" 等,
      "note": "星が見やすい空です" 等,
      "visibility_km": 平均視程（km）,
      "temp_c": 平均気温（°C）,
      "source": "open-meteo",
      "fetched_at": "ISO"
    }
    取得不能時は None。
    """
    now = datetime.now(tz=JST)
    if (
        _cache["data"] is not None
        and _cache["fetched_at"] is not None
        and (now - _cache["fetched_at"]).total_seconds() < CACHE_MINUTES * 60
    ):
        raw = _cache["data"]
    else:
        raw = await _fetch_raw()
        if raw is None:
            return None
        _cache["data"] = raw
        _cache["fetched_at"] = now

    try:
        clouds, vis, temps = _extract_window_values(raw, obs_start, obs_end)
        if not clouds:
            return None

        avg_cloud = sum(clouds) / len(clouds)
        state_jp, note = _summarize_cloud(avg_cloud)

        return {
            "cloud_pct": round(avg_cloud, 0),
            "cloud_pct_min": round(min(clouds), 0),
            "cloud_pct_max": round(max(clouds), 0),
            "state_jp": state_jp,
            "note": note,
            "visibility_km": round(sum(vis) / len(vis) / 1000, 1) if vis else None,
            "temp_c": round(sum(temps) / len(temps), 1) if temps else None,
            "source": "open-meteo",
            "fetched_at": now.isoformat(),
        }
    except Exception as e:
        logger.warning("Open-Meteoレスポンスのパース失敗: %s", e)
        return None
