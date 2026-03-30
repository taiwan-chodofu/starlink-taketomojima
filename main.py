"""
竹富島・西桟橋 スターリンク トレイン ビューワー

観測地点: 西桟橋（竹富島） 24.3237°N, 124.0893°E
対象時間: 18:00〜21:00 JST
条件: 高度30°以上、薄明中（太陽 -6°〜-18°）
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.almanac import dark_twilight_day
from skyfield.framelib import ecliptic_frame
from datetime import datetime, timedelta, timezone
from pathlib import Path
import httpx
import asyncio
import logging
import math
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Starlink Nishi-Sanbashi")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# --- 定数 ---
LAT = 24.3237
LON = 124.0893
OBSERVER = wgs84.latlon(LAT, LON)
JST = timezone(timedelta(hours=9))
TLE_URLS = [
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/supplemental/starlink.txt",
    "https://www.celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
]
TLE_CACHE_MINUTES = 120
MIN_ALT_DEG = 30.0
OBS_START_HOUR = 18
OBS_END_HOUR = 21
# トレイン判定: 同一時間帯に近い軌道を通る衛星数の閾値
TRAIN_CLUSTER_THRESHOLD = 3
TRAIN_TIME_WINDOW_SEC = 300  # 5分以内に通過する衛星群
TRAIN_AZ_TOLERANCE_DEG = 30.0  # 方位角の許容差

ts = load.timescale()
eph = load('de421.bsp')

# --- TLEキャッシュ ---
_tle_cache = {"data": None, "fetched_at": None}


# --- 方角ユーティリティ ---
DIRECTION_NAMES = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"]
DIRECTION_CONTEXT = {
    "北": "島の奥側",
    "北東": "島の上空",
    "東": "石垣島の方向",
    "南東": "石垣島寄りの海",
    "南": "海の方向",
    "南西": "海の方向（夕日側）",
    "西": "夕日の方向",
    "北西": "島の奥・海寄り",
}


def az_to_direction(az_deg: float) -> str:
    """方位角（度）を8方位名に変換する。"""
    index = int((az_deg + 22.5) // 45) % 8
    return DIRECTION_NAMES[index]


def az_to_context(az_deg: float) -> str:
    """方位角を西桟橋基準の体験的な表現に変換する。"""
    name = az_to_direction(az_deg)
    return DIRECTION_CONTEXT.get(name, name)


# --- TLE取得（キャッシュ付き） ---
async def fetch_tle_data() -> list[tuple[str, str, str]]:
    """CelestrakからスターリンクTLEを取得する。キャッシュ有効期間内は再取得しない。"""
    now = datetime.now(tz=JST)
    if (
        _tle_cache["data"] is not None
        and _tle_cache["fetched_at"] is not None
        and (now - _tle_cache["fetched_at"]).total_seconds() < TLE_CACHE_MINUTES * 60
    ):
        logger.info("TLEキャッシュ使用（%d衛星）", len(_tle_cache["data"]))
        return _tle_cache["data"]

    logger.info("TLEデータ取得開始")
    headers = {"User-Agent": "StarlinkNishi/1.0 (satellite-viewer)"}
    last_error = None
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        for url in TLE_URLS:
            try:
                logger.info("TLE URL試行: %s", url)
                resp = await client.get(url)
                resp.raise_for_status()
                break
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                logger.warning("TLE取得失敗(%s): %s", url, e)
                last_error = e
                continue
        else:
            raise last_error or RuntimeError("全TLE URLが失敗")

    lines = resp.text.strip().splitlines()
    sats = []
    for i in range(0, len(lines) - 2, 3):
        name = lines[i].strip()
        l1 = lines[i + 1].strip()
        l2 = lines[i + 2].strip()
        if l1.startswith("1 ") and l2.startswith("2 "):
            sats.append((name, l1, l2))

    _tle_cache["data"] = sats
    _tle_cache["fetched_at"] = now
    logger.info("TLEデータ取得完了: %d衛星", len(sats))
    return sats


# --- 薄明判定 ---
def is_observable_twilight(t_skyfield) -> bool:
    """
    太陽高度が-18°〜-6°（天文薄明〜市民薄明）の間かを判定する。
    衛星が太陽光を反射して見えるのは、地上は暗いが衛星高度では太陽が当たる時間帯。
    """
    sun = eph['sun']
    earth = eph['earth']
    observer_pos = earth + OBSERVER
    astrometric = observer_pos.at(t_skyfield).observe(sun)
    apparent = astrometric.apparent()
    alt, _, _ = apparent.altaz()
    sun_alt = alt.degrees
    return -18.0 <= sun_alt <= -6.0


# --- 衛星パス計算 ---
def compute_satellite_pass(
    sat: EarthSatellite, t_skyfield
) -> dict | None:
    """
    指定時刻における衛星の高度・方位角を計算する。
    高度がMIN_ALT_DEG以上の場合のみ結果を返す。
    """
    diff = sat - OBSERVER
    topocentric = diff.at(t_skyfield)
    alt, az, dist = topocentric.altaz()
    if alt.degrees >= MIN_ALT_DEG:
        return {
            "alt": alt.degrees,
            "az": az.degrees,
            "dist_km": dist.km,
        }
    return None


# --- トレイン検出 ---
def find_train_passes(
    sats_tle: list[tuple[str, str, str]],
    obs_start: datetime,
    obs_end: datetime,
    scan_interval_min: int = 1,
) -> list[dict]:
    """
    観測時間帯をスキャンし、高度条件を満たす衛星パスを収集する。
    各パスには衛星名・時刻・高度・方位角を含む。
    """
    passes = []
    current = obs_start
    time_steps = []
    while current <= obs_end:
        time_steps.append(current)
        current += timedelta(minutes=scan_interval_min)

    for name, l1, l2 in sats_tle:
        try:
            sat = EarthSatellite(l1, l2, name, ts)
        except Exception:
            continue

        for t in time_steps:
            t_sf = ts.from_datetime(t)
            if not is_observable_twilight(t_sf):
                continue
            result = compute_satellite_pass(sat, t_sf)
            if result:
                passes.append({
                    "name": name,
                    "time": t,
                    "alt": result["alt"],
                    "az": result["az"],
                    "dist_km": result["dist_km"],
                })
                break  # 1衛星につき最初の可視パスのみ

    return passes


def cluster_into_trains(passes: list[dict]) -> list[list[dict]]:
    """
    時間的・空間的に近い衛星パスをクラスタリングしてトレインを検出する。
    TRAIN_TIME_WINDOW_SEC以内かつTRAIN_AZ_TOLERANCE_DEG以内のパスをグループ化。
    """
    if not passes:
        return []

    sorted_passes = sorted(passes, key=lambda p: p["time"])
    clusters = []
    current_cluster = [sorted_passes[0]]

    for p in sorted_passes[1:]:
        last = current_cluster[-1]
        dt = (p["time"] - last["time"]).total_seconds()
        daz = abs(p["az"] - last["az"])
        if daz > 180:
            daz = 360 - daz

        if dt <= TRAIN_TIME_WINDOW_SEC and daz <= TRAIN_AZ_TOLERANCE_DEG:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]

    clusters.append(current_cluster)
    return [c for c in clusters if len(c) >= TRAIN_CLUSTER_THRESHOLD]


def select_best_train(trains: list[list[dict]]) -> dict | None:
    """
    最も衛星数が多いトレインを選び、ベストパス情報を返す。
    同数の場合は平均高度が高い方を優先する。
    """
    if not trains:
        return None

    best_train = max(
        trains,
        key=lambda t: (len(t), sum(p["alt"] for p in t) / len(t)),
    )

    avg_time = best_train[len(best_train) // 2]["time"]
    azimuths = [p["az"] for p in best_train]
    start_az = azimuths[0]
    end_az = azimuths[-1]
    avg_alt = sum(p["alt"] for p in best_train) / len(best_train)

    return {
        "time": avg_time,
        "start_az": start_az,
        "end_az": end_az,
        "start_dir": az_to_direction(start_az),
        "end_dir": az_to_direction(end_az),
        "start_context": az_to_context(start_az),
        "end_context": az_to_context(end_az),
        "avg_alt": avg_alt,
        "sat_count": len(best_train),
        "time_str": avg_time.strftime("%H:%M"),
    }


# --- メインAPI ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """トップページ: 今夜のスターリンク可視判定を表示する。"""
    now = datetime.now(tz=JST)
    today = now.date()

    obs_start = datetime(today.year, today.month, today.day,
                         OBS_START_HOUR, 0, tzinfo=JST)
    obs_end = datetime(today.year, today.month, today.day,
                       OBS_END_HOUR, 0, tzinfo=JST)

    # 観測時間帯を過ぎていたら翌日を対象にする
    if now > obs_end:
        obs_start += timedelta(days=1)
        obs_end += timedelta(days=1)

    result = None
    error_msg = None

    try:
        sats_tle = await fetch_tle_data()
        # 最新600衛星を優先（トレインは打ち上げ直後に見える）
        recent_sats = sats_tle[-600:]
        passes = find_train_passes(recent_sats, obs_start, obs_end)
        trains = cluster_into_trains(passes)
        result = select_best_train(trains)
    except httpx.HTTPError as e:
        logger.error("TLE取得エラー: %s", e)
        error_msg = "衛星データの取得に失敗しました"
    except Exception as e:
        logger.error("計算エラー: %s", e)
        error_msg = "計算中にエラーが発生しました"

    target_date = obs_start.strftime("%m/%d")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "error_msg": error_msg,
        "target_date": target_date,
    })


@app.get("/api/tonight")
async def api_tonight():
    """API: 今夜の可視判定をJSONで返す。"""
    now = datetime.now(tz=JST)
    today = now.date()

    obs_start = datetime(today.year, today.month, today.day,
                         OBS_START_HOUR, 0, tzinfo=JST)
    obs_end = datetime(today.year, today.month, today.day,
                       OBS_END_HOUR, 0, tzinfo=JST)

    if now > obs_end:
        obs_start += timedelta(days=1)
        obs_end += timedelta(days=1)

    try:
        sats_tle = await fetch_tle_data()
        recent_sats = sats_tle[-600:]
        passes = find_train_passes(recent_sats, obs_start, obs_end)
        trains = cluster_into_trains(passes)
        result = select_best_train(trains)
    except Exception as e:
        return {"visible": False, "error": str(e)}

    if result:
        return {
            "visible": True,
            "time": result["time_str"],
            "direction": f"{result['start_dir']}（{result['start_context']}）→ {result['end_dir']}",
            "altitude": round(result["avg_alt"], 1),
            "satellite_count": result["sat_count"],
            "date": obs_start.strftime("%Y-%m-%d"),
        }
    return {"visible": False, "date": obs_start.strftime("%Y-%m-%d")}


# --- Static配信 & PWA ---
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(
    directory=str(Path(__file__).parent / "static")
), name="static")


@app.get("/manifest.json")
async def manifest():
    """PWAマニフェストを返す。"""
    manifest_path = Path(__file__).parent / "static" / "manifest.json"
    return HTMLResponse(
        content=manifest_path.read_text(encoding="utf-8"),
        media_type="application/manifest+json",
    )


@app.get("/sw.js")
async def service_worker():
    """Service Workerを返す。キャッシュ戦略: Network First。"""
    sw_code = """
const CACHE_NAME = 'starlink-nishi-v1';
const URLS_TO_CACHE = ['/'];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(URLS_TO_CACHE))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    fetch(event.request)
      .then(response => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
"""
    return HTMLResponse(content=sw_code.strip(), media_type="application/javascript")
