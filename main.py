"""
竹富島・西桟橋 スターリンク トレイン ビューワー

観測地点: 西桟橋（竹富島） 24.3237°N, 124.0893°E
対象時間: 18:00〜21:00 JST
条件: 高度30°以上、薄明中（太陽 -6°〜-18°）

「空を見上げるきっかけ装置」として、場所・時間・方角だけを静かに提示する。
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from skyfield.api import load, EarthSatellite, wgs84
from skyfield import almanac
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from pathlib import Path
import httpx
import logging
import json
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Lifespan（起動時TLEプリフェッチ） ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリ起動時にTLEデータを事前取得する。"""
    try:
        sats = await fetch_tle_data()
        logger.info("起動時TLEプリフェッチ完了: %d衛星", len(sats))
    except Exception as e:
        logger.warning("起動時TLEプリフェッチ失敗: %s", e)
    yield


app = FastAPI(title="Starlink Nishi-Sanbashi", lifespan=lifespan)

# --- テンプレート & Static ---
_here = Path(__file__).parent
_templates_dir = _here / "templates"
_static_dir = _here / "static"
templates = Jinja2Templates(directory=str(_templates_dir))
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# --- 観測地・時間帯 ---
LAT = 24.3237
LON = 124.0893
OBSERVER = wgs84.latlon(LAT, LON)
JST = timezone(timedelta(hours=9))

# --- TLEソース（多重フォールバック） ---
TLE_URLS = [
    "https://tle.ivanstanojevic.me/api/tle/?search=starlink&page-size=100&sort=popularity&sort-dir=desc",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/supplemental/starlink.txt",
]
TLE_CACHE_FILE = _here / "tle_cache.json"
TLE_MEM_CACHE_MINUTES = 120
TLE_FILE_CACHE_HOURS = 24

# --- 可視判定パラメータ ---
MIN_ALT_DEG = 30.0
OBS_START_HOUR = 18
OBS_END_HOUR = 21
TRAIN_CLUSTER_THRESHOLD = 3
TRAIN_TIME_WINDOW_SEC = 300
TRAIN_AZ_TOLERANCE_DEG = 30.0
SCAN_INTERVAL_MIN = 5
NEXT_SCAN_INTERVAL_MIN = 10
MAX_SATS = 200

# --- Skyfield リソース ---
ts = load.timescale()
eph = load('de421.bsp')
_tle_cache: dict = {"data": None, "fetched_at": None}

# --- 方角の言葉 ---
DIRECTION_NAMES = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"]
DIRECTION_CONTEXT = {
    "北":   "島の奥側",
    "北東": "島の上空",
    "東":   "石垣島の方向",
    "南東": "石垣島寄りの海",
    "南":   "海の方向",
    "南西": "海の方向（夕日側）",
    "西":   "夕日の方向",
    "北西": "島の奥・海寄り",
}


def az_to_direction(az_deg: float) -> str:
    return DIRECTION_NAMES[int((az_deg + 22.5) // 45) % 8]


def az_to_context(az_deg: float) -> str:
    return DIRECTION_CONTEXT.get(az_to_direction(az_deg), "")


# --- TLE取得（メモリ → ネットワーク → ファイルキャッシュ） ---
def _load_file_cache() -> list[tuple[str, str, str]] | None:
    if not TLE_CACHE_FILE.exists():
        return None
    try:
        data = json.loads(TLE_CACHE_FILE.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(data["fetched_at"])
        age_hours = (datetime.now(tz=JST) - cached_at).total_seconds() / 3600
        logger.info("ファイルキャッシュ: %d衛星, %.1f時間前", len(data["sats"]), age_hours)
        if age_hours <= TLE_FILE_CACHE_HOURS:
            return [tuple(s) for s in data["sats"]]
        logger.info("ファイルキャッシュ期限切れ")
    except Exception as e:
        logger.warning("ファイルキャッシュ読み込み失敗: %s", e)
    return None


def _save_file_cache(sats: list[tuple[str, str, str]]) -> None:
    try:
        data = {"fetched_at": datetime.now(tz=JST).isoformat(), "sats": sats}
        TLE_CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info("ファイルキャッシュ保存: %d衛星", len(sats))
    except Exception as e:
        logger.warning("ファイルキャッシュ保存失敗: %s", e)


def _parse_tle_response(text: str, url: str) -> list[tuple[str, str, str]]:
    """TLE/JSONレスポンスをパースする。"""
    sats: list[tuple[str, str, str]] = []

    if "ivanstanojevic" in url or text.strip().startswith("{"):
        try:
            data = json.loads(text)
            for m in data.get("member", []):
                name = m.get("name", "")
                l1 = m.get("line1", "")
                l2 = m.get("line2", "")
                if l1.startswith("1 ") and l2.startswith("2 "):
                    sats.append((name, l1, l2))
        except json.JSONDecodeError:
            pass

    if not sats:
        lines = text.strip().splitlines()
        for i in range(0, len(lines) - 2, 3):
            name = lines[i].strip()
            l1, l2 = lines[i + 1].strip(), lines[i + 2].strip()
            if l1.startswith("1 ") and l2.startswith("2 "):
                sats.append((name, l1, l2))
    return sats


async def fetch_tle_data() -> list[tuple[str, str, str]]:
    """TLEを取得する。メモリキャッシュ → ネットワーク → ファイルキャッシュの順。"""
    now = datetime.now(tz=JST)

    if (
        _tle_cache["data"] is not None
        and _tle_cache["fetched_at"] is not None
        and (now - _tle_cache["fetched_at"]).total_seconds() < TLE_MEM_CACHE_MINUTES * 60
    ):
        return _tle_cache["data"]

    headers = {"User-Agent": "Mozilla/5.0 (compatible; StarlinkNishi/2.0)"}
    async with httpx.AsyncClient(timeout=60.0, headers=headers, follow_redirects=True) as client:
        for url in TLE_URLS:
            try:
                logger.info("TLE取得試行: %s", url)
                resp = await client.get(url)
                resp.raise_for_status()
                sats = _parse_tle_response(resp.text, url)
                if sats:
                    _tle_cache["data"] = sats
                    _tle_cache["fetched_at"] = now
                    _save_file_cache(sats)
                    logger.info("TLE取得成功: %d衛星", len(sats))
                    return sats
            except Exception as e:
                logger.warning("TLE取得失敗(%s): %s", url, e)

    logger.info("ネットワーク失敗、ファイルキャッシュを試行")
    cached = _load_file_cache()
    if cached:
        _tle_cache["data"] = cached
        _tle_cache["fetched_at"] = now
        return cached

    raise RuntimeError("TLEデータを取得できません")


# --- 薄明・パス計算 ---
def is_observable_twilight(t_sf) -> bool:
    """太陽高度が -18°〜-6°（天文薄明〜市民薄明）かを判定する。"""
    alt, _, _ = (eph['earth'] + OBSERVER).at(t_sf).observe(eph['sun']).apparent().altaz()
    return -18.0 <= alt.degrees <= -6.0


def compute_pass(sat: EarthSatellite, t_sf) -> dict | None:
    alt, az, dist = (sat - OBSERVER).at(t_sf).altaz()
    if alt.degrees >= MIN_ALT_DEG:
        return {"alt": alt.degrees, "az": az.degrees, "dist_km": dist.km}
    return None


def find_train_passes(
    sats_tle: list[tuple[str, str, str]],
    obs_start: datetime,
    obs_end: datetime,
    interval_min: int | None = None,
) -> list[dict]:
    """観測時間帯をスキャンし、高度条件を満たす衛星パスを収集する。"""
    if interval_min is None:
        interval_min = SCAN_INTERVAL_MIN

    time_steps: list[datetime] = []
    current = obs_start
    while current <= obs_end:
        time_steps.append(current)
        current += timedelta(minutes=interval_min)

    twilight_ok = {t: is_observable_twilight(ts.from_datetime(t)) for t in time_steps}

    passes: list[dict] = []
    for name, l1, l2 in sats_tle:
        try:
            sat = EarthSatellite(l1, l2, name, ts)
        except Exception:
            continue
        for t in time_steps:
            if not twilight_ok[t]:
                continue
            result = compute_pass(sat, ts.from_datetime(t))
            if result:
                passes.append({"name": name, "time": t, **result})
                break
    return passes


def find_train_passes_relaxed(sats_tle, obs_start, obs_end):
    """翌日以降の予測用: 高度20°、10分間隔の軽量スキャン。"""
    min_alt = 20.0
    time_steps: list[datetime] = []
    current = obs_start
    while current <= obs_end:
        time_steps.append(current)
        current += timedelta(minutes=NEXT_SCAN_INTERVAL_MIN)

    twilight_ok = {t: is_observable_twilight(ts.from_datetime(t)) for t in time_steps}

    passes: list[dict] = []
    for name, l1, l2 in sats_tle:
        try:
            sat = EarthSatellite(l1, l2, name, ts)
        except Exception:
            continue
        for t in time_steps:
            if not twilight_ok[t]:
                continue
            alt, az, dist = (sat - OBSERVER).at(ts.from_datetime(t)).altaz()
            if alt.degrees >= min_alt:
                passes.append({
                    "name": name, "time": t,
                    "alt": alt.degrees, "az": az.degrees, "dist_km": dist.km,
                })
                break
    return passes


def cluster_into_trains(passes: list[dict]) -> list[list[dict]]:
    """時間的・空間的に近い衛星パスをグループ化する。"""
    if not passes:
        return []
    sorted_p = sorted(passes, key=lambda p: p["time"])
    clusters, current = [], [sorted_p[0]]
    for p in sorted_p[1:]:
        last = current[-1]
        dt = (p["time"] - last["time"]).total_seconds()
        daz = abs(p["az"] - last["az"])
        if daz > 180:
            daz = 360 - daz
        if dt <= TRAIN_TIME_WINDOW_SEC and daz <= TRAIN_AZ_TOLERANCE_DEG:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)
    return [c for c in clusters if len(c) >= TRAIN_CLUSTER_THRESHOLD]


def select_best_train(trains: list[list[dict]]) -> dict | None:
    """最も衛星数が多く、平均高度が高いトレインを選ぶ。"""
    if not trains:
        return None
    best = max(trains, key=lambda t: (len(t), sum(p["alt"] for p in t) / len(t)))
    mid = best[len(best) // 2]
    return {
        "time_str": mid["time"].strftime("%H:%M"),
        "time_iso": mid["time"].isoformat(),
        "start_dir": az_to_direction(best[0]["az"]),
        "end_dir": az_to_direction(best[-1]["az"]),
        "start_context": az_to_context(best[0]["az"]),
        "sat_count": len(best),
    }


# --- 次回候補・夜空条件 ---
def _format_month_day(d) -> str:
    """月/日をゼロ埋めなしで返す（Windows/Linux共通）。"""
    return f"{d.month}/{d.day}"


def find_next_visible(sats_tle, start_date, max_days: int = 7) -> dict | None:
    """max_days日分をスキャンし、最初に可視パスが見つかった日を返す。"""
    for day_offset in range(max_days):
        target = start_date + timedelta(days=day_offset)
        obs_start = datetime(target.year, target.month, target.day,
                             OBS_START_HOUR, 0, tzinfo=JST)
        obs_end = datetime(target.year, target.month, target.day,
                           OBS_END_HOUR, 0, tzinfo=JST)
        passes = find_train_passes_relaxed(sats_tle, obs_start, obs_end)
        trains = cluster_into_trains(passes)
        best = select_best_train(trains)
        if best:
            return {"date": _format_month_day(target), **best}
    return None


def find_next_dark_sky(start_date, max_days: int = 30) -> dict | None:
    """衛星が見つからない時のフォールバック: 次の新月期（輝面比<15%）を探す。"""
    for day_offset in range(max_days):
        target = start_date + timedelta(days=day_offset)
        t_obs = ts.from_datetime(
            datetime(target.year, target.month, target.day, 20, 0, tzinfo=JST)
        )
        phase_angle = almanac.moon_phase(eph, t_obs).degrees
        illumination = (1 - math.cos(math.radians(phase_angle))) / 2 * 100
        if illumination < 15:
            return {
                "date": _format_month_day(target),
                "illumination": round(illumination, 0),
            }
    return None


# --- 月・潮 ---
MOON_PHASE_NAMES = [
    "🌑 新月", "🌒 三日月", "🌓 上弦", "🌔 十三夜",
    "🌕 満月", "🌖 十八夜", "🌗 下弦", "🌘 二十六夜",
]
TIDE_TYPES = {
    "大潮": [0, 1, 14, 15, 29],
    "中潮": [2, 3, 12, 13, 16, 17, 27, 28],
    "小潮": [4, 5, 11, 18, 19, 26],
    "長潮": [6, 20],
    "若潮": [7, 21],
}


def get_moon_info(target_date) -> dict:
    """月齢・月相・潮の種類・夜空条件を計算する。"""
    t_obs = ts.from_datetime(
        datetime(target_date.year, target_date.month, target_date.day,
                 20, 0, tzinfo=JST)
    )
    phase_angle = almanac.moon_phase(eph, t_obs).degrees
    phase_name = MOON_PHASE_NAMES[int(phase_angle / 45) % 8]
    moon_age = round(phase_angle / 360 * 29.53059, 1)

    moon_age_int = int(moon_age + 0.5) % 30
    tide_type = "中潮"
    for t_name, days in TIDE_TYPES.items():
        if moon_age_int in days:
            tide_type = t_name
            break

    moon_alt, _, _ = (eph['earth'] + OBSERVER).at(t_obs).observe(eph['moon']).apparent().altaz()
    illumination = round((1 - math.cos(math.radians(phase_angle))) / 2 * 100, 0)
    moon_is_bright = moon_alt.degrees > 0 and illumination > 30

    if illumination < 15:
        sky_note = "新月期 — 星空の条件◎"
    elif not moon_is_bright:
        sky_note = "月は沈んでいます — 星空◎"
    elif illumination > 70:
        sky_note = "月が明るい — 星は見えにくい"
    else:
        sky_note = "月明かりあり"

    return {
        "phase": phase_name,
        "age": moon_age,
        "tide_type": tide_type,
        "moon_alt": round(moon_alt.degrees, 1),
        "illumination": illumination,
        "sky_note": sky_note,
    }


# --- エンドポイント ---
def _compute_observation_window() -> tuple[datetime, datetime, bool]:
    """今夜の観測窓を返す。既に過ぎていれば翌日にシフト。"""
    now = datetime.now(tz=JST)
    today = now.date()
    obs_start = datetime(today.year, today.month, today.day,
                         OBS_START_HOUR, 0, tzinfo=JST)
    obs_end = datetime(today.year, today.month, today.day,
                       OBS_END_HOUR, 0, tzinfo=JST)
    shifted = False
    if now > obs_end:
        obs_start += timedelta(days=1)
        obs_end += timedelta(days=1)
        shifted = True
    return obs_start, obs_end, shifted


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """ローディング画面を即返し、データはJSで非同期取得する。"""
    obs_start, _, _ = _compute_observation_window()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "target_date": obs_start.strftime("%m/%d"),
    })


@app.get("/api/tonight")
async def api_tonight():
    """今夜の判定をJSONで返す。"""
    obs_start, obs_end, _ = _compute_observation_window()
    try:
        sats = await fetch_tle_data()
        recent = sats[-MAX_SATS:] if len(sats) > MAX_SATS else sats
        passes = find_train_passes(recent, obs_start, obs_end)
        trains = cluster_into_trains(passes)
        result = select_best_train(trains)

        next_visible = None
        next_dark_sky = None
        if not result:
            tomorrow = obs_start.date() + timedelta(days=1)
            next_visible = find_next_visible(recent, tomorrow, max_days=7)
            if not next_visible:
                next_dark_sky = find_next_dark_sky(tomorrow, max_days=30)

        moon = get_moon_info(obs_start.date())
    except Exception as e:
        logger.error("判定エラー: %s", e)
        return JSONResponse({"error": "衛星データの取得に失敗しました"})

    return {
        "visible": result is not None,
        "result": result,
        "next_visible": next_visible,
        "next_dark_sky": next_dark_sky,
        "moon": moon,
        "obs_window": {
            "start": obs_start.isoformat(),
            "end": obs_end.isoformat(),
        },
    }


# --- PWA ---
@app.get("/manifest.json")
async def manifest():
    p = _static_dir / "manifest.json"
    if not p.exists():
        return JSONResponse({})
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/sw.js")
async def service_worker():
    sw = """
const CACHE_NAME = 'starlink-nishi-v2';
const URLS = ['/'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(URLS)));
});
self.addEventListener('fetch', e => {
  e.respondWith(
    fetch(e.request)
      .then(r => {
        const clone = r.clone();
        caches.open(CACHE_NAME).then(c => c.put(e.request, clone));
        return r;
      })
      .catch(() => caches.match(e.request))
  );
});
""".strip()
    return HTMLResponse(content=sw, media_type="application/javascript")



