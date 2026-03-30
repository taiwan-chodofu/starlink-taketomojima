"""
竹富島・西桟橋 スターリンク トレイン ビューワー

観測地点: 西桟橋（竹富島） 24.3237°N, 124.0893°E
対象時間: 18:00〜21:00 JST
条件: 高度30°以上、薄明中（太陽 -6°〜-18°）
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from skyfield.api import load, EarthSatellite, wgs84
from datetime import datetime, timedelta, timezone
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Starlink Nishi-Sanbashi")

# --- 定数 ---
LAT = 24.3237
LON = 124.0893
OBSERVER = wgs84.latlon(LAT, LON)
JST = timezone(timedelta(hours=9))
TLE_URLS = [
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/supplemental/starlink.txt",
    "https://www.celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    "https://tle.ivanstanojevic.me/api/tle/?search=starlink&page_size=100&format=text",
]

TLE_CACHE_MINUTES = 120
MIN_ALT_DEG = 30.0
OBS_START_HOUR = 18
OBS_END_HOUR = 21
TRAIN_CLUSTER_THRESHOLD = 3
TRAIN_TIME_WINDOW_SEC = 300
TRAIN_AZ_TOLERANCE_DEG = 30.0

ts = load.timescale()
eph = load('de421.bsp')
_tle_cache = {"data": None, "fetched_at": None}

# --- 方角 ---
DIRECTION_NAMES = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"]
DIRECTION_CONTEXT = {
    "北": "島の奥側", "北東": "島の上空", "東": "石垣島の方向",
    "南東": "石垣島寄りの海", "南": "海の方向", "南西": "海の方向（夕日側）",
    "西": "夕日の方向", "北西": "島の奥・海寄り",
}


def az_to_direction(az_deg: float) -> str:
    return DIRECTION_NAMES[int((az_deg + 22.5) // 45) % 8]


def az_to_context(az_deg: float) -> str:
    return DIRECTION_CONTEXT.get(az_to_direction(az_deg), "")


# --- TLE取得 ---
async def fetch_tle_data() -> list[tuple[str, str, str]]:
    now = datetime.now(tz=JST)
    if (
        _tle_cache["data"] is not None
        and _tle_cache["fetched_at"] is not None
        and (now - _tle_cache["fetched_at"]).total_seconds() < TLE_CACHE_MINUTES * 60
    ):
        return _tle_cache["data"]

    headers = {"User-Agent": "StarlinkNishi/1.0 (satellite-viewer)"}
    last_error = None
    async with httpx.AsyncClient(timeout=60.0, headers=headers) as client:
        for url in TLE_URLS:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                break
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_error = e
                continue
        else:
            raise last_error or RuntimeError("TLE取得失敗")

    lines = resp.text.strip().splitlines()
    sats = []
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i].strip(), lines[i+1].strip(), lines[i+2].strip()
        if l1.startswith("1 ") and l2.startswith("2 "):
            sats.append((name, l1, l2))

    _tle_cache["data"] = sats
    _tle_cache["fetched_at"] = now
    return sats


# --- 薄明・パス計算 ---
def is_observable_twilight(t_sf) -> bool:
    sun = eph['sun']
    earth = eph['earth']
    alt, _, _ = (earth + OBSERVER).at(t_sf).observe(sun).apparent().altaz()
    return -18.0 <= alt.degrees <= -6.0


def compute_pass(sat: EarthSatellite, t_sf) -> dict | None:
    alt, az, dist = (sat - OBSERVER).at(t_sf).altaz()
    if alt.degrees >= MIN_ALT_DEG:
        return {"alt": alt.degrees, "az": az.degrees, "dist_km": dist.km}
    return None


def find_train_passes(sats_tle, obs_start, obs_end):
    passes = []
    time_steps = []
    current = obs_start
    while current <= obs_end:
        time_steps.append(current)
        current += timedelta(minutes=1)

    for name, l1, l2 in sats_tle:
        try:
            sat = EarthSatellite(l1, l2, name, ts)
        except Exception:
            continue
        for t in time_steps:
            t_sf = ts.from_datetime(t)
            if not is_observable_twilight(t_sf):
                continue
            result = compute_pass(sat, t_sf)
            if result:
                passes.append({"name": name, "time": t, **result})
                break
    return passes


def cluster_into_trains(passes):
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


def select_best_train(trains):
    if not trains:
        return None
    best = max(trains, key=lambda t: (len(t), sum(p["alt"] for p in t) / len(t)))
    mid = best[len(best) // 2]
    return {
        "time_str": mid["time"].strftime("%H:%M"),
        "start_dir": az_to_direction(best[0]["az"]),
        "end_dir": az_to_direction(best[-1]["az"]),
        "start_context": az_to_context(best[0]["az"]),
        "sat_count": len(best),
    }


# --- HTML（テンプレートファイル不要） ---
def render_html(target_date: str, result=None, error_msg=None) -> str:
    if error_msg:
        content = f'<div class="error"><div class="status">{error_msg}</div></div>'
    elif result:
        content = f"""<div class="visible">
  <div class="status">スターリンク 見える</div>
  <div class="time">{result['time_str']}</div>
  <div class="direction">{result['start_dir']}
    <span class="context">（{result['start_context']}）</span>を見る</div>
  <div class="arrow">↓</div>
  <div class="direction">{result['end_dir']} へ流れる</div>
  <div class="count">約{result['sat_count']}機の連なり</div>
</div>"""
    else:
        content = '<div class="not-visible"><div class="status">今夜は<br>見えなそうです</div></div>'

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#000000">
<title>西桟橋 — Starlink</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans JP",sans-serif;
background:#000;color:#fff;min-height:100vh;min-height:100dvh;display:flex;flex-direction:column;
align-items:center;justify-content:center;padding:2rem 1.5rem;text-align:center;
-webkit-font-smoothing:antialiased}}
.place{{font-size:.85rem;letter-spacing:.3em;color:rgba(255,255,255,.4);margin-bottom:3rem}}
.date{{font-size:.9rem;color:rgba(255,255,255,.35);margin-bottom:2rem}}
.visible .status{{font-size:1.1rem;color:rgba(255,255,255,.6);margin-bottom:2.5rem;letter-spacing:.15em}}
.visible .time{{font-size:4rem;font-weight:200;letter-spacing:.05em;margin-bottom:2.5rem;line-height:1}}
.visible .direction{{font-size:1.3rem;line-height:2;color:rgba(255,255,255,.85)}}
.visible .direction .context{{font-size:.85rem;color:rgba(255,255,255,.4)}}
.visible .arrow{{font-size:1.8rem;color:rgba(255,255,255,.3);margin:.8rem 0}}
.visible .count{{font-size:.8rem;color:rgba(255,255,255,.25);margin-top:3rem}}
.not-visible .status{{font-size:1.3rem;color:rgba(255,255,255,.5);line-height:1.8}}
.error .status{{font-size:1rem;color:rgba(255,100,100,.6)}}
.footer{{position:fixed;bottom:1.5rem;font-size:.7rem;color:rgba(255,255,255,.15);letter-spacing:.1em}}
@media(min-width:600px){{.visible .time{{font-size:5.5rem}}.visible .direction{{font-size:1.5rem}}}}
</style>
</head>
<body>
<div class="place">竹富島・西桟橋</div>
<div class="date">{target_date}</div>
{content}
<div class="footer">Starlink Train Viewer</div>
</body>
</html>"""


# --- エンドポイント ---
@app.get("/", response_class=HTMLResponse)
async def home():
    now = datetime.now(tz=JST)
    today = now.date()
    obs_start = datetime(today.year, today.month, today.day, OBS_START_HOUR, 0, tzinfo=JST)
    obs_end = datetime(today.year, today.month, today.day, OBS_END_HOUR, 0, tzinfo=JST)
    if now > obs_end:
        obs_start += timedelta(days=1)
        obs_end += timedelta(days=1)

    result, error_msg = None, None
    try:
        sats = await fetch_tle_data()
        passes = find_train_passes(sats[-600:], obs_start, obs_end)
        trains = cluster_into_trains(passes)
        result = select_best_train(trains)
    except Exception as e:
        logger.error("エラー: %s", e)
        error_msg = "衛星データの取得に失敗しました"

    return render_html(obs_start.strftime("%m/%d"), result, error_msg)


@app.get("/api/tonight")
async def api_tonight():
    now = datetime.now(tz=JST)
    today = now.date()
    obs_start = datetime(today.year, today.month, today.day, OBS_START_HOUR, 0, tzinfo=JST)
    obs_end = datetime(today.year, today.month, today.day, OBS_END_HOUR, 0, tzinfo=JST)
    if now > obs_end:
        obs_start += timedelta(days=1)
        obs_end += timedelta(days=1)
    try:
        sats = await fetch_tle_data()
        passes = find_train_passes(sats[-600:], obs_start, obs_end)
        trains = cluster_into_trains(passes)
        result = select_best_train(trains)
    except Exception as e:
        return {"visible": False, "error": str(e)}
    if result:
        return {"visible": True, "time": result["time_str"],
                "direction": f"{result['start_dir']}（{result['start_context']}）→ {result['end_dir']}",
                "satellite_count": result["sat_count"]}
    return {"visible": False}
