"""
竹富島・西桟橋 スターリンク トレイン ビューワー
v4: ローディング画面 + 非同期取得 + 計算軽量化 + 月判定改善
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from skyfield.api import load, EarthSatellite, wgs84
from skyfield import almanac
from datetime import datetime, timedelta, timezone
from pathlib import Path
import httpx
import logging
import json
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Starlink Nishi-Sanbashi")

# --- 定数 ---
LAT = 24.3237
LON = 124.0893
OBSERVER = wgs84.latlon(LAT, LON)
JST = timezone(timedelta(hours=9))
TLE_URLS = [
    "https://tle.ivanstanojevic.me/api/tle/?search=starlink&page-size=100&sort=popularity&sort-dir=desc",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/supplemental/starlink.txt",
]
TLE_CACHE_FILE = Path(__file__).parent / "tle_cache.json"
TLE_CACHE_MINUTES = 120
MIN_ALT_DEG = 30.0
OBS_START_HOUR = 18
OBS_END_HOUR = 21
TRAIN_CLUSTER_THRESHOLD = 3
TRAIN_TIME_WINDOW_SEC = 300
TRAIN_AZ_TOLERANCE_DEG = 30.0
SCAN_INTERVAL_MIN = 5          # メインスキャン間隔（分）
NEXT_SCAN_INTERVAL_MIN = 10    # 翌日以降スキャン間隔（分）
MAX_SATS = 200                 # 計算対象の衛星数上限

ts = load.timescale()
eph = load('de421.bsp')
_tle_cache = {"data": None, "fetched_at": None}

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


# --- TLE取得（ファイルキャッシュ付き） ---
def _load_file_cache() -> list[tuple[str, str, str]] | None:
    """ファイルキャッシュからTLEデータを読み込む。"""
    if not TLE_CACHE_FILE.exists():
        return None
    try:
        data = json.loads(TLE_CACHE_FILE.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(data["fetched_at"])
        age_hours = (datetime.now(tz=JST) - cached_at).total_seconds() / 3600
        logger.info("ファイルキャッシュ発見: %d衛星, %.1f時間前", len(data["sats"]), age_hours)
        # 24時間以内なら使用可能（TLEは1日程度は有効）
        if age_hours <= 24:
            return [tuple(s) for s in data["sats"]]
        logger.info("ファイルキャッシュ期限切れ（%.1f時間）", age_hours)
    except Exception as e:
        logger.warning("ファイルキャッシュ読み込み失敗: %s", e)
    return None


def _save_file_cache(sats: list[tuple[str, str, str]]) -> None:
    """TLEデータをファイルキャッシュに保存する。"""
    try:
        data = {
            "fetched_at": datetime.now(tz=JST).isoformat(),
            "sats": sats,
        }
        TLE_CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info("ファイルキャッシュ保存: %d衛星", len(sats))
    except Exception as e:
        logger.warning("ファイルキャッシュ保存失敗: %s", e)


def _parse_tle_response(text: str, url: str) -> list[tuple[str, str, str]]:
    """TLEレスポンスをパースする。JSON（TLE API）と3行TLE（Celestrak）の両方に対応。"""
    sats = []
    # JSON形式（tle.ivanstanojevic.me）
    if "ivanstanojevic" in url or text.strip().startswith("{"):
        try:
            data = json.loads(text)
            for member in data.get("member", []):
                name = member.get("name", "")
                l1 = member.get("line1", "")
                l2 = member.get("line2", "")
                if l1.startswith("1 ") and l2.startswith("2 "):
                    sats.append((name, l1, l2))
        except json.JSONDecodeError:
            pass
    # 3行TLE形式（Celestrak）
    if not sats:
        lines = text.strip().splitlines()
        for i in range(0, len(lines) - 2, 3):
            name = lines[i].strip()
            l1, l2 = lines[i+1].strip(), lines[i+2].strip()
            if l1.startswith("1 ") and l2.startswith("2 "):
                sats.append((name, l1, l2))
    return sats


async def fetch_tle_data() -> list[tuple[str, str, str]]:
    """TLEデータを取得する。メモリキャッシュ → ネットワーク → ファイルキャッシュの順。"""
    now = datetime.now(tz=JST)

    # 1. メモリキャッシュ
    if (
        _tle_cache["data"] is not None
        and _tle_cache["fetched_at"] is not None
        and (now - _tle_cache["fetched_at"]).total_seconds() < TLE_CACHE_MINUTES * 60
    ):
        return _tle_cache["data"]

    # 2. ネットワーク取得（タイムアウト60秒）
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StarlinkNishi/1.0)"}
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
                    logger.info("TLE取得成功: %d衛星 from %s", len(sats), url)
                    return sats
            except Exception as e:
                logger.warning("TLE取得失敗(%s): %s", url, e)

    # 3. ファイルキャッシュにフォールバック
    logger.info("ネットワーク取得失敗、ファイルキャッシュを試行")
    cached = _load_file_cache()
    if cached:
        _tle_cache["data"] = cached
        _tle_cache["fetched_at"] = now
        return cached

    raise RuntimeError("TLEデータを取得できません")


# --- 薄明・パス計算 ---
def is_observable_twilight(t_sf) -> bool:
    alt, _, _ = (eph['earth'] + OBSERVER).at(t_sf).observe(eph['sun']).apparent().altaz()
    return -18.0 <= alt.degrees <= -6.0


def compute_pass(sat: EarthSatellite, t_sf) -> dict | None:
    alt, az, dist = (sat - OBSERVER).at(t_sf).altaz()
    if alt.degrees >= MIN_ALT_DEG:
        return {"alt": alt.degrees, "az": az.degrees, "dist_km": dist.km}
    return None


def find_train_passes(sats_tle, obs_start, obs_end, interval_min=None):
    if interval_min is None:
        interval_min = SCAN_INTERVAL_MIN
    passes, time_steps, current = [], [], obs_start
    while current <= obs_end:
        time_steps.append(current)
        current += timedelta(minutes=interval_min)
    # 薄明チェックを事前計算してキャッシュ
    twilight_ok = {}
    for t in time_steps:
        t_sf = ts.from_datetime(t)
        twilight_ok[t] = is_observable_twilight(t_sf)
    for name, l1, l2 in sats_tle:
        try:
            sat = EarthSatellite(l1, l2, name, ts)
        except Exception:
            continue
        for t in time_steps:
            if not twilight_ok[t]:
                continue
            t_sf = ts.from_datetime(t)
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


def find_next_visible(sats_tle, start_date, max_days=3) -> dict | None:
    """今夜以降max_days日分をスキャンし、最初に可視パスが見つかった日の情報を返す。"""
    for day_offset in range(max_days):
        target = start_date + timedelta(days=day_offset)
        obs_start = datetime(target.year, target.month, target.day,
                             OBS_START_HOUR, 0, tzinfo=JST)
        obs_end = datetime(target.year, target.month, target.day,
                           OBS_END_HOUR, 0, tzinfo=JST)
        passes = find_train_passes(sats_tle, obs_start, obs_end,
                                   interval_min=NEXT_SCAN_INTERVAL_MIN)
        trains = cluster_into_trains(passes)
        best = select_best_train(trains)
        if best:
            date_str = target.strftime("%-m/%-d")
            return {"date": date_str, **best}
    return None


# --- 月・潮汐情報（Skyfield天文計算） ---
MOON_PHASE_NAMES = ["🌑 新月", "🌒 三日月", "🌓 上弦", "🌔 十三夜",
                     "🌕 満月", "🌖 十八夜", "🌗 下弦", "🌘 二十六夜"]
TIDE_TYPES = {
    "大潮": [0, 1, 14, 15, 29],      # 新月・満月前後
    "中潮": [2, 3, 12, 13, 16, 17, 27, 28],
    "小潮": [4, 5, 11, 18, 19, 26],
    "長潮": [6, 20],
    "若潮": [7, 21],
}


def get_moon_info(target_date) -> dict:
    """月齢・月相・潮の種類を計算する（skyfield.almanac使用）。"""
    t_obs = ts.from_datetime(
        datetime(target_date.year, target_date.month, target_date.day,
                 20, 0, tzinfo=JST)
    )
    earth = eph['earth']
    sun = eph['sun']
    moon = eph['moon']

    # almanacで正確な月相角度を取得（0〜360°、0=新月、180=満月）
    phase_angle = almanac.moon_phase(eph, t_obs).degrees

    # 月相名（45°刻み）
    phase_index = int(phase_angle / 45) % 8
    phase_name = MOON_PHASE_NAMES[phase_index]

    # 月齢（synodic month = 29.53059日、phase_angle比例）
    moon_age = round(phase_angle / 360 * 29.53059, 1)

    # 潮の種類（月齢ベース）
    moon_age_int = int(moon_age + 0.5) % 30
    tide_type = "中潮"
    for t_name, days in TIDE_TYPES.items():
        if moon_age_int in days:
            tide_type = t_name
            break

    # 月の高度
    obs_pos = earth + OBSERVER
    moon_alt, _, _ = obs_pos.at(t_obs).observe(moon).apparent().altaz()

    # 輝面比（phase_angle: 0°=新月→照度0%, 180°=満月→照度100%）
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


# --- HTML（ローディング画面 + JS非同期取得） ---
def render_loading_html(target_date: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="ja"><head>
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
.loading .status{{color:rgba(255,255,255,.3);font-size:.9rem;letter-spacing:.1em}}
.loading .dots span{{opacity:.2;animation:blink 1.4s infinite}}
.loading .dots span:nth-child(2){{animation-delay:.2s}}
.loading .dots span:nth-child(3){{animation-delay:.4s}}
@keyframes blink{{0%,80%,100%{{opacity:.2}}40%{{opacity:1}}}}
.next-hint{{margin-top:3rem;padding-top:2rem;border-top:1px solid rgba(255,255,255,.08)}}
.next-label{{font-size:.75rem;color:rgba(255,255,255,.25);letter-spacing:.2em;margin-bottom:.8rem}}
.next-date{{font-size:1.1rem;color:rgba(255,255,255,.45);margin-bottom:.4rem}}
.next-dir{{font-size:.85rem;color:rgba(255,255,255,.3)}}
.moon-info{{margin-top:2.5rem;padding-top:1.5rem;border-top:1px solid rgba(255,255,255,.06)}}
.moon-phase{{font-size:1rem;color:rgba(255,255,255,.4);margin-bottom:.4rem}}
.moon-detail{{font-size:.8rem;color:rgba(255,255,255,.25);margin-bottom:.3rem}}
.moon-note{{font-size:.75rem;color:rgba(255,255,255,.2)}}
.footer{{position:fixed;bottom:1.5rem;font-size:.7rem;color:rgba(255,255,255,.15);letter-spacing:.1em}}
#content{{transition:opacity .5s ease}}
@media(min-width:600px){{.visible .time{{font-size:5.5rem}}.visible .direction{{font-size:1.5rem}}}}
</style></head><body>
<div class="place">竹富島・西桟橋</div>
<div class="date">{target_date}</div>
<div id="content">
  <div class="loading">
    <div class="status">衛星を探しています<span class="dots"><span>.</span><span>.</span><span>.</span></span></div>
  </div>
</div>
<div class="footer">Starlink Train Viewer</div>
<script>
(async()=>{{
  const el=document.getElementById('content');
  try{{
    const r=await fetch('/api/tonight');
    const d=await r.json();
    if(d.error){{
      el.innerHTML='<div class="error"><div class="status">'+d.error+'</div></div>';
      return;
    }}
    let h='';
    if(d.visible&&d.result){{
      const s=d.result;
      h='<div class="visible">'
        +'<div class="status">スターリンク 見える</div>'
        +'<div class="time">'+s.time_str+'</div>'
        +'<div class="direction">'+s.start_dir
        +' <span class="context">（'+s.start_context+'）</span>を見る</div>'
        +'<div class="arrow">↓</div>'
        +'<div class="direction">'+s.end_dir+' へ流れる</div>'
        +'<div class="count">約'+s.sat_count+'機の連なり</div></div>';
    }}else{{
      h='<div class="not-visible"><div class="status">今夜は<br>見えなそうです</div></div>';
      if(d.next_visible){{
        const n=d.next_visible;
        h+='<div class="next-hint">'
          +'<div class="next-label">次の候補</div>'
          +'<div class="next-date">'+n.date+' '+n.time_str+'</div>'
          +'<div class="next-dir">'+n.start_dir+'（'+n.start_context+'）→ '+n.end_dir+'</div></div>';
      }}
    }}
    if(d.moon){{
      const m=d.moon;
      h+='<div class="moon-info">'
        +'<div class="moon-phase">'+m.phase+'</div>'
        +'<div class="moon-detail">月齢 '+m.age+'　'+m.tide_type+'</div>'
        +'<div class="moon-note">'+m.sky_note+'</div></div>';
    }}
    el.style.opacity='0';
    setTimeout(()=>{{el.innerHTML=h;el.style.opacity='1';}},200);
  }}catch(e){{
    el.innerHTML='<div class="error"><div class="status">データの取得に失敗しました</div></div>';
  }}
}})();
</script>
</body></html>"""


# --- エンドポイント ---
@app.get("/", response_class=HTMLResponse)
async def home():
    """ローディング画面を即座に返し、データはJSで非同期取得する。"""
    now = datetime.now(tz=JST)
    today = now.date()
    obs_start = datetime(today.year, today.month, today.day, OBS_START_HOUR, 0, tzinfo=JST)
    obs_end = datetime(today.year, today.month, today.day, OBS_END_HOUR, 0, tzinfo=JST)
    if now > obs_end:
        obs_start += timedelta(days=1)
    target_date = obs_start.strftime("%m/%d")
    return render_loading_html(target_date)


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
        recent = sats[-MAX_SATS:] if len(sats) > MAX_SATS else sats
        passes = find_train_passes(recent, obs_start, obs_end)
        trains = cluster_into_trains(passes)
        result = select_best_train(trains)
        next_visible = None
        if not result:
            tomorrow = obs_start.date() + timedelta(days=1)
            next_visible = find_next_visible(recent, tomorrow, max_days=3)
        moon = get_moon_info(obs_start.date())
    except Exception as e:
        logger.error("エラー: %s", e)
        return JSONResponse({"error": "衛星データの取得に失敗しました"})

    return {
        "visible": result is not None,
        "result": result,
        "next_visible": next_visible,
        "moon": moon,
    }


# --- 起動時にTLEプリフェッチ ---
@app.on_event("startup")
async def startup_prefetch():
    """アプリ起動時にTLEデータを事前取得する。"""
    try:
        sats = await fetch_tle_data()
        logger.info("起動時TLEプリフェッチ完了: %d衛星", len(sats))
    except Exception as e:
        logger.warning("起動時TLEプリフェッチ失敗: %s", e)
