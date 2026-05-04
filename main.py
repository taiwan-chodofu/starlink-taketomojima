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
from datetime import datetime, timedelta, timezone, date as date_type
from contextlib import asynccontextmanager
from pathlib import Path
import httpx
import logging
import json
import math

from tide import get_tide_info

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
# Celestrak全量（約10,000機）は理想だが 403 Forbidden/rate-limit が頻発するため
# 人気順API（ivanstanojevic）を主に使い、Celestrakは補助扱いにする。
# page-size=200 は人気順上位の"新しい打ち上げ＋運用中の主要機"を網羅できる実用的な上限。
TLE_URLS = [
    "https://tle.ivanstanojevic.me/api/tle/?search=starlink&page-size=100&sort=popularity&sort-dir=desc&page=1",
    "https://tle.ivanstanojevic.me/api/tle/?search=starlink&page-size=100&sort=popularity&sort-dir=desc&page=2",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
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

# --- Deployed/Not-deployed 判別 ---
# Starlink は運用軌道 550km 前後、展開前の低軌道 210-350km が "train" として見やすい。
# TLE の平均運動 (mean motion, rev/day) から近似高度を算出:
#   a = (mu / n^2)^(1/3),  h = a - R_earth
# 運用軌道かつ新しめの衛星を含めた幅広なフィルタ閾値:
TRAIN_ALT_MIN_KM = 200.0
TRAIN_ALT_MAX_KM = 600.0
MAX_SATS_AFTER_FILTER = 1500   # 軌道フィルタ後の上限（それでも多ければ直近のものに絞る）

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
    """TLEデータをファイルキャッシュに保存。ただし既存キャッシュよりサンプルが少なければ保存しない。"""
    try:
        existing = _load_file_cache()
        if existing is not None and len(existing) > len(sats) * 5:
            logger.info("既存キャッシュの方が大量（%d → %d）、上書きスキップ", len(existing), len(sats))
            return
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

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/plain,application/json,*/*",
    }
    async with httpx.AsyncClient(timeout=60.0, headers=headers, follow_redirects=True) as client:
        # ivanstanojevic のページングを優先: page=1 と page=2 の両方を取って連結（最大200機）
        accumulated: list[tuple[str, str, str]] = []
        seen_names: set[str] = set()
        for url in TLE_URLS:
            try:
                logger.info("TLE取得試行: %s", url)
                resp = await client.get(url)
                resp.raise_for_status()
                sats = _parse_tle_response(resp.text, url)
                if not sats:
                    continue
                # ivanstanojevic はページング用なので重複除去しつつ蓄積
                if "ivanstanojevic" in url:
                    for s in sats:
                        if s[0] not in seen_names:
                            accumulated.append(s)
                            seen_names.add(s[0])
                    continue
                # Celestrak 等の全量ソースで成功した場合はそちらを優先して即return
                _tle_cache["data"] = sats
                _tle_cache["fetched_at"] = now
                _save_file_cache(sats)
                logger.info("TLE取得成功(全量ソース): %d衛星", len(sats))
                return sats
            except Exception as e:
                logger.warning("TLE取得失敗(%s): %s", url, e)

        # ivanstanojevic の累積結果を使う（2ページ連結 = 最大200機）
        if accumulated:
            _tle_cache["data"] = accumulated
            _tle_cache["fetched_at"] = now
            _save_file_cache(accumulated)
            logger.info("TLE取得成功(ページング連結): %d衛星", len(accumulated))
            return accumulated

    logger.info("ネットワーク失敗、ファイルキャッシュを試行")
    cached = _load_file_cache()
    if cached:
        _tle_cache["data"] = cached
        _tle_cache["fetched_at"] = now
        return cached

    raise RuntimeError("TLEデータを取得できません")


# --- 軌道高度フィルタ（Deployed/Not-deployed 判別） ---
# 地球重力定数 mu = GM_earth [km^3 / s^2]、地球赤道半径 R_earth [km]
_MU_EARTH = 398600.4418
_R_EARTH = 6378.137


def _mean_motion_to_altitude_km(mean_motion_rev_per_day: float) -> float:
    """平均運動（rev/day）から軌道長半径 → 平均高度を近似計算する。"""
    # n [rad/s] = mean_motion * 2π / 86400
    n_rad_s = mean_motion_rev_per_day * 2.0 * math.pi / 86400.0
    if n_rad_s <= 0:
        return 0.0
    # 軌道長半径 a [km]: n^2 * a^3 = mu → a = (mu / n^2)^(1/3)
    a_km = (_MU_EARTH / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
    return a_km - _R_EARTH


def _tle_line2_mean_motion(line2: str) -> float | None:
    """TLE 2行目から平均運動（rev/day）を抽出する。"""
    # TLE Line 2 のカラム 53-63（1-indexed）が mean motion
    # Python slice: [52:63]
    if len(line2) < 63:
        return None
    try:
        return float(line2[52:63].strip())
    except ValueError:
        return None


def filter_train_candidates(
    sats: list[tuple[str, str, str]],
    alt_min: float = TRAIN_ALT_MIN_KM,
    alt_max: float = TRAIN_ALT_MAX_KM,
    max_count: int = MAX_SATS_AFTER_FILTER,
) -> list[tuple[str, str, str]]:
    """TLE 2行目の平均運動から平均高度を計算し、トレイン候補の軌道にある衛星に絞る。

    alt_min〜alt_max の範囲にある衛星のみを返す。
    max_count を超えた場合は末尾から優先（リスト末尾が新しい打ち上げであることが多い）。
    """
    candidates: list[tuple[str, str, str]] = []
    for name, l1, l2 in sats:
        mm = _tle_line2_mean_motion(l2)
        if mm is None:
            continue
        alt = _mean_motion_to_altitude_km(mm)
        if alt_min <= alt <= alt_max:
            candidates.append((name, l1, l2))
    if len(candidates) > max_count:
        candidates = candidates[-max_count:]
    logger.info("軌道高度フィルタ: %d → %d衛星", len(sats), len(candidates))
    return candidates


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
    return summarize_train(best)


def summarize_train(train: list[dict]) -> dict:
    """1トレインを表示用dictに変換する。"""
    mid = train[len(train) // 2]
    avg_alt = sum(p["alt"] for p in train) / len(train)
    return {
        "time_str": mid["time"].strftime("%H:%M"),
        "time_iso": mid["time"].isoformat(),
        "start_dir": az_to_direction(train[0]["az"]),
        "end_dir": az_to_direction(train[-1]["az"]),
        "start_context": az_to_context(train[0]["az"]),
        "sat_count": len(train),
        "avg_alt": round(avg_alt, 1),
    }


def select_trains(trains: list[list[dict]], top_n: int = 3) -> list[dict]:
    """上位N個のトレインを表示用dictで返す。衛星数×平均高度でランキング。"""
    if not trains:
        return []
    ranked = sorted(
        trains,
        key=lambda t: (len(t), sum(p["alt"] for p in t) / len(t)),
        reverse=True,
    )
    return [summarize_train(t) for t in ranked[:top_n]]


# --- B5: 統合可視スコア（0-100） ---
def calc_visibility_score(
    train_summary: dict,
    moon_info: dict,
    obs_start: datetime,
) -> int:
    """
    トレインの"見えやすさ"を0〜100で返す統合スコア。
    - 衛星数: 多いほど良い（上限20機で満点）
    - 平均高度: 高いほど良い（30°〜90°を線形にスケール）
    - 月の明るさ: 照度が低いほど良い、かつ月が沈んでいれば加点
    - 日没経過: 日没直後の薄明中盤がピーク
    """
    score = 0.0

    # 衛星数: 最大40点
    sat_count = train_summary.get("sat_count", 0)
    score += min(sat_count / 20.0, 1.0) * 40

    # 平均高度: 最大30点
    avg_alt = train_summary.get("avg_alt", 0.0)
    alt_norm = max(0.0, min((avg_alt - MIN_ALT_DEG) / (90.0 - MIN_ALT_DEG), 1.0))
    score += alt_norm * 30

    # 月の影響: 最大20点（月が暗い/沈んでいるほど良い）
    illumination = moon_info.get("illumination", 50.0)
    moon_alt = moon_info.get("moon_alt", 0.0)
    moon_penalty = (illumination / 100.0) if moon_alt > 0 else 0.2  # 沈んでれば軽微
    score += (1.0 - moon_penalty) * 20

    # 薄明適性: 最大10点（日没から30〜90分後がピーク）
    # 簡略: obs_start が 18:30-20:00 なら満点、それ以外は漸減
    hour = obs_start.hour + obs_start.minute / 60.0
    if 18.5 <= hour <= 20.0:
        twilight_bonus = 1.0
    elif 18.0 <= hour <= 20.5:
        twilight_bonus = 0.7
    else:
        twilight_bonus = 0.4
    score += twilight_bonus * 10

    return max(0, min(100, int(round(score))))


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


# --- 日月イベント（日の出/日の入り/月の出/月の入り/薄明） ---
def _find_event_time(f, t0, t1, target_value: int) -> str | None:
    """Skyfield almanac の離散関数 f から、指定値への遷移時刻を'HH:MM'で返す。

    f: Skyfield discrete function (e.g., sunrise_sunset, risings_and_settings)
    target_value: 1=昇る(sunrise/moonrise)、0=沈む(sunset/moonset)
    """
    try:
        times, events = almanac.find_discrete(t0, t1, f)
    except Exception:
        return None
    for t, e in zip(times, events):
        if int(e) == target_value:
            return t.astimezone(JST).strftime("%H:%M")
    return None


def _find_twilight_bounds(t0, t1) -> tuple[str | None, str | None]:
    """天文薄明（太陽高度 -18°）の開始・終了時刻を返す。
    対象時間帯内で太陽高度が -18° を横切る最初/最後の時刻を検出。"""
    try:
        f = almanac.dark_twilight_day(eph, OBSERVER)
        times, events = almanac.find_discrete(t0, t1, f)
    except Exception:
        return None, None

    twilight_start = None
    twilight_end = None
    # イベント値: 0=night, 1=astronomical, 2=nautical, 3=civil, 4=day
    # 日没後: day → civil → nautical → astronomical → night
    # 1（astronomical twilight開始）への遷移 = 薄明開始（観測に使える時刻帯）
    for t, e in zip(times, events):
        event_val = int(e)
        if event_val == 3:   # civil twilight 終了 = 観測条件（-6°）の開始
            twilight_start = t.astimezone(JST).strftime("%H:%M")
        elif event_val == 1:  # astronomical 終了 = 真の夜（-18°）の開始
            twilight_end = t.astimezone(JST).strftime("%H:%M")
    return twilight_start, twilight_end


def get_sun_moon_events(target_date: date_type) -> dict:
    """指定日の日の出/日の入り/月の出/月の入り/薄明開始・終了を返す。"""
    # その日の00:00〜翌日00:00 JST
    day_start = datetime(target_date.year, target_date.month, target_date.day,
                         0, 0, tzinfo=JST)
    day_end = day_start + timedelta(days=1)
    t0 = ts.from_datetime(day_start)
    t1 = ts.from_datetime(day_end)

    # 日の出・日の入り
    f_sun = almanac.sunrise_sunset(eph, OBSERVER)
    sunrise = _find_event_time(f_sun, t0, t1, 1)
    sunset = _find_event_time(f_sun, t0, t1, 0)

    # 月の出・月の入り
    f_moon = almanac.risings_and_settings(eph, eph['moon'], OBSERVER)
    moonrise = _find_event_time(f_moon, t0, t1, 1)
    moonset = _find_event_time(f_moon, t0, t1, 0)

    # 薄明の開始・終了（観測に適した時間帯の境界）
    twilight_start, twilight_end = _find_twilight_bounds(t0, t1)

    return {
        "sunrise": sunrise,
        "sunset": sunset,
        "moonrise": moonrise,
        "moonset": moonset,
        "twilight_start": twilight_start,
        "twilight_end": twilight_end,
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
        candidates = filter_train_candidates(sats)
        passes = find_train_passes(candidates, obs_start, obs_end)
        trains = cluster_into_trains(passes)

        target_date = obs_start.date()
        moon = get_moon_info(target_date)
        events = get_sun_moon_events(target_date)
        tide = await get_tide_info(target_date)

        # 複数候補提示 + 可視スコア
        all_trains = select_trains(trains, top_n=3)
        for t in all_trains:
            t["score"] = calc_visibility_score(t, moon, obs_start)
        # プライマリ候補は最高スコアのもの
        result = all_trains[0] if all_trains else None

        next_visible = None
        next_dark_sky = None
        if not result:
            tomorrow = obs_start.date() + timedelta(days=1)
            next_visible = find_next_visible(candidates, tomorrow, max_days=7)
            if not next_visible:
                next_dark_sky = find_next_dark_sky(tomorrow, max_days=30)
    except Exception as e:
        logger.error("判定エラー: %s", e)
        return JSONResponse({"error": "衛星データの取得に失敗しました"})

    return {
        "visible": result is not None,
        "result": result,
        "all_trains": all_trains,
        "next_visible": next_visible,
        "next_dark_sky": next_dark_sky,
        "moon": moon,
        "events": events,
        "tide": tide,
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



