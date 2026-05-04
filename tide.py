"""
気象庁潮位予測データ（石垣 IS.txt）の取得・キャッシュ・パース。

データソース: https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/txt/{YYYY}/IS.txt
- 石垣は西桟橋（竹富島）に最も近い気象庁潮汐予測地点。
- 年次データを年1回取得してローカルキャッシュ。
- フォーマット: 各行に1日分 (72桁毎時潮位 + 6桁年月日 + 2桁地点 + 満潮4x7桁 + 干潮4x7桁)
"""

from __future__ import annotations

from datetime import datetime, date, timedelta, timezone
from pathlib import Path
import logging
import httpx

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
STATION_CODE = "IS"   # 石垣
DATUM_NOTE = "石垣・平均海面基準（cm）"
TIDE_URL_TEMPLATE = "https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/txt/{year}/IS.txt"

_cache_dir = Path(__file__).parent
_tide_cache_mem: dict[int, dict[date, dict]] = {}


def _cache_file(year: int) -> Path:
    return _cache_dir / f"tide_cache_{year}.txt"


def _parse_hhmm(s: str) -> str | None:
    """4桁HHMM文字列を'HH:MM'に。9999は欠番としてNone。"""
    s = s.strip()
    if not s or s == "9999":
        return None
    try:
        hhmm = int(s)
        h, m = divmod(hhmm, 100)
        if 0 <= h < 24 and 0 <= m < 60:
            return f"{h:02d}:{m:02d}"
    except ValueError:
        pass
    return None


def _parse_cm(s: str) -> int | None:
    s = s.strip()
    if not s or s == "999":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_line(line: str) -> tuple[date, dict] | None:
    """IS.txt の1行をパースして (日付, {highs, lows}) を返す。不正行はNone。"""
    if len(line) < 136:
        return None
    try:
        # 年月日: 73-78カラム（1-indexed）→ Python slice [72:78]
        yy = int(line[72:74].strip())
        mm = int(line[74:76].strip())
        dd = int(line[76:78].strip())
        # 2000年代想定
        full_year = 2000 + yy
        d = date(full_year, mm, dd)

        # 地点記号: 79-80 → slice [78:80]
        station = line[78:80].strip()
        if station != STATION_CODE:
            return None

        # 満潮: 81-108 → [80:108], 4ペア x 7桁（時刻4+潮位3）
        highs: list[tuple[str, int]] = []
        for i in range(4):
            base = 80 + i * 7
            t = _parse_hhmm(line[base:base + 4])
            cm = _parse_cm(line[base + 4:base + 7])
            if t is not None and cm is not None:
                highs.append((t, cm))

        # 干潮: 109-136 → [108:136]
        lows: list[tuple[str, int]] = []
        for i in range(4):
            base = 108 + i * 7
            t = _parse_hhmm(line[base:base + 4])
            cm = _parse_cm(line[base + 4:base + 7])
            if t is not None and cm is not None:
                lows.append((t, cm))

        return d, {"highs": highs, "lows": lows}
    except (ValueError, IndexError):
        return None


def _parse_all(text: str) -> dict[date, dict]:
    """IS.txt 全文をパースして {date: {highs, lows}} を返す。"""
    result: dict[date, dict] = {}
    for line in text.splitlines():
        parsed = _parse_line(line)
        if parsed:
            d, info = parsed
            result[d] = info
    return result


async def _fetch_year(year: int) -> dict[date, dict]:
    """指定年のIS.txtをネットワーク取得してパース。"""
    url = TIDE_URL_TEMPLATE.format(year=year)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StarlinkNishi/2.0)"}
    async with httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        text = resp.text
    # キャッシュ保存
    try:
        _cache_file(year).write_text(text, encoding="utf-8")
        logger.info("潮汐データキャッシュ保存: %d年", year)
    except Exception as e:
        logger.warning("潮汐キャッシュ保存失敗: %s", e)
    return _parse_all(text)


def _load_cached(year: int) -> dict[date, dict] | None:
    """ローカルキャッシュから読み込む。存在しない・空なら None。"""
    p = _cache_file(year)
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8")
        if not text.strip():
            return None
        logger.info("潮汐データ ファイルキャッシュ使用: %d年", year)
        return _parse_all(text)
    except Exception as e:
        logger.warning("潮汐キャッシュ読み込み失敗: %s", e)
        return None


async def get_tide_data(year: int) -> dict[date, dict]:
    """指定年の潮汐データを取得。メモリ → ファイル → ネットワークの順。"""
    if year in _tide_cache_mem:
        return _tide_cache_mem[year]

    cached = _load_cached(year)
    if cached:
        _tide_cache_mem[year] = cached
        return cached

    try:
        fetched = await _fetch_year(year)
        if fetched:
            _tide_cache_mem[year] = fetched
            return fetched
    except Exception as e:
        logger.warning("潮汐データ取得失敗 (%d年): %s", year, e)

    return {}


async def get_tide_info(target_date: date) -> dict | None:
    """
    指定日の満干潮情報を返す。
    戻り値: {"highs": [{"time": "HH:MM", "cm": 174}, ...],
             "lows":  [{"time": "HH:MM", "cm": 18},  ...],
             "datum_note": "..."}
    取得失敗時は None。
    """
    data = await get_tide_data(target_date.year)
    if not data:
        return None

    info = data.get(target_date)
    if not info:
        return None

    return {
        "highs": [{"time": t, "cm": c} for t, c in info["highs"]],
        "lows":  [{"time": t, "cm": c} for t, c in info["lows"]],
        "datum_note": DATUM_NOTE,
    }
