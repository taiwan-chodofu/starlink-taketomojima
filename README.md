# 西桟橋 Starlink Viewer

竹富島・西桟橋からスターリンク衛星トレインが見えるタイミングを案内するWebアプリ。

## コンセプト

> 衛星観測ツールではなく、西桟橋に立った人が空を見上げる"きっかけ"をつくる装置。
> 場所・時間・方角だけを、静かに提示する。

## 機能

- 今夜スターリンクのトレインが見える可能性を判定する
- 見える場合: 時刻 / 方角 / カウントダウン / 流れる方向を表示する
- 見えない場合: 次の候補日 または 新月期（星空の好条件）を提示する
- 今夜の月齢・月相・潮の種類を折りたたみで提示する（トグル）
- 時刻に連動して背景色が変化する（夕焼け → 薄明 → 夜）

## 技術スタック

| レイヤ | 採用 |
|--------|------|
| Webフレームワーク | FastAPI + Jinja2 |
| 衛星軌道計算 | Skyfield |
| TLEデータ | tle.ivanstanojevic.me / Celestrak（多重フォールバック） |
| キャッシュ | メモリ（2時間）+ ファイル（24時間）|
| フロント | バニラJS（依存ゼロ） |
| PWA | manifest.json + Service Worker |

## 観測パラメータ

| 項目 | 値 |
|------|-----|
| 緯度経度 | 24.3237°N, 124.0893°E（西桟橋） |
| 対象時間 | 18:00〜21:00 JST |
| 最低高度 | 30°（今夜）/ 20°（翌日以降の予測） |
| 薄明条件 | 太陽高度 -18°〜-6° |
| トレイン判定 | 5分以内・方位角30°以内に3機以上 |

## ローカル実行

```bash
cd starlink_nishi
pip install -r requirements.txt
uvicorn main:app --reload
```

ブラウザで http://127.0.0.1:8000 を開く。

## デプロイ（Render）

1. GitHubリポジトリをRenderに接続する
2. `render.yaml` が自動検出される
3. デプロイ完了

## ファイル構成

```
starlink_nishi/
├── main.py              # FastAPIアプリ本体（計算・API）
├── templates/
│   └── index.html       # UI（時刻連動背景・カウントダウン）
├── static/
│   └── manifest.json    # PWAマニフェスト
├── de421.bsp            # 惑星暦（太陽・月の位置計算用）
├── tle_cache.json       # TLEファイルキャッシュ（自動生成）
├── test_mock.py         # 単体テスト（モック）
├── test_integration.py  # 統合テスト（ネットワーク必要）
├── requirements.txt
└── render.yaml
```

## テスト

```bash
# ネットワーク不要のロジックテスト
py test_mock.py

# TLE取得を含む統合テスト
py test_integration.py
```
