# 西桟橋 Starlink Viewer

竹富島・西桟橋からスターリンク衛星トレインが見えるタイミングを案内するWebアプリ。

## コンセプト

衛星観測ツールではなく、西桟橋に立った人が空を見上げる"きっかけ"をつくる装置。
場所・時間・方向だけを静かに提示する。

## 機能

- 今夜スターリンクのトレインが見える可能性を判定
- 見える場合: 時刻・方角・流れる方向を表示
- 見えない場合: 「今夜は見えなそうです」

## 技術スタック

- FastAPI + Jinja2
- Skyfield（衛星軌道計算）
- Celestrak TLEデータ
- PWA対応

## ローカル実行

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## デプロイ（Render）

1. GitHubリポジトリをRenderに接続
2. `render.yaml` が自動検出される
3. デプロイ完了
