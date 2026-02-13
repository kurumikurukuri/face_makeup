# クイックリファレンス

基本的な使い方をまとめています。

---

## 基本的な使い方

### セットアップ

```bash
# 仮想環境をセットアップ
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存をインストール
pip install -e ".[app]"

# Streamlitアプリを起動
streamlit run apps/streamlit_app.py
```

Windows では `run_streamlit_app.bat` で自動セットアップ可能。

### ライブラリとして使う

```python
from face_makeup import MakeupProcessor

processor = MakeupProcessor()
ctx = processor.open_path("input.jpg")

# メイクを適用
face = ctx.faces[0]
face.lip(color=(255, 100, 100), opacity=0.5) \
    .cheek(color=(255, 180, 180), opacity=0.3) \
    .eyelid(color=(180, 130, 150), opacity=0.4)

# 保存
ctx.save("output.png")
```

---

## 主要API

### MakeupProcessor

画像を開いて顔検出・ランドマーク抽出を行う。

```python
from face_makeup import MakeupProcessor, ProcessorConfig

# 設定をカスタマイズ
config = ProcessorConfig(
    min_detection_confidence=0.5,  # 顔検出の信頼度
    max_num_faces=10               # 最大検出数
)
processor = MakeupProcessor(config)

# ファイルパスから開く
ctx = processor.open_path("image.jpg")

# NumPy配列から開く
import cv2
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ctx = processor.open(image_rgb)
```

### ImageContext

画像ごとの処理コンテキスト。

```python
# 顔を取得
face = ctx.faces[0]
print(f"検出された顔の数: {ctx.face_count}")

# レンダリング
result = ctx.render()

# 保存
ctx.save("output.png")

# デバッグモード（outputs/debug/ に詳細情報を出力）
result_debug = ctx.render(debug=True)
```

### Face

メイクを適用するメソッドチェーン。

| メソッド | 説明 | 主要パラメータ |
|---------|------|--------------|
| `lip(color, opacity, blur_strength)` | 口紅 | color (RGB), opacity (0-1) |
| `cheek(color, opacity, side, center, radius, blur_strength)` | チーク | side ('left'/'right'/'both'), center (-1〜1), radius (0〜1) |
| `eyelid(color, opacity, side, blur_strength)` | アイシャドウ | side ('left'/'right'/'both') |
| `eyelid_edge(color, opacity, side, blur_strength)` | まぶたの際 | side ('left'/'right'/'both') |
| `eyeliner(color, opacity, thickness, side, blur_strength)` | アイライナー | thickness (線の太さ) |
| `eyeiris(color, opacity, side, blur_strength)` | カラーコンタクト | side ('left'/'right'/'both') |
| `clear()` | 全メイクをクリア | - |

**使用例:**

```python
# 基本的な使い方
face.lip(color=(255, 100, 100), opacity=0.5)

# メソッドチェーン
face.lip(color=(255, 100, 100), opacity=0.5) \
    .cheek(color=(255, 180, 180), opacity=0.3) \
    .eyeliner(color=(0, 0, 0), opacity=1.0, thickness=2)

# 左右で異なる設定
face.cheek(
    color=(255, 180, 180),
    opacity=0.3,
    side='left',
    center=(0.1, 0.0)
).cheek(
    color=(255, 200, 200),
    opacity=0.25,
    side='right',
    center=(-0.1, 0.0)
)

# チーク位置・形状の詳細設定
face.cheek(
    color=(255, 180, 180),
    opacity=0.3,
    side='both',
    center=(0.1, -0.1),   # X, Y オフセット
    radius=(0.6, 0.6),    # 楕円の半径
    blur_strength=0.5
)
```

---

## よくある問題と対処法

### 顔が検出されない

**症状:** `ctx.face_count == 0` になる、またはエラー

**対処:**
1. 画像内に顔が含まれているか確認
2. 顔が十分な大きさで写っているか
3. 顔が極端に傾いていないか確認
4. `min_detection_confidence` を下げる (例: 0.3)

```python
config = ProcessorConfig(min_detection_confidence=0.3)
processor = MakeupProcessor(config)
```

### 処理が遅い

**症状:** メイク適用に時間がかかる

**対処:**
1. 入力画像のサイズを 2048x2048 以下に縮小
2. 最大顔数を制限 (`max_num_faces` を小さくする)
3. デバッグモードを無効にする

### MediaPipeのインストールエラー

**症状:** `pip install mediapipe` が失敗

**対処:**
1. Python 3.10〜3.12 を使用
2. `mediapipe==0.10.21` を明示的に指定
3. OS が対応しているか確認 (Windows/macOS/Linux)

### HEIF/HEIC 画像が読み込めない

**症状:** HEIF形式の画像が開けない

**対処:**
```bash
pip install pillow-heif>=0.13.0
```

### `ctx.show()` が動作しない

**症状:** OpenCVウィンドウが表示されない

**対処:**
- WSL やヘッドレス環境では `show()` は使用不可
- `ctx.save("output.png")` で保存して確認

---

## デバッグTips

### デバッグモードを使う

`render(debug=True)` で詳細情報を `outputs/debug/` に保存：

- 元画像
- バウンディングボックス
- ランドマーク描画
- マスクのヒートマップ
- 結果画像と差分

```python
result = ctx.render(debug=True)
# outputs/debug/<timestamp>/ に各種画像が保存される
```

### マスク重複の理解

レイヤー間のマスク重複は自動解決：

- **同一型**: 重複OK (例: `eyelid` + `eyelid_edge`)
- **異なる型**: 優先度の高い型が勝つ

優先度: `EyeIrisParams` > `EyeLinerParams` > `EyeShadowParams` > `LipsParams` > `CheekParams`

---

## 環境変数・設定

本プロジェクトでは `.env` ファイルは不要。すべて `ProcessorConfig` で設定。

```python
config = ProcessorConfig(
    min_detection_confidence=0.5,  # 顔検出の最小信頼度 (0.0-1.0)
    min_tracking_confidence=0.5,   # トラッキングの最小信頼度 (0.0-1.0)
    refine_landmarks=True,         # 詳細ランドマーク (478点) を使用
    max_num_faces=10               # 検出する最大顔数
)
```

---

## セキュリティ（Streamlitアプリ）

- **SSRF 対策**: URL 入力はプライベート IP をブロック
- **ファイルサイズ制限**: アップロード/URL 画像は 10MB まで
- 本番環境では `--server.address 0.0.0.0` の使用に注意

---

## ロールバック（トラブル時）

### venv の再構築

```bash
# venv を削除
rm -rf venv  # Linux/macOS
rmdir /s /q venv  # Windows

# 再構築
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev,app]"
```

### 依存の再インストール

```bash
pip install -r requirements.txt
```
