# アーキテクチャ

本ドキュメントでは、Face Makeup Library の内部設計について説明します。

## ディレクトリ構成

```
makeup/
├── src/face_makeup/              # メインライブラリ
│   ├── __init__.py              # パッケージエントリポイント
│   ├── processor.py             # MakeupProcessorクラス（ステートレス）
│   ├── face.py                  # FaceクラスとFluent API
│   ├── context.py               # ImageContext（画像ごとの状態管理）
│   ├── renderer.py              # 内部レンダリングエンジン
│   ├── constants.py             # 顔パーツのランドマークインデックス
│   └── core/                    # コア処理モジュール
│       ├── types.py             # データ構造（FaceBBox, FaceLandmarksなど）
│       ├── face_detector.py     # MediaPipe顔検出ラッパー
│       ├── landmark_extractor.py # MediaPipe Face Mesh（478ランドマーク）
│       ├── segmenter.py         # マスク生成
│       ├── makeup_blender.py    # HSVベースの色適用ロジック
│       └── debug_visualizer.py  # デバッグ可視化ユーティリティ
├── apps/
│   └── streamlit_app.py         # Streamlit デモアプリ
├── tests/                       # テストスイート
├── pyproject.toml              # パッケージメタデータ
└── requirements.txt            # 依存パッケージリスト
```

## 処理パイプライン

```
画像入力
    │
    ▼
FaceDetector (MediaPipe Face Detection)
    │  → FaceBBox[] (バウンディングボックス)
    ▼
LandmarkExtractor (MediaPipe Face Mesh)
    │  → FaceLandmarks[] (478ランドマーク + 3D座標)
    ▼
[Face オブジェクト作成]
    │
    ▼
[ユーザーがメイクレイヤーを設定]
    │  face.lip().cheek().eyelid()...
    ▼
render() 呼び出し
    │
    ▼
Renderer.compose()
    ├→ FaceSegmenter.segment()
    │    ランドマーク座標からパーツマスクを生成
    │    (polygon / convex / curve / ellipse)
    ├→ _resolve_overlaps()
    │    型ベースの優先度でマスク重複を解決
    └→ MakeupBlender
         ぼかし + HSVカラーブレンド → 最終画像
    │
    ▼
出力画像 (RGB uint8)
```

### セグメンテーション方式

`constants.py` の `PART_LANDMARK_MAP` で各パーツのランドマークインデックスとセグメンテーション方式が定義されています。

| 方式 | 説明 | 使用パーツ |
|------|------|-----------|
| `polygon` | ランドマーク点を結ぶポリゴン | lips, eyelid, eyelid_edge, eye |
| `convex` | 凸包（convex hull） | cheek |
| `curve` | ポリライン（非閉合曲線） | eyeliner |
| `ellipse` | 楕円フィッティング（`cv2.fitEllipse`） | eyeiris |

`eyelid_edge` は `('mid', idx1, idx2)` タプルによる中間点計算をサポートし、まぶたの際の精密なマスク生成を実現しています。

### マスク重複解決

`PARAMS_PRIORITY` で定義された型ベースの優先度により、異なるPartParams型間のマスク重複を自動解決します。

| 優先度 | PartParams型 | パーツ |
|--------|--------------|--------|
| 40 (最上層) | EyeIrisParams | 虹彩（カラコン） |
| 30 | EyeLinerParams | アイライナー |
| 20 | EyeShadowParams | アイシャドウ、まぶた際 |
| 10 | LipsParams | リップ |
| 0 (最下層) | CheekParams | チーク |

**ルール:**
- 同一型のレイヤーは重複OK（例: eyelid + eyelid_edge は両方 EyeShadowParams）
- 異なる型は高優先度が勝つ（例: eyeiris のマスク領域では eyelid のマスクが削られる）
- 処理順: 高優先度のレイヤーから順に確定し、低優先度レイヤーのマスクを `np.clip(mask - used_mask, 0, 1)` で減算

### カラーブレンド処理

`MakeupBlender` はHSV色空間でのブレンドを行います。

1. 画像をRGB→HSVに変換
2. ターゲット色のH（色相）とS（彩度）を適用
3. V（明度）は元画像の80% + ターゲットの20%で混合
4. opacityに基づくアルファブレンド
5. ニュートラル色（彩度 < 0.05）の場合は明度のみ下げる（黒系アイライナー等）

### 虹彩マスク処理

カラーコンタクト（`eyeiris`）は特殊なマスク処理を行います。

1. `eyeiris_left/right` のマスク: 虹彩ランドマーク5点から楕円フィッティング
2. `eye_left/right` のマスク: 目の輪郭ランドマーク16点からポリゴン
3. 両マスクの Bitwise AND により、まぶたで隠れた部分を除外

### チーク3Dマスク処理

3Dランドマーク（`points_3d`）が利用可能な場合、チークは顔の曲面に沿った楕円マスクを生成します。

1. チーク領域のランドマークから接線空間（tangent space）のU/V/W基底を構築
2. 二次曲面フィッティング（最小二乗法）で顔の曲面を近似
3. 接線空間上で楕円を定義し、曲面上に投影
4. `cv2.fillPoly` で投影された楕円内部のピクセルを判定

### ML依存とルールベース処理の区分

| 処理 | 方式 | 説明 |
|------|------|------|
| 顔検出 | ML (MediaPipe Face Detection) | 画像内の顔位置を検出 |
| ランドマーク抽出 | ML (MediaPipe Face Mesh) | 478点の顔特徴点 + 3D座標を抽出 |
| パーツマスク生成 | ルールベース | ランドマーク座標からマスクを生成 |
| マスク重複解決 | ルールベース | 型ベース優先度によるレイヤー間重複解決 |
| 色適用 | ルールベース | HSV色空間でのブレンド処理 |
| チーク曲面近似 | ルールベース | 3Dランドマークから二次曲面フィッティング |

## レイヤー構成

```
┌─────────────────────────────────────────────────┐
│          ユーザーインターフェース層              │
│  (streamlit_app.py - カラーピッカー付きWeb UI)  │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│       パブリックAPI層 (processor.py)             │
│  MakeupProcessor.open() / open_path()           │
│  → ImageContext を返す                          │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│      状態管理層                                  │
│  ImageContext (レンダリングキャッシュ管理)       │
│  + Face (Fluent API, MakeupLayer蓄積)          │
│  → 変更時にキャッシュ自動無効化                 │
└────────────────┬────────────────────────────────┘
                 │  render() 呼び出し時
┌────────────────▼────────────────────────────────┐
│        レンダリングエンジン (renderer.py)        │
│  Renderer                                       │
│  ├ FaceSegmenter  → パーツマスク生成             │
│  ├ _resolve_overlaps → マスク重複解決           │
│  └ MakeupBlender  → HSVカラーブレンド           │
│  (DebugVisualizer → デバッグ可視化)             │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│      基盤モデル                                  │
│  MediaPipe Face Detection (顔検出)              │
│  MediaPipe Face Mesh (ランドマーク抽出)         │
└─────────────────────────────────────────────────┘
```

## 設計原則

### ステートレス設計

`MakeupProcessor` はMediaPipeモデルを保持しますが、画像処理の状態は持ちません。
各画像の処理状態は `ImageContext` に分離されているため、同一の `MakeupProcessor` インスタンスで複数の画像を処理できます。

### Fluent API

`Face` クラスはメソッドチェーンをサポートし、直感的なメイク適用を可能にします：

```python
face.lip(color=(255, 100, 100), opacity=0.5) \
    .cheek(color=(255, 180, 180), opacity=0.3) \
    .eyelid(color=(180, 130, 150), opacity=0.4)
```

### 遅延レンダリングとキャッシュ

メイクの設定（`lip()`, `cheek()` など）はレイヤー情報を蓄積するのみで、実際の画像処理は `render()` 呼び出し時に一括実行されます。

- 複数のメイク設定を効率的にバッチ処理
- レンダリング結果は `ImageContext` にキャッシュされる
- メイク設定の変更（`Face._notify_change()`）でキャッシュが自動無効化
- `debug=True` のレンダリングはキャッシュされない

### 二重API設計

各メイクメソッドは2通りの指定方法を提供します：
- **簡易API**: `face.lip(color=(255,0,0), opacity=0.5)` — キーワード引数で直接指定
- **詳細API**: `face.lip(params=LipsParams(...))` — Paramsオブジェクトで指定

`params` が指定された場合はそれが優先され、内部でバリデーション済みのオブジェクトがそのまま使用されます。
