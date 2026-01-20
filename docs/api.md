# APIリファレンス

## MakeupProcessor

メインのエントリポイント。モデルを一度読み込み、複数の画像を処理できます。

### コンストラクタ

```python
MakeupProcessor(config: ProcessorConfig = None)
```

初期化時にMediaPipeのモデルを読み込みます。

**引数:**
- `config`: プロセッサ設定（省略時はデフォルト値を使用）

### ProcessorConfig

プロセッサの動作を設定するデータクラス。

```python
from face_makeup import MakeupProcessor, ProcessorConfig

config = ProcessorConfig(
    min_detection_confidence=0.5,  # 顔検出の最小信頼度 (0.0-1.0)
    min_tracking_confidence=0.5,   # トラッキングの最小信頼度 (0.0-1.0)
    refine_landmarks=True,         # 詳細ランドマーク抽出を有効化
    max_num_faces=10               # 検出する最大顔数
)

processor = MakeupProcessor(config)
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `min_detection_confidence` | `float` | `0.5` | 顔検出の最小信頼度 |
| `min_tracking_confidence` | `float` | `0.5` | トラッキングの最小信頼度 |
| `refine_landmarks` | `bool` | `True` | 詳細ランドマーク（478点）を抽出 |
| `max_num_faces` | `int` | `10` | 検出する最大顔数 |

### メソッド

#### `open_path(path: str) -> ImageContext`

ファイルパスから画像を開き、顔検出・ランドマーク抽出を実行します。

**引数:**
- `path`: 画像ファイルのパス（JPEG, PNG, WebP対応）

**戻り値:**
- `ImageContext`: 画像処理コンテキスト

**例外:**
- `FileNotFoundError`: ファイルが存在しない場合
- `ValueError`: パスがファイルでない場合、または画像形式が不正な場合
- `ImportError`: Pillow がインストールされていない場合

#### `open(image: np.ndarray) -> ImageContext`

NumPy配列から画像を開きます。

**引数:**
- `image`: RGB形式のNumPy配列 (H, W, 3), dtype=uint8

**戻り値:**
- `ImageContext`: 画像処理コンテキスト

**例外:**
- `ValueError`: 画像がNone/空、dtypeがuint8でない、shapeが(H, W, 3)でない場合

**注意:** 顔が検出されない場合でもエラーは発生しません。`ImageContext.face_count` が `0` になるため、呼び出し側で確認してください。

### 使用例

```python
from face_makeup import MakeupProcessor

processor = MakeupProcessor()

# ファイルパスから
ctx = processor.open_path("image.jpg")

# NumPy配列から
import cv2
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ctx = processor.open(image_rgb)
```

### マスク重複解決

レイヤー間のマスク重複は、`PartParams`の型に基づいて自動的に解決されます：

- **同一型のレイヤー**: 重複OK（例: `eyelid` + `eyelid_edge` は両方 `EyeShadowParams`）
- **異なる型のレイヤー**: 高優先度の型が勝つ

優先度順: `EyeIrisParams` > `EyeLinerParams` > `EyeShadowParams` > `LipsParams` > `CheekParams`

---

## ImageContext

画像ごとの処理コンテキスト。検出された顔とレンダリング機能を提供します。

### プロパティ

| プロパティ | 型 | 説明 |
|-----------|-----|------|
| `original_image` | `np.ndarray` | 元画像のコピー (H, W, 3) |
| `faces` | `list[Face]` | 検出された顔のリスト |
| `face_count` | `int` | 検出された顔の数 |

### メソッド

#### `face(index: int = 0) -> Face`

インデックスで顔を取得します。

**引数:**
- `index`: 顔のインデックス（0始まり、デフォルト: 0）

**戻り値:**
- `Face`: 指定された顔オブジェクト

**例外:**
- `IndexError`: インデックスが範囲外の場合

#### `render(debug: bool = False) -> np.ndarray`

メイクを適用した画像をレンダリングします。結果はキャッシュされ、メイク設定の変更時に自動で無効化されます。

**引数:**
- `debug`: `True` の場合、デバッグ情報を `outputs/debug/` に出力（キャッシュ無効）

**戻り値:**
- `np.ndarray`: RGB形式の画像配列 (H, W, 3)

**デバッグモード:**
`debug=True` を指定すると、以下の情報が `outputs/debug/` に保存されます：
- 元画像
- バウンディングボックス描画
- ランドマーク描画
- ハードマスク / ソフトマスクのヒートマップ
- チーク3Dマスク可視化（3Dランドマーク使用時）
- UV座標の可視化
- 結果画像と差分ヒートマップ

#### `save(path: str, debug: bool = False) -> None`

レンダリング結果を保存します。

**引数:**
- `path`: 保存先のファイルパス
- `debug`: デバッグモードでレンダリングするか

#### `show(window_name: str = 'preview', debug: bool = False, wait_key: int = 0) -> None`

レンダリング結果をOpenCVウィンドウで表示します。

**引数:**
- `window_name`: ウィンドウ名
- `debug`: デバッグモードでレンダリングするか
- `wait_key`: `cv2.waitKey` に渡す値（0で無限待機）

**注意:** GUI環境が必要です。WSLやヘッドレス環境では動作しない場合があります。

#### `invalidate_cache() -> None`

レンダリングキャッシュを無効化します。Faceのメイク設定変更時に自動で呼ばれます。

### 使用例

```python
# 顔未検出時は face_count == 0 となる（例外は送出されない）
ctx = processor.open_path("input.jpg")

face = ctx.faces[0]
face.lip(color=(255, 100, 100), opacity=0.5)

result = ctx.render()
ctx.save("output.png")

# デバッグモードでレンダリング
result_debug = ctx.render(debug=True)
# outputs/debug/ にデバッグ情報が保存される
```

---

## Face

検出された顔を表すクラス。Fluent API（メソッドチェーン）でメイクを適用します。

### プロパティ

| プロパティ | 型 | 説明 |
|-----------|-----|------|
| `face_id` | `int` | 顔のID（0始まり） |
| `bbox` | `FaceBBox` | バウンディングボックス |
| `landmarks` | `FaceLandmarks` | ランドマーク（478点） |
| `makeup_layers` | `list[MakeupLayer]` | 設定済みメイクレイヤーのコピー |

### メイクアップメソッド

すべてのメソッドは `self` を返すため、メソッドチェーンが可能です。
各メソッドは `params` オブジェクトまたはキーワード引数のいずれかで指定できます。`params` が指定された場合、キーワード引数は無視されます。

#### `lip(params=None, *, color, opacity=0.5, blur_strength=0.5) -> Face`

口紅を適用します。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `params` | `LipsParams` | `None` | 詳細パラメータ（指定時は他の引数を上書き） |
| `color` | `tuple[int, int, int]` | 必須 | RGB色 (0-255) |
| `opacity` | `float` | `0.5` | 透明度 (0.0-1.0) |
| `blur_strength` | `float` | `0.5` | エッジ減衰（[PartParams](#partparams基底クラス)参照） |

---

#### `cheek(params=None, side='both', *, color, opacity=0.3, center=(0.0, 0.0), radius=(0.5, 0.5), blur_strength=0.5) -> Face`

チーク（ブラッシュ）を適用します。3Dランドマークが利用可能な場合、顔の曲面に沿った楕円マスクが生成されます。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `params` | `CheekParams` | `None` | 詳細パラメータ（指定時は他の引数を上書き） |
| `side` | `'left' \| 'right' \| 'both'` | `'both'` | 適用する側 |
| `color` | `tuple[int, int, int]` | 必須 | RGB色 (0-255) |
| `opacity` | `float` | `0.3` | 透明度 (0.0-1.0) |
| `center` | `tuple[float, float]` | `(0.0, 0.0)` | 楕円中心のオフセット (-1.0〜1.0) |
| `radius` | `tuple[float, float]` | `(0.5, 0.5)` | 楕円の半径 (0.0〜1.0) |
| `blur_strength` | `float` | `0.5` | エッジ減衰（[PartParams](#partparams基底クラス)参照） |

**使用例:**
```python
# 基本的な使用
face.cheek(color=(255, 180, 180), opacity=0.3)

# 左頬のみ、位置調整
face.cheek(color=(255, 180, 180), opacity=0.3, side='left', center=(0.1, -0.1))

# 大きめの丸いチーク
face.cheek(color=(255, 180, 180), opacity=0.3, radius=(0.6, 0.6))
```

---

#### `eyelid(params=None, side='both', *, color, opacity=0.3, blur_strength=0.5) -> Face`

アイシャドウを適用します。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `params` | `EyeShadowParams` | `None` | 詳細パラメータ |
| `side` | `'left' \| 'right' \| 'both'` | `'both'` | 適用する側 |
| `color` | `tuple[int, int, int]` | 必須 | RGB色 (0-255) |
| `opacity` | `float` | `0.3` | 透明度 (0.0-1.0) |
| `blur_strength` | `float` | `0.5` | エッジ減衰（[PartParams](#partparams基底クラス)参照） |

---

#### `eyelid_edge(params=None, side='both', *, color, opacity=0.3, blur_strength=0.5) -> Face`

まぶたの際に色を適用します。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `params` | `EyeShadowParams` | `None` | 詳細パラメータ |
| `side` | `'left' \| 'right' \| 'both'` | `'both'` | 適用する側 |
| `color` | `tuple[int, int, int]` | 必須 | RGB色 (0-255) |
| `opacity` | `float` | `0.3` | 透明度 (0.0-1.0) |
| `blur_strength` | `float` | `0.5` | エッジ減衰（[PartParams](#partparams基底クラス)参照） |

---

#### `eyeliner(params=None, side='both', *, color, opacity=1.0, thickness=2, blur_strength=0.5) -> Face`

アイライナーを適用します。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `params` | `EyeLinerParams` | `None` | 詳細パラメータ |
| `side` | `'left' \| 'right' \| 'both'` | `'both'` | 適用する側 |
| `color` | `tuple[int, int, int]` | 必須 | RGB色 (0-255) |
| `opacity` | `float` | `1.0` | 透明度 (0.0-1.0) |
| `thickness` | `int` | `2` | 線の太さ |
| `blur_strength` | `float` | `0.5` | エッジ減衰（[PartParams](#partparams基底クラス)参照） |

---

#### `eyeiris(params=None, side='both', *, color, opacity=0.5, blur_strength=0.5) -> Face`

カラーコンタクト（虹彩の着色）を適用します。虹彩マスクと目の領域マスクのBitwise ANDにより、虹彩のうち目が開いている部分のみに色が適用されます。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `params` | `EyeIrisParams` | `None` | 詳細パラメータ |
| `side` | `'left' \| 'right' \| 'both'` | `'both'` | 適用する側 |
| `color` | `tuple[int, int, int]` | 必須 | RGB色 (0-255) |
| `opacity` | `float` | `0.5` | 透明度 (0.0-1.0) |
| `blur_strength` | `float` | `0.5` | エッジ減衰（[PartParams](#partparams基底クラス)参照） |

**使用例:**
```python
# ブルー系カラコン
face.eyeiris(color=(100, 150, 255), opacity=0.5)

# 左目のみ
face.eyeiris(color=(100, 150, 255), opacity=0.5, side='left')
```

---

#### `clear() -> Face`

設定されたすべてのメイクレイヤーをクリアします。

---

### バリデーション

範囲外の値を指定すると `ValueError` が発生します：

```python
# ValueError: Color values must be integers 0-255
face.lip(color=(300, 100, 100), opacity=0.5)

# ValueError: Opacity must be 0.0-1.0
face.lip(color=(255, 100, 100), opacity=1.5)

# ValueError: center values must be in range [-1.0, 1.0]
face.cheek(color=(255, 180, 180), opacity=0.3, center=(2.0, 0.0))

# ValueError: radius values must be in range (0.0, 1.0]
face.cheek(color=(255, 180, 180), opacity=0.3, radius=(0.0, 0.5))

# ValueError: Either 'params' or 'color' must be provided
face.lip(opacity=0.5)
```

### メソッドチェーン例

```python
# メソッドチェーンで複数適用
face.lip(color=(255, 100, 100), opacity=0.5) \
    .cheek(color=(255, 180, 180), opacity=0.3) \
    .eyelid(color=(180, 130, 150), opacity=0.4) \
    .eyeliner(color=(0, 0, 0), opacity=1.0) \
    .eyeiris(color=(100, 150, 255), opacity=0.5)

# 左右で異なるチーク
face.cheek(color=(255, 180, 180), opacity=0.3, side='left') \
    .cheek(color=(255, 200, 200), opacity=0.25, side='right')

# レイヤーをクリアして再設定
face.clear()
face.lip(color=(200, 50, 50), opacity=0.6)
```

---

## パラメータクラス

### ColorParams

色指定のデータクラス。

```python
from face_makeup.core.types import ColorParams

color = ColorParams(target_rgb=(255, 100, 100))
```

### PartParams（基底クラス）

すべてのパーツパラメータの基底クラス。

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `color_params` | `ColorParams` | 必須 | 色指定 |
| `opacity` | `float` | 必須 | 透明度 (0.0-1.0) |
| `blur_strength` | `float` | `0.5` | エッジ減衰 (0.0-1.0)。境界でこの割合の不透明度を維持し、中心は1.0 |

### LipsParams

リップの詳細パラメータ。`PartParams`を継承。追加フィールドなし。

### CheekParams

チークの詳細パラメータ。

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `center` | `tuple[float, float]` | `(0.0, 0.0)` | 楕円中心のオフセット (-1.0〜1.0) |
| `radius` | `tuple[float, float]` | `(0.5, 0.5)` | 楕円の半径 (0.0〜1.0、0.0は不可) |

### EyeShadowParams

アイシャドウの詳細パラメータ。`PartParams`を継承。追加フィールドなし。

### EyeLinerParams

アイライナーの詳細パラメータ。

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `thickness` | `int` | `2` | 線の太さ |

### EyeIrisParams

カラーコンタクトの詳細パラメータ。`PartParams`を継承。追加フィールドなし。

### paramsオブジェクトの使用例

```python
from face_makeup.core.types import (
    LipsParams, CheekParams, EyeShadowParams,
    EyeLinerParams, EyeIrisParams, ColorParams
)

# リップ
lip_params = LipsParams(
    color_params=ColorParams(target_rgb=(255, 100, 100)),
    opacity=0.5,
    blur_strength=0.5
)
face.lip(params=lip_params)

# チーク（位置・サイズ指定）
cheek_params = CheekParams(
    color_params=ColorParams(target_rgb=(255, 180, 180)),
    opacity=0.3,
    blur_strength=0.5,
    center=(0.0, 0.0),
    radius=(0.5, 0.5)
)
face.cheek(params=cheek_params, side='both')

# アイライナー（太さ指定）
eyeliner_params = EyeLinerParams(
    color_params=ColorParams(target_rgb=(0, 0, 0)),
    opacity=1.0,
    thickness=3,
    blur_strength=0.5
)
face.eyeliner(params=eyeliner_params)
```

---

## データ型

### FaceBBox

顔のバウンディングボックス。

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `x` | `int` | 左上X座標（ピクセル） |
| `y` | `int` | 左上Y座標（ピクセル） |
| `w` | `int` | 幅（ピクセル） |
| `h` | `int` | 高さ（ピクセル） |
| `score` | `float` | 検出信頼度 |

### FaceLandmarks

顔のランドマーク情報。

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `points` | `np.ndarray` | 正規化された2D座標 (478, 2)、値は0.0〜1.0 |
| `points_3d` | `np.ndarray \| None` | 正規化された3D座標 (478, 3) |

#### `to_image_coords(img_width, img_height) -> np.ndarray`

正規化座標をピクセル座標に変換します。

### MakeupLayer

メイクレイヤーの内部表現。

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `part_name` | `str` | パーツ名（`lips`, `cheek_left`, `eyelid_right` など） |
| `params` | `PartParams` | パラメータオブジェクト |
| `mask` | `np.ndarray \| None` | レンダリング時に生成されるソフトマスク |
