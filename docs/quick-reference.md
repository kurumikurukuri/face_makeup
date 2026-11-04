# クイックリファレンス

`face_makeup` の全公開 API とパラメータ一覧です。内部設計は [architecture.md](architecture.md) を参照してください。

## 基本フロー

```python
from face_makeup import MakeupProcessor

processor = MakeupProcessor()          # モデルロード(重い処理は1回だけ)
ctx = processor.open_path("input.jpg")  # 顔検出 + ランドマーク抽出
face = ctx.faces[0]
face.lip(color=(255, 100, 100), opacity=0.5)
result = ctx.render()                   # numpy ndarray (H, W, 3) RGB uint8
ctx.save("output.png")
```

顔が1枚も検出できない場合、`open()` / `open_path()` は `ValueError("No face detected")` を送出します(空リストは返しません)。呼び出し側で `try/except ValueError` を使ってハンドリングしてください。

```python
try:
    ctx = processor.open_path("input.jpg")
except ValueError:
    print("顔が検出されませんでした")
```

---

## `MakeupProcessor`

```python
MakeupProcessor(config: ProcessorConfig | None = None)
```

| メソッド/プロパティ | 説明 |
|---|---|
| `config` | 現在の `ProcessorConfig` を取得(読み取り専用) |
| `open(image: np.ndarray) -> ImageContext` | RGB uint8 の `(H, W, 3)` ndarray を受け取り、顔検出・ランドマーク抽出を行って `ImageContext` を返す。顔未検出時は `ValueError`。processor が close 済みなら `RuntimeError` |
| `open_path(image_source: str) -> ImageContext` | 画像ファイル(JPEG/PNG/WebP等)を読み込んで `open()` と同様に処理する。ファイル未存在は `FileNotFoundError` |
| `close() -> None` | 検出器・ランドマーク抽出器・レンダリングパイプライン(髪セグメンタ含む)のリソースを解放。冪等(複数回呼び出し可)。close 後の `open()`/`open_path()` は `RuntimeError` |

### `ProcessorConfig`

| フィールド | 型 / デフォルト | 説明 |
|---|---|---|
| `min_detection_confidence` | `float = 0.5`(0.0–1.0) | 顔検出の最小信頼度 |
| `min_tracking_confidence` | `float = 0.5`(0.0–1.0) | ランドマーク追跡の最小信頼度 |
| `refine_landmarks` | `bool = True` | ランドマーク精緻化を行うか |
| `max_num_faces` | `int = 10`(1以上) | 検出する最大顔数 |
| `model_dir` | `str \| None = None` | 髪セグメンテーションモデルの保存先ディレクトリ。見つからないモデルはここにダウンロードされる(まつ毛ライン補正モデルはパッケージと一緒にインストールされるので対象外)。`None` の場合は環境変数→ソースチェックアウト→プラットフォームキャッシュの優先順で自動解決 |
| `landmark_correction` | `'ml' \| 'rule' = 'ml'` | まつ毛ラインのランドマーク補正。顔ごとに1回補正し、アイライナー、アイシャドウ、涙袋、およびファンデーション(目元の除外範囲)が同じまつ毛ラインを使う。`'ml'` は [EyeContourRefiner](https://github.com/kurumikurukuri/EyeContourRefiner) のまつ毛ライン補正モデルによる機械学習補正。モデルはインストール時に取得し、描画時にはダウンロードしない。モデルがない、または読み込めないときは `RuntimeError` / `ModelContractError` を送出する(フォールバックしない)。`'rule'` は2階微分ゼロ交差検出によるルールベース補正。目のメイクやファンデーションを含まない顔では補正しない |

範囲外の値を渡すと `__post_init__` で `ValueError` が送出されます。

---

## `ImageContext`

`processor.open()` / `open_path()` の戻り値。1枚の画像と検出済み `Face` 群を保持します。

| メソッド/プロパティ | 説明 |
|---|---|
| `original_image` | 元画像のコピー(ndarray) |
| `faces` | 検出された `Face` のリスト(浅いコピー) |
| `face_count` | 検出された顔の数 |
| `face(index: int = 0) -> Face` | インデックス指定で `Face` を取得。範囲外は `IndexError` |
| `render(debug: bool = False) -> np.ndarray` | 全 `Face` のメイクを合成した最終画像を返す。メイク内容が変わらない限り結果をキャッシュする |
| `save(path: str, debug: bool = False) -> None` | `render()` の結果をファイルに保存 |
| `show(window_name: str = "preview", debug: bool = False, wait_key: int = 0) -> None` | OpenCV のウィンドウでプレビュー表示(GUI 非対応環境では動作しない。`save()` を使うこと) |

---

## `Face` — メイク適用 API

`ctx.faces[i]` から取得する `Face` はメソッドチェーン(fluent API)でメイクを蓄積します。全メソッドはキーワード専用引数(`*` 以降)で、`color` は `(R, G, B)` の `int` タプル(各要素 0–255)。共通パラメータ `opacity` と `softness` は 0.0–1.0 の範囲外だと `ValueError`。

### `lip(*, color, opacity=0.5, softness=0.5) -> Face`
口紅。

### `cheek(*, side='both', color, opacity=0.3, center=(0.0, 0.0), radius=(0.5, 0.5), softness=0.6) -> Face`
チーク。`side` は `'left' | 'right' | 'both'`。`center`/`radius` は頬正規化座標での楕円中心・半径。

### `eyelid(*, side='both', zone='full', color, opacity=0.3, softness=0.4, size=1.0, glitter=0.0) -> Face`
アイシャドウ(まつ毛のキワ〜クレースラインの帯)。
- `zone`: `'full' | 'outer' | 'center' | 'inner' | 'inner_center' | 'outer_center'` — 塗る水平領域。
- `size`(0.0–1.0): 帯の高さ。まつ毛のキワ(0.0)からクレースライン(1.0)へ補間。小さいほど細い帯。複数回呼んで色を重ねられる(例: 濃い縁 + 薄い全体)。
- `glitter`(0.0–1.0): ラメ効果の強さ。0.0はマット。

### `eyeliner(*, side='both', color, opacity=1.0, thickness=2, softness=0.5) -> Face`
アイライナー。`thickness` は 1 以上の整数(線幅)。線は補正後のまつ毛ラインに沿って描く(補正方法は `ProcessorConfig.landmark_correction` で指定)。目尻のまつ毛ラインのランドマーク1408/1608は描画の対象外(上ラインは1407/1607から目尻へ直接つなぐ)なので、0.1.0 と比べて目尻付近の描画がわずかに変わる。

### `eyeiris(*, side='both', color, opacity=0.5, softness=0.1, hue_shift=False) -> Face`
カラーコンタクト。`hue_shift=True` で H チャンネルのみをシフトし元の S/V を保持(`False` は従来の HSV 置換ブレンド)。

### `tear_bag_highlight(*, side='both', color=(255,255,255), opacity=0.3, softness=0.5, size=1.0, glitter=0.0) -> Face`
涙袋ハイライト(下まぶたの膨らみを screen ブレンドで明るくする)。`size`(0.0–1.0): 0.0で涙袋の外縁がまつ毛のキワに収縮(涙袋が見えなくなる)、1.0で自然な涙袋の輪郭。

### `tear_bag_shadow(*, side='both', color=(150,110,90), opacity=0.4, thickness=2, softness=0.5, size=1.0) -> Face`
涙袋シャドウ(涙袋外縁をなぞる開いたポリライン)。`size` は `tear_bag_highlight` と揃えるとハイライトとシャドウが連動する。

### `foundation(*, color, opacity=0.3, softness=0.5) -> Face`
ファンデーション(顔全体の肌トーン調整)。目・唇領域は自動的に除外される。

### `clear() -> Face`
そのFaceに追加した全メイク層をクリアする。

### 共通プロパティ

| プロパティ | 説明 |
|---|---|
| `face_id` | 画像内での顔の識別子(0始まり) |
| `bbox` | 検出されたバウンディングボックス(`FaceBBox`) |
| `landmarks` | 顔ランドマーク(`FaceLandmarks`) |
| `makeup_passes` | 現在蓄積されている `MakeupPass` のリスト(コピー) |

### バリデーション早見表

| パラメータ | 範囲 | 違反時 |
|---|---|---|
| `color` | `(R, G, B)`、各要素 int 0–255 | `ValueError` |
| `opacity` | float 0.0–1.0 | `ValueError` |
| `softness` | float 0.0–1.0 | `ValueError` |
| `glitter` | float 0.0–1.0 | `ValueError` |
| `size`(`eyelid`/`tear_bag_highlight`/`tear_bag_shadow`) | float 0.0–1.0 | `ValueError` |
| `thickness`(`eyeliner`/`tear_bag_shadow`) | int 1 以上 | `ValueError` |
| `center`(`cheek`) | 各軸 float -1.0–1.0 | `ValueError` |
| `radius`(`cheek`) | 各軸 float 0.0–1.0(0.0 不可) | `ValueError` |
| `side` | `'left' \| 'right' \| 'both'` | `ValueError` |
| `zone`(`eyelid` のみ) | `'full' \| 'outer' \| 'center' \| 'inner' \| 'inner_center' \| 'outer_center'` | `ValueError` |
