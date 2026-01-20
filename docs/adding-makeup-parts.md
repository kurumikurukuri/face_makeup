# メイク部位追加ガイド

このドキュメントでは、Face Makeup ライブラリに新しいメイク部位を追加する手順を説明します。

## TDD ワークフロー

本ライブラリでは **テスト駆動開発（TDD）** を採用しています。

```
RED → GREEN → REFACTOR

1. RED:      テストを書く（失敗する）
2. GREEN:   最小限の実装でテストを通す
3. REFACTOR: コードを改善する
```

---

## 概要

新しいメイク部位を追加するには、以下の順序で進めます：

| フェーズ | # | ファイル | 変更内容 |
|---------|---|----------|----------|
| **RED** | 1 | `tests/test_face.py` | 単体テストを先に書く |
| **GREEN** | 2 | `constants.py` | ランドマーク索引を定義 |
| | 3 | `constants.py` | `PART_LANDMARK_MAP` に登録 |
| | 4 | `core/types.py` | パラメータクラスを作成 |
| | 4.5 | `core/types.py` | `PARAMS_PRIORITY` に登録 |
| | 5 | `face.py` | `MakeupLayer` サブクラスを作成 |
| | 6 | `face.py` | `Face` クラスにメソッドを追加 |
| | 7 | `core/makeup_blender.py` | ブレンドメソッドを追加 |
| | 8 | `renderer.py` | レンダラーに型判定を追加 |
| | 9 | `tests/test_integration.py` | 統合テストを追加 |
| **REFACTOR** | 10 | - | コードの改善・リファクタリング |

---

## 手順詳細

以下、**眉毛（Eyebrow）** を追加する例で説明します。

### 1. テストを書く（RED フェーズ）

**ファイル:** `tests/test_face.py`

まず、期待する動作をテストとして記述します。この時点ではテストは失敗します。

```python
import pytest
from face_makeup.face import Face, EyebrowLayer
from face_makeup.core.types import EyebrowParams, ColorParams


class TestEyebrowMethod:
    """eyebrow() メソッドのテスト。"""

    def test_eyebrow_with_color(self, sample_face):
        """色指定で眉毛メイクを適用できる。"""
        face = sample_face.eyebrow(color=(80, 60, 40), opacity=0.5)

        assert len(face._makeup_layers) == 2  # 左右
        assert all(isinstance(layer, EyebrowLayer) for layer in face._makeup_layers)

    def test_eyebrow_left_only(self, sample_face):
        """片側のみ適用できる。"""
        face = sample_face.eyebrow(color=(80, 60, 40), side='left')

        assert len(face._makeup_layers) == 1
        assert face._makeup_layers[0].part_name == 'eyebrow_left'

    def test_eyebrow_right_only(self, sample_face):
        """右側のみ適用できる。"""
        face = sample_face.eyebrow(color=(80, 60, 40), side='right')

        assert len(face._makeup_layers) == 1
        assert face._makeup_layers[0].part_name == 'eyebrow_right'

    def test_eyebrow_requires_color_or_params(self, sample_face):
        """color も params も未指定の場合はエラー。"""
        with pytest.raises(ValueError, match="Either 'params' or 'color'"):
            sample_face.eyebrow()

    def test_eyebrow_with_params(self, sample_face):
        """EyebrowParams で詳細設定できる。"""
        params = EyebrowParams(
            color_params=ColorParams(target_rgb=(80, 60, 40)),
            opacity=0.5
        )
        face = sample_face.eyebrow(params=params)

        assert len(face._makeup_layers) == 2

    def test_eyebrow_method_chaining(self, sample_face):
        """メソッドチェーンで他のメイクと組み合わせられる。"""
        face = sample_face.eyebrow(color=(80, 60, 40)).lip(color=(200, 100, 100))

        assert len(face._makeup_layers) == 3  # 眉毛2 + リップ1
```

テストを実行して失敗することを確認：

```bash
pytest tests/test_face.py::TestEyebrowMethod -v
# → ImportError や AttributeError で失敗する（期待通り）
```

---

### 2. ランドマーク索引の定義（GREEN フェーズ開始）

**ファイル:** `src/face_makeup/constants.py`

MediaPipe Face Mesh の 478 ポイントから、対象部位のランドマーク索引を特定して定義します。

```python
# 眉毛のランドマーク索引
EYEBROW_LEFT_INDICES = [55, 107, 66, 52, 53, 46, 124, 35, 111, 117, 118, 119, 120, 121, 128, 245]
EYEBROW_RIGHT_INDICES = [285, 336, 296, 282, 283, 276, 353, 265, 340, 346, 347, 348, 349, 350, 357, 465]
```

**Tips:**
- ランドマーク索引の確認には `debug=True` でレンダリングするか、[MediaPipe Face Mesh ドキュメント](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) を参照
- 仮想ランドマーク（複数点の中点）を使う場合は `("mid", idx1, idx2, ...)` 形式で定義

### 3. PART_LANDMARK_MAP への登録

**ファイル:** `src/face_makeup/constants.py`

`PART_LANDMARK_MAP` に新しい部位を登録します。

```python
PART_LANDMARK_MAP = {
    # 既存の定義...
    'cheek_left': (CHEEK_LEFT_INDICES, "convex"),
    'lips': (LIPS_INDICES, 'polygon'),
    # ...

    # 新規追加
    'eyebrow_left': (EYEBROW_LEFT_INDICES, 'polygon'),
    'eyebrow_right': (EYEBROW_RIGHT_INDICES, 'polygon'),
}
```

**セグメンテーションタイプ:**

| タイプ | 説明 | 用途 |
|--------|------|------|
| `polygon` | ポイントを順序通り結ぶ多角形 | リップ、アイシャドウ、眉毛 |
| `convex` | 凸包（Convex Hull） | チーク |
| `curve` | 曲線（線の太さ指定可） | アイライナー |
| `ellipse` | 楕円フィッティング | カラーコンタクト（虹彩） |

### 4. パラメータクラスの作成

**ファイル:** `src/face_makeup/core/types.py`

新しい `PartParams` サブクラスを定義します。

```python
@dataclass
class EyebrowParams(PartParams):
    """眉毛メイクのパラメータ。

    Attributes:
        color_params: 色設定
        opacity: 不透明度 (0.0-1.0)
    """
    pass  # 基本機能のみの場合は pass
```

**特殊機能を追加する場合:**

```python
@dataclass
class EyebrowParams(PartParams):
    """眉毛メイクのパラメータ。"""
    thickness: float = 1.0  # 眉の太さ調整
    gradient: bool = False  # グラデーション有効化
```

### 4.5. PARAMS_PRIORITY への登録

**ファイル:** `src/face_makeup/core/types.py`

新しいパラメータクラスを `PARAMS_PRIORITY` に登録します。この登録を忘れると、マスク重複解決が正しく動作しません。

```python
PARAMS_PRIORITY: Dict["Type[PartParams]", int] = {
    CheekParams: 0,
    LipsParams: 10,
    EyeShadowParams: 20,
    EyeLinerParams: 30,
    EyeIrisParams: 40,
    EyebrowParams: 50,     # 新規追加（適切な優先度を設定）
}
```

**優先度のガイドライン:**
- 値が大きいほど上のレイヤーに描画される
- 重複する可能性がある他のパーツとの優先関係を考慮して設定

### 5. MakeupLayer サブクラスの作成

**ファイル:** `src/face_makeup/face.py`

レイヤークラスを定義します。

```python
@dataclass
class EyebrowLayer(MakeupLayer):
    """眉毛メイクレイヤー。"""
    pass
```

### 6. Face クラスにメソッドを追加

**ファイル:** `src/face_makeup/face.py`

ユーザーが呼び出す API メソッドを追加します。

```python
def eyebrow(
    self,
    params: Optional[EyebrowParams] = None,
    side: Literal['left', 'right', 'both'] = 'both',
    *,
    color: Optional[Tuple[int, int, int]] = None,
    opacity: float = 0.4
) -> Face:
    """眉毛メイクを適用する。

    Args:
        params: EyebrowParams インスタンス（詳細設定用）
        side: 適用する側 ('left', 'right', 'both')
        color: RGB色 (0-255)。params 未指定時は必須
        opacity: 不透明度 (0.0-1.0)

    Returns:
        self（メソッドチェーン用）

    Example:
        >>> face.eyebrow(color=(80, 60, 40), opacity=0.5)
    """
    if params is None:
        if color is None:
            raise ValueError("Either 'params' or 'color' must be provided")
        self._validate_color(color)
        self._validate_opacity(opacity)
        params = EyebrowParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity
        )

    if side in ('left', 'both'):
        self._makeup_layers.append(EyebrowLayer(
            part_name='eyebrow_left',
            params=params
        ))
    if side in ('right', 'both'):
        self._makeup_layers.append(EyebrowLayer(
            part_name='eyebrow_right',
            params=params
        ))

    self._notify_change()
    return self
```

### 7. ブレンドメソッドの追加

**ファイル:** `src/face_makeup/core/makeup_blender.py`

`MakeupBlender` クラスにブレンドメソッドを追加します。

```python
def blend_eyebrow(
    self,
    image: np.ndarray,
    mask: np.ndarray,
    params: EyebrowParams
) -> np.ndarray:
    """眉毛メイクを適用する。

    Args:
        image: 入力画像 (RGB)
        mask: マスク画像 (float32 0.0-1.0)
        params: 眉毛パラメータ

    Returns:
        メイク適用後の画像
    """
    return self.blend(image, mask, params.color_params, params.opacity, params.blur_strength)
```

**特殊なブレンド処理が必要な場合:**

```python
def blend_eyebrow(
    self,
    image: np.ndarray,
    mask: np.ndarray,
    params: EyebrowParams
) -> np.ndarray:
    """眉毛メイクを適用する（RGB画像、float32マスク）。"""
    # 前処理（例: マスクのぼかし調整）
    if params.thickness != 1.0:
        kernel_size = int(5 * params.thickness)
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    # 基本ブレンド実行
    return self.blend(image, mask, params.color_params, params.opacity)
```

### 8. レンダラーへの型判定追加

**ファイル:** `src/face_makeup/renderer.py`

`_apply_layer` メソッドに新しい型の判定を追加します。

```python
def _apply_layer(
    self,
    image: np.ndarray,
    mask: np.ndarray,
    layer: MakeupLayer,
    face: Face
) -> np.ndarray:
    """レイヤーを適用する。"""
    params = layer.params

    if isinstance(params, LipsParams):
        return self._blender.blend_lips(image, mask, params)

    if isinstance(params, CheekParams):
        cheek_side = getattr(layer, 'cheek_side', 'left')
        cheek_mask = self._segmenter.create_cheek_mask(
            image.shape[:2], mask, params,
            landmarks=face.landmarks,
            cheek_side=cheek_side
        )
        return self._blender.blend_cheek(
            image, mask, cheek_mask, params
        )

    if isinstance(params, EyeShadowParams):
        return self._blender.blend_eyeshadow(image, mask, params)

    if isinstance(params, EyeLinerParams):
        return self._blender.blend_eyeliner(image, mask, params)

    # 新規追加
    if isinstance(params, EyebrowParams):
        return self._blender.blend_eyebrow(image, mask, params)

    # Default blending
    return self._blender.blend(
        image, mask, params.color_params, params.opacity
    )
```

**import の追加も忘れずに:**

```python
from face_makeup.core.types import (
    LipsParams,
    CheekParams,
    EyeShadowParams,
    EyeLinerParams,
    EyebrowParams,  # 追加
)
```

### 9. テストを実行して確認（GREEN 達成）

```bash
pytest tests/test_face.py::TestEyebrowMethod -v
# → すべてのテストが PASSED になる
```

### 10. 統合テストの追加

**ファイル:** `tests/test_integration.py`

単体テストに加えて、エンドツーエンドの統合テストを追加します。

#### 10.1 既存テストへの追加

**`TestEndToEndWorkflow.test_full_makeup_workflow`** に新しいメイクを追加：

```python
def test_full_makeup_workflow(self, processor, test_image):
    """Test complete makeup application workflow."""
    ctx = processor.open(test_image)
    face = ctx.faces[0]
    face.lip(color=(255, 100, 100), opacity=0.5)
    face.cheek(color=(255, 180, 180), opacity=0.3)
    face.eyelid(color=(180, 130, 150), opacity=0.3)
    face.eyeliner(color=(0, 0, 0), opacity=0.8)
    face.eyebrow(color=(80, 60, 40), opacity=0.5)  # 新規追加

    result = ctx.render()
    assert result.shape == test_image.shape
```

**`TestFluentAPIChaining.test_chain_all_methods`** にメソッドチェーンを追加：

```python
def test_chain_all_methods(self, processor, test_image):
    """Test chaining all makeup methods."""
    ctx = processor.open(test_image)

    result = (
        ctx.faces[0]
        .lip(color=(255, 100, 100), opacity=0.5)
        .cheek(color=(255, 180, 180), opacity=0.3, side='both')
        .eyelid(color=(180, 130, 150), opacity=0.3, side='both')
        .eyeliner(color=(0, 0, 0), opacity=0.8, side='both')
        .eyebrow(color=(80, 60, 40), opacity=0.5, side='both')  # 新規追加
    )

    assert result is ctx.faces[0]
    assert len(ctx.faces[0].makeup_layers) >= 9  # 数を更新
```

#### 10.2 単独テストの追加

**`TestMakeupLayerCombinations`** クラスに新しいテストを追加：

```python
def test_eyebrow_only(self, processor, test_image):
    """Test eyebrow makeup only."""
    ctx = processor.open(test_image)
    ctx.faces[0].eyebrow(color=(80, 60, 40), opacity=0.5)
    result = ctx.render()
    assert not np.array_equal(result, test_image)
```

#### 10.3 統合テストの実行

```bash
pytest tests/test_integration.py -v
# → すべてのテストが PASSED になる
```

### 11. リファクタリング（REFACTOR フェーズ）

テストが通ったら、コードの品質を改善します：

- 重複コードの抽出
- 命名の改善
- ドキュメントの追加
- パフォーマンスの最適化

リファクタリング後も必ずテストを実行して、動作が壊れていないことを確認します。

---

## チェックリスト

新しいメイク部位を追加する際の確認リスト：

### RED フェーズ
- [ ] テストクラスを作成した
- [ ] 正常系のテストを書いた
- [ ] 異常系のテストを書いた（バリデーションエラー等）
- [ ] メソッドチェーンのテストを書いた
- [ ] テストが失敗することを確認した

### GREEN フェーズ
- [ ] ランドマーク索引が正しい順序で定義されている
- [ ] `PART_LANDMARK_MAP` に登録されている
- [ ] パラメータクラスが `PartParams` を継承している
- [ ] `PARAMS_PRIORITY` に優先度を登録している
- [ ] `MakeupLayer` サブクラスが定義されている
- [ ] `Face` クラスのメソッドがフルエント API に対応（`return self`）
- [ ] ブレンドメソッドが追加されている
- [ ] レンダラーの `isinstance` チェックに追加されている
- [ ] 必要な import が追加されている
- [ ] 単体テストが通ることを確認した
- [ ] 統合テスト (`test_integration.py`) に追加した
  - [ ] `test_full_makeup_workflow` に新メイクを追加
  - [ ] `test_chain_all_methods` にメソッドチェーンを追加
  - [ ] `TestMakeupLayerCombinations` に単独テストを追加
- [ ] 統合テストが通ることを確認した

### REFACTOR フェーズ
- [ ] コードの重複を解消した
- [ ] 命名が適切か確認した
- [ ] テストが引き続き通ることを確認した

---

## トラブルシューティング

### テストが import エラーで失敗する

- 新しいクラスを `__init__.py` でエクスポートしているか確認
- 循環 import が発生していないか確認

### マスクが正しく生成されない

- ランドマーク索引の順序を確認（時計回りまたは反時計回りで一貫させる）
- `debug=True` でレンダリングしてマスク画像を確認

### ブレンド結果が期待と異なる

- `opacity` の範囲を確認（0.0-1.0）
- 色の形式を確認（RGB、0-255）
- セグメンテーションタイプが適切か確認

### リファクタリング後にテストが失敗する

- 変更を小さく分割して、どの変更が問題か特定
- `git diff` で意図しない変更がないか確認
