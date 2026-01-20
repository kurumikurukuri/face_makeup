# Face Makeup Library（要件定義書）

本書は Claude Code の Plan Mode に入力する要件定義書である。
実装の詳細ではなく、実装すべき機能・設計方針・スコープを定義する。

---

## 1. 背景・目的

スマートフォンや PC 上で、ユーザーが **自身の顔写真にメイクをシミュレーション**できる処理を提供する。
本ライブラリは **生成系 AI を使用せず**、顔解析と画像処理を組み合わせて自然なメイクシミュレーションを実現することを目的とする。

---

## 2. 対象範囲

### 入力画像

- PNG / JPEG
- RGB
- uint8
- 推奨 256×256 ～ 2048×2048
- 単一顔を主対象（複数顔対応可能な設計）

### 対象顔部位

- lips
- eyelid_left / eyelid_right
- eyeliner_left / eyeliner_right
- cheek_left / cheek_right

### 対象外

- 動画・リアルタイム処理
- メイク生成AI
- 似合い度評価

---

## 3. システム構成

ライブラリは責務ごとに以下のモジュールへ分割する。

```
MakeupProcessor
    ↓
ImageContext
    ↓
Face
    ↓
MakeupLayer
    ↓
Renderer
```

各画像に対して `ImageContext` を生成し、各顔ごとに `Face` オブジェクトを持つ。

メイクは `MakeupLayer` を追加することで構築する。

---

## 4. 顔解析

### 顔検出

MediaPipe Face Detection を使用する。

出力

```
FaceBBox
```

### ランドマーク

MediaPipe Face Mesh を使用する。

ランドマークは Renderer・Segmenter が共通利用する。

---

## 5. セグメンテーション

ランドマークから部位マスクを生成する。

初期対応

- polygon
- convex hull
- curve

必要に応じて除外マスクを追加できる設計とする。

---

## 6. メイク処理

各部位は独立した Layer として実装する。

初期対応

- Lip
- Cheek
- EyeShadow
- EyeLiner

各 Layer は

- マスク
- 色
- opacity

を保持する。

Renderer は Layer を順番に画像へ適用する。

---

## 7. 色ブレンド

RGB ベースの色ブレンドを採用する。

各 Layer は

- target color
- opacity

を指定できる。

---

## 8. API

エントリポイント

```python
processor = MakeupProcessor()

ctx = processor.open(image)

face = ctx.faces[0]

face.lip(...)
face.cheek(...)
face.eyelid(...)
face.eyeliner(...)

result = ctx.render()
```

Fluent API によりメイクを組み合わせられる設計とする。

---

## 9. デバッグ

各処理段階を可視化できる。

例

- bbox
- landmarks
- mask
- blend
- result
- diff

---

## 10. アプリケーション

- **Streamlit** にてデモアプリを作成する。

---

## 11. 非機能要件

### 動作環境

- Python 3.10～
- OpenCV
- NumPy
- MediaPipe

### 品質

- CPU 動作可能
- 再利用可能な Processor
- UI 非依存
- モジュール単位でテスト可能

---

## 12. 開発フェーズ

### Phase 1

- Face Detection
- Face Mesh
- Lip
- Eyelid
- Eyeliner
- Cheek

### Phase 2

- マスク品質改善
- デバッグ可視化拡充
- 色ブレンド改善

### Phase 3

- 顔輪郭
- 鼻
- 眉
- 複数顔対応改善

---

## 13. 設計思想

### レイヤードアーキテクチャ

画像管理・顔状態・描画処理を分離する。

### Fluent API

メイク追加を自然に記述できる API を提供する。

### Renderer の独立

描画ロジックは Renderer に集約し、Layer は部位情報のみ保持する。

---

## 14. 用語

| 用語 | 定義 |
|------|------|
| Face | 1人分の顔編集状態 |
| Layer | 1つのメイク効果 |
| Renderer | Layer を画像へ合成する描画エンジン |
| ImageContext | 1画像分の処理コンテキスト |
| opacity | ブレンド強度 (0～1) |

---

## 15. 使用ライブラリ

- OpenCV
- NumPy
- MediaPipe
- Pillow