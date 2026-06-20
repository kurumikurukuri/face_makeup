# Face Makeup Library

## 概要

顔写真に仮想メイクアップ効果を適用するPythonライブラリです。

## コンセプト

本ライブラリは、「似合うメイクを効率的かつ再現性高く検討したい」という課題を解決することを目的として作成しました。  
従来のメイク探索には、以下のような課題があります。

- 実際に試行錯誤して探す方法
  - 化粧品の購入やタッチアップに金銭的・時間的コストがかかる
  - 短期間にメイクとメイク落としを繰り返す場合、肌への負担がある
- スマホアプリによる加工
  - 処理内容がブラックボックス化されており、パラメータと結果の因果関係が不透明
  - 複数パラメータを組み合わせて検討することが難しい
- 生成AIによる加工
  - 処理が非決定論的である(ComfyUI等のノードベースツールである程度制御可能)
  - ローカル実行には、高性能なGPUや大容量メモリを要する。
  - クラウドAPIでは従量課金が発生しうる。
  - クラウドAPI等で画像を外部サービスに送信する場合、顔情報流出のリスクがある
- ペイントソフトによる加工
  - 柔軟な表現は可能だが、作業工数が大きい
  - 手作業のため再現性に難がある
  - 多数のパラメータ組み合わせを試す用途には不向き

本ライブラリでは、これらの課題に対してルールベースの画像処理によるメイク加工を採用することで対応しています。

- 処理にGPUが不要で、入力パラメータと出力の挙動が明示的
- 実際のメイク検討に知見として還元しやすい

---

### 特徴

本ライブラリでは、以下の顔パーツに対してメイクのシミュレーションが可能です。

- リップ：口紅の色および透明度を調整
- チーク：頬に自然なブラッシュを適用
- アイシャドウ：まぶたにカスタマイズ可能なカラーを適用
- アイライナー：目の輪郭に沿ったラインを描画
- カラコン：虹彩にカラーコンタクト効果を適用

上記以外の部位についても、今後の拡張を予定しています。

## インストール

### 動作環境

| 項目 | 要件 |
|------|------|
| Python | 3.10, 3.11, 3.12 |
| OS | Windows, macOS, Linux |


### インストール手順

```bash
# リポジトリをクローン
git clone https://github.com/kurumikurukuri/face_makeup.git
cd face_makeup
```

#### Poetry（推奨）

```bash
# インストール
poetry install

# 開発用（テスト含む）
poetry install --with dev

# Streamlit Webアプリを使う場合
poetry install --with app

# すべてインストール
poetry install --with dev,app
```

#### pip

```bash
# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# インストール
pip install -e .

# Streamlit Webアプリを使う場合
pip install -e ".[app]"
```

#### uv

```bash
# インストール
uv pip install -e .

# Streamlit Webアプリを使う場合
uv pip install -e ".[app]"
```

## クイックスタート

### Pythonコード例

```python
from face_makeup import MakeupProcessor

# プロセッサを初期化（モデルの読み込み）
processor = MakeupProcessor()

# 画像を開く（顔検出・ランドマーク抽出を実行）
ctx = processor.open_path("input.jpg")

# 顔が検出されたか確認
if ctx.face_count == 0:
    print("顔が検出されませんでした")
else:
    # 検出された顔にメイクを適用
    face = ctx.faces[0]
    face.lip(color=(255, 100, 100), opacity=0.5)    # リップ
    face.cheek(color=(255, 180, 180), opacity=0.3)  # チーク
    face.eyelid(color=(180, 130, 150), opacity=0.4) # アイシャドウ

    # レンダリングして保存
    result = ctx.render()
    ctx.save("output.png")
```

### Streamlit Webアプリ（デモ）

ブラウザ上でメイクアップを試せるデモアプリを提供しています。

```bash
streamlit run apps/streamlit_app.py
```

ブラウザで `http://localhost:8501` を開くと利用できます。

**機能:**
- 画像のアップロード
- カラーピッカーでメイク色を選択
- スライダーで透明度を調整
- 結果画像のダウンロード

## FAQ

### Q: どのような画像形式に対応していますか？

**A:** JPEG、PNG、WebPなどの一般的な画像形式に対応しています。

### Q: `ctx.show()` が動作しません

**A:** `show()` はOpenCVのGUIウィンドウを使用するため、WSLやヘッドレス環境では動作しません。代わりに `ctx.save()` で保存してください。

### Q: 顔が検出されない場合は？

**A:** 以下を確認してください：
- 画像内に顔が含まれていること
- 顔が十分な大きさで写っていること（画像の10%以上推奨）
- 顔が極端に傾いていないこと

## ドキュメント

リリース時に追加予定

## ライセンス

本プロジェクトは **MIT License** でライセンスされています。商用利用を含むすべての用途で自由に使用できます。

詳細は [LICENSE](LICENSE) を参照してください。

---

本リポジトリは [Claude Code](https://claude.com/claude-code) により生成されています。
