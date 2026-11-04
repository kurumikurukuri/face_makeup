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
- 例外として、まつ毛ラインの位置合わせにのみCPUで動作する小型の機械学習モデル（[EyeContourRefiner](https://github.com/kurumikurukuri/EyeContourRefiner) のまつ毛ライン補正モデル）を使用（ルールベース補正への切り替えも可能。FAQを参照）
- 実際のメイク検討に知見として還元しやすい

---

### 特徴

本ライブラリでは、以下の顔パーツに対してメイクのシミュレーションが可能です。

- リップ：口紅の色および透明度を調整
- チーク：頬に自然なブラッシュを適用
- アイシャドウ：まぶた全体または目の淵（キワ）にカスタマイズ可能なカラーを適用（グリッター/ラメ対応）
- アイライナー：まつ毛ラインに沿ったラインを描画
- カラコン：虹彩にカラーコンタクト効果を適用
- 涙袋：下まぶたの涙袋にハイライトとシャドウを適用（グリッター/ラメ対応）
- ファンデーション：顔全体の肌トーン調整

今後も継続的に部位の拡張を予定しています。

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

インストール時に、まつ毛ライン補正モデル（約15MB）を [EyeContourRefiner](https://github.com/kurumikurukuri/EyeContourRefiner) の `main` から取得します（描画時にはダウンロードしません）。開発時は、`git pull` のたびに最新のモデルへ更新されるよう、clone 後に一度だけ次を実行してください。

```bash
git config core.hooksPath .githooks
```

詳細は [models/README.md](models/README.md) を参照してください。

## クイックスタート

### Pythonコード例

```python
from face_makeup import MakeupProcessor

# プロセッサを初期化（モデルの読み込み）
processor = MakeupProcessor()

# 画像を開く（顔検出・ランドマーク抽出を実行）
# 顔が1枚も検出できない場合は ValueError が送出される
try:
    ctx = processor.open_path("input.jpg")
except ValueError:
    print("顔が検出されませんでした")
else:
    # 検出された顔にメイクを適用
    face = ctx.faces[0]
    face.lip(color=(255, 100, 100), opacity=0.5)                          # リップ
    face.cheek(color=(255, 180, 180), opacity=0.3)                        # チーク
    face.eyelid(color=(180, 130, 150), opacity=0.4)                       # アイシャドウ
    face.eyeliner(color=(0, 0, 0), opacity=0.8, thickness=2)              # アイライナー
    face.eyeiris(color=(100, 150, 255), opacity=0.5)                      # カラコン
    face.tear_bag_highlight(color=(255, 255, 255), opacity=0.3, size=0.8) # 涙袋ハイライト
    face.tear_bag_shadow(color=(150, 110, 90), opacity=0.4, size=0.8)    # 涙袋シャドウ
    face.foundation(color=(230, 200, 190), opacity=0.3)                  # ファンデーション

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

### Q: オフライン環境で使えますか？

**A:** はい。ただし、デフォルトの機械学習補正（`landmark_correction='ml'`）は、インストール時に取得したまつ毛ライン補正モデルが存在しない、または読み込めない場合、目のメイクやファンデーションを含む描画で例外を送出します。face_makeup をインストールし直すか（ソースからなら `pip install -e .`）、ルールベース補正を指定してください。

```python
from face_makeup import MakeupProcessor, ProcessorConfig

processor = MakeupProcessor(ProcessorConfig(landmark_correction='rule'))
```

事前にモデルファイルを保存先へ配置しておけば、機械学習補正もオフラインで使えます（保存先は [models/README.md](models/README.md) を参照）。

## ドキュメント

- [アーキテクチャ](docs/architecture.md) — パイプライン設計・Strategy パターン・2フェーズレンダリング
- [クイックリファレンス](docs/quick-reference.md) — Face API 全メソッドとパラメータ

## テスト画像について

本リポジトリのテストでは、実際の顔写真を使用したテストを実行できます。
テスト画像は `tests/test_images/` ディレクトリに配置してください。

### テスト画像がない場合

テスト画像がなくてもテスト全体が停止することはありません。
画像を必要とするテストは自動的にスキップされ、その他のテストは通常通り実行されます。

```bash
# テスト画像なしでも全テストを実行可能
pytest
```

## ライセンス

本プロジェクトは **MIT License** でライセンスされています。商用利用を含むすべての用途で自由に使用できます。

詳細は [LICENSE](LICENSE) を参照してください。

---

本リポジトリのコードは [Claude Code](https://claude.com/claude-code) との協働により作成されています。
