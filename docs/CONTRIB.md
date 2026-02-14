# 開発ガイド

開発環境セットアップとテスト方法をまとめています。

---

## 環境セットアップ

### 動作環境

| 項目 | 要件 |
|------|------|
| Python | 3.10, 3.11, 3.12 |
| OS | Windows, macOS, Linux |

### インストール

```bash
git clone https://github.com/kurumikurukuri/face_makeup.git
cd face_makeup

# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 開発用依存を含めてインストール
pip install -e ".[dev,app]"
```

---

## コア依存パッケージ

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| `opencv-python` | >=4.7.0 | 画像処理 |
| `numpy` | >=1.25.0 | 数値計算 |
| `mediapipe` | ==0.10.21 | 顔検出・ランドマーク抽出 |
| `pytest` | >=7.0.0 | テストフレームワーク |
| `pytest-cov` | >=4.0.0 | カバレッジ計測 |
| `streamlit` | >=1.24.0 | Web UI |

---

## コマンドリファレンス

### テスト実行

```bash
# 全テストを実行
pytest tests/

# 詳細出力
pytest tests/ -v

# 特定のテストファイル
pytest tests/test_face.py

# カバレッジレポート付き
pytest tests/ --cov=src/face_makeup --cov-report=html
# レポートは htmlcov/index.html に生成されます
```

### Streamlit アプリ起動

```bash
streamlit run apps/streamlit_app.py
```

Windows では `run_streamlit_app.bat` も利用可能。

---

## テスト駆動開発（TDD）

新機能を追加する際は、テストを先に書いてから実装します：

```
1. RED:      テストを書く（失敗する）
2. GREEN:    最小限の実装でテストを通す
3. REFACTOR: コードを改善する
```

部位の追加については [メイク部位追加ガイド](adding-makeup-parts.md) を参照してください。

---

## デバッグ

`render(debug=True)` で詳細情報を `outputs/debug/` に出力：

```python
ctx = processor.open_path("input.jpg")
face = ctx.faces[0]
face.cheek(color=(255, 180, 180), opacity=0.3)
result = ctx.render(debug=True)
```

出力内容：
- マスクのヒートマップ
- チーク座標の可視化
- ランドマーク位置のプロット


---

## ディレクトリ構成

```
makeup/
├── src/face_makeup/            # メインライブラリ
│   ├── processor.py            # MakeupProcessor（顔検出・初期化）
│   ├── face.py                 # Face API（メイク指定）
│   ├── context.py              # ImageContext（画像コンテキスト）
│   ├── rendering_pipeline.py   # レンダリング管理・キャッシュ
│   ├── renderer.py             # 低レベル合成処理
│   ├── io_handler.py           # 画像の保存・表示
│   ├── constants.py            # ランドマークインデックス
│   └── core/                   # コア処理
│       ├── types.py            # パラメータ型定義
│       ├── face_detector.py    # 顔検出
│       ├── landmark_extractor.py # ランドマーク抽出
│       ├── segmenter.py        # マスク生成
│       ├── makeup_blender.py   # メイクブレンド
│       └── debug_visualizer.py # デバッグ可視化
├── apps/
│   └── streamlit_app.py        # Streamlit デモ
├── tests/                      # テストスイート
├── docs/                       # ドキュメント
└── pyproject.toml              # パッケージ設定
```

---

## コーディングスタイル

- **型ヒント**: Python 3.10+ の型ヒントを使用
- **Docstring**: Google 形式
- **イミュータビリティ**: オブジェクトの変更は新しいオブジェクトを返す

---

## ライセンス

MIT License。主要依存パッケージのライセンス：

| パッケージ | ライセンス |
|-----------|-----------|
| MediaPipe | Apache 2.0 |
| OpenCV | Apache 2.0 |
| NumPy | BSD-3 |
