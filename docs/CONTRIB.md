# コントリビューションガイド (CONTRIB)

## 開発ワークフロー

本プロジェクトでは **テスト駆動開発（TDD）** を採用しています。

```
RED → GREEN → REFACTOR

1. RED:      テストを書く（失敗する）
2. GREEN:    最小限の実装でテストを通す
3. REFACTOR: コードを改善する
```

詳細は [メイク部位追加ガイド](adding-makeup-parts.md) を参照してください。

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
```

#### Poetry（推奨）

```bash
# コア + 開発用依存
poetry install --with dev

# Streamlit アプリも含む場合
poetry install --with dev,app

# シェルを起動
poetry shell
```

#### pip

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# コアライブラリ
pip install -e .

# 開発用依存を含む
pip install -e ".[dev]"

# Streamlit アプリも含む場合
pip install -e ".[app]"
```

#### uv

```bash
uv pip install -e .
uv pip install -e ".[dev]"
uv pip install -e ".[app]"
```

---

## 依存パッケージ

*ソース: `pyproject.toml`*

### コア依存

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| `opencv-python` | >=4.7.0 | 画像処理 |
| `numpy` | >=1.25.0 | 数値計算 |
| `mediapipe` | ==0.10.21 | 顔検出・ランドマーク抽出 |

### 開発用依存 (`[dev]`)

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| `pytest` | >=7.0.0 | テストフレームワーク |
| `pytest-cov` | >=4.0.0 | カバレッジ計測 |

### アプリ依存 (`[app]`)

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| `streamlit` | >=1.24.0 | Web UI |
| `requests` | >=2.28.0 | HTTP クライアント |
| `pillow-heif` | >=0.13.0 | HEIF/HEIC 画像対応 |

---

## コマンドリファレンス

### テスト

```bash
# 全テストを実行
pytest tests/

# 詳細出力
pytest tests/ -v

# 特定のテストファイル
pytest tests/test_face.py

# 特定のテストクラス
pytest tests/test_face.py::TestLipMethod

# カバレッジレポート付き（ターミナル）
pytest tests/ --cov=src/face_makeup

# カバレッジレポート付き（HTML）
pytest tests/ --cov=src/face_makeup --cov-report=html
```

### Streamlit アプリ起動

```bash
streamlit run apps/streamlit_app.py
```

Windows の場合はバッチファイルも利用可能:

```bash
"run streamlit_app.bat"
```

### パッケージビルド

```bash
pip install build
python -m build
```

---

## テスト

### テストファイル構成

| ファイル | 対象 |
|----------|------|
| `test_processor.py` | プロセッサの初期化、画像オープン |
| `test_context.py` | ImageContext 操作 |
| `test_face.py` | Face API、レイヤー、色/透明度バリデーション |
| `test_makeup_blender.py` | MakeupBlender クラス、HSV ブレンド |
| `test_ellipse_mask_debug_info.py` | 楕円マスクのデバッグ情報 |
| `test_integration.py` | エンドツーエンドワークフロー |
| `test_edge_cases.py` | エッジケース、境界値 |
| `test_renderer.py` | レンダラー処理 |

### pytest 設定

*ソース: `pyproject.toml` の `[tool.pytest.ini_options]`*

```toml
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

### カバレッジ目標

プルリクエストでは **80% 以上** のテストカバレッジを維持してください。

---

## コーディング規約

- **Python**: 3.10+ の型ヒントを使用
- **Docstring**: Google 形式
- **フォーマッタ**: black
- **リンター**: ruff
- **イミュータビリティ**: オブジェクトの変更は新しいオブジェクトを返す
- **関数サイズ**: 50 行以下を推奨
- **ファイルサイズ**: 800 行以下を推奨

---

## プルリクエスト手順

1. フィーチャーブランチを作成
2. TDD ワークフローでテストと実装を追加
3. `pytest tests/` が全て通ることを確認
4. カバレッジ 80% 以上を維持
5. PR を作成

### コミットメッセージ形式

```
<type>: <description>
```

`type`: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

---

## ディレクトリ構成

```
makeup/
├── src/face_makeup/              # メインライブラリ
│   ├── __init__.py              # パッケージエントリポイント
│   ├── processor.py             # MakeupProcessor（ステートレス）
│   ├── face.py                  # Face クラスと Fluent API
│   ├── context.py               # ImageContext（画像ごとの状態管理）
│   ├── renderer.py              # 内部レンダリングエンジン
│   ├── constants.py             # 顔パーツのランドマークインデックス
│   └── core/                    # コア処理モジュール
│       ├── types.py             # データ構造
│       ├── face_detector.py     # MediaPipe 顔検出ラッパー
│       ├── landmark_extractor.py # MediaPipe Face Mesh
│       ├── segmenter.py         # マスク生成
│       ├── makeup_blender.py    # HSV ベースの色適用
│       └── debug_visualizer.py  # デバッグ可視化
├── apps/
│   └── streamlit_app.py         # Streamlit デモアプリ
├── tests/                       # テストスイート
├── docs/                        # ドキュメント
├── pyproject.toml              # パッケージメタデータ
└── requirements.txt            # 依存パッケージリスト
```

---

## デバッグ

`render(debug=True)` を使用すると、`outputs/debug/` にデバッグ情報が出力されます。

```python
ctx = processor.open_path("input.jpg")
face = ctx.faces[0]
face.cheek(color=(255, 180, 180), opacity=0.3)
result = ctx.render(debug=True)
```

出力内容:
- マスクのヒートマップ画像
- UV 座標の可視化（接線空間）
- ランドマーク位置のプロット
- 楕円パラメータの詳細

---

## ライセンス

MIT License。主要依存パッケージのライセンス:

| パッケージ | ライセンス |
|-----------|-----------|
| MediaPipe | Apache 2.0 |
| OpenCV | Apache 2.0 |
| NumPy | BSD-3 |
