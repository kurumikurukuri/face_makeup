# 運用マニュアル (RUNBOOK)

## デプロイ手順

### Streamlit アプリ（ローカル）

```bash
# 1. 仮想環境をセットアップ
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 依存をインストール
pip install -e ".[app]"

# 3. アプリを起動
streamlit run apps/streamlit_app.py
```

ブラウザで `http://localhost:8501` にアクセスして利用可能。

#### Windows バッチファイル

```bash
"run streamlit_app.bat"
```

自動で venv 作成、依存インストール、Streamlit 起動を行います。

### Streamlit アプリ（リモート）

```bash
# ポート指定で起動
streamlit run apps/streamlit_app.py --server.port 8501

# ヘッドレスモード（ブラウザを自動で開かない）
streamlit run apps/streamlit_app.py --server.headless true

# 外部アクセスを許可
streamlit run apps/streamlit_app.py --server.address 0.0.0.0
```

### ライブラリとしての利用

```bash
pip install -e .
```

```python
from face_makeup import MakeupProcessor

processor = MakeupProcessor()
ctx = processor.open_path("input.jpg")
face = ctx.faces[0]
face.lip(color=(255, 100, 100), opacity=0.5)
result = ctx.render()
ctx.save("output.png")
```

---

## 環境変数

本プロジェクトでは `.env` ファイルや環境変数による設定は不要です。
すべての設定は Python コード内（`ProcessorConfig`）で行います。

```python
from face_makeup import MakeupProcessor, ProcessorConfig

config = ProcessorConfig(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,
    max_num_faces=10
)
processor = MakeupProcessor(config)
```

---

## モニタリング

### Streamlit アプリ

- Streamlit のログはターミナルに出力されます
- デバッグモードを有効にすると `outputs/debug/` にデバッグ画像が保存されます
- セッション ID がタイムスタンプベースで自動生成されます

### ヘルスチェック

```bash
# Streamlit プロセスが動作しているか確認
# Linux/macOS:
ps aux | grep streamlit

# Windows:
tasklist | findstr streamlit
```

### リソース使用量

- MediaPipe モデルは初回ロード時にメモリを消費（`@st.cache_resource` でキャッシュ）
- 画像サイズに比例してメモリを消費
- URL 入力の場合、最大 10MB の制限あり (`MAX_IMAGE_SIZE`)

---

## よくある問題と対処法

### 顔が検出されない

**症状:** `ValueError` が発生し、「顔が検出されない」旨のメッセージ

**対処:**
1. 画像内に顔が含まれていることを確認
2. 顔が十分な大きさで写っていること（画像の 10% 以上推奨）
3. 顔が極端に傾いていないことを確認
4. `min_detection_confidence` を下げて再試行

### `ctx.show()` が動作しない

**症状:** OpenCV ウィンドウが表示されない

**対処:**
- WSL やヘッドレス環境では `show()` は使用不可
- `ctx.save("output.png")` で保存して確認

### MediaPipe のインストールエラー

**症状:** `pip install mediapipe` が失敗

**対処:**
1. Python バージョンが 3.10〜3.12 であることを確認
2. `mediapipe==0.10.21` を明示的に指定
3. OS が対応していることを確認（Windows, macOS, Linux）

### Streamlit アプリで画像処理が遅い

**症状:** メイク適用に時間がかかる

**対処:**
1. 入力画像のサイズを 2048x2048 以下に縮小
2. 検出する最大顔数を制限（`max_num_faces` を小さくする）
3. デバッグモードを無効にする

### HEIF/HEIC 画像が読み込めない

**症状:** HEIF 形式の画像がアップロードできない

**対処:**
1. `pillow-heif` がインストールされていることを確認
2. `pip install pillow-heif>=0.13.0`

### URL からの画像読み込みエラー

**症状:** URL 入力で「内部ネットワークの URL は許可されていません」

**対処:**
- SSRF 対策により、ローカルネットワークの URL はブロックされます
- パブリックにアクセス可能な URL を使用してください

---

## ロールバック手順

### Git によるロールバック

```bash
# 直前のコミットに戻す
git revert HEAD

# 特定のコミットに戻す
git revert <commit-hash>

# 複数コミットをまとめて戻す
git revert <oldest-commit>..<newest-commit>
```

### pip パッケージのロールバック

```bash
# 特定バージョンの依存をインストール
pip install mediapipe==0.10.21
pip install opencv-python==4.7.0

# requirements.txt から再インストール
pip install -r requirements.txt
```

### venv の再構築

問題が解決しない場合、仮想環境を再構築:

```bash
# 既存の venv を削除
rm -rf venv  # Linux/macOS
rmdir /s /q venv  # Windows

# 再構築
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev,app]"
```

---

## セキュリティ考慮事項

### Streamlit アプリ

- **SSRF 対策**: URL 入力はプライベート IP をブロック（`resolve_safe_url`）
- **ファイルサイズ制限**: アップロード/URL 画像は 10MB まで
- **DNS リバインディング防止**: ホスト名解決後に IP をピン留め
- 本番環境では `--server.address 0.0.0.0` の使用に注意

### ライブラリ

- ファイルパス入力は `open_path()` で検証
- 画像形式は OpenCV の読み込みで検証
- NumPy 配列入力は dtype と shape を検証
