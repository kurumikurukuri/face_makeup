# Changelog

本プロジェクトの変更履歴。フォーマットは [Keep a Changelog](https://keepachangelog.com/) に準拠し、バージョニングは [Semantic Versioning](https://semver.org/) に従う。

## [Unreleased]

### Added

- まつ毛ラインのランドマーク補正。`EyelinerMLCorrector`（`core/eyeliner_ml_corrector.py`）が [EyeContourRefiner](https://github.com/kurumikurukuri/EyeContourRefiner) のモデル（dual-branch ONNX、CPU、onnxruntime）を使い、まつ毛ラインのランドマーク（1401〜1407・1501〜1507・1601〜1607・1701〜1707）を実際のまつ毛ライン上へ補正する。
  - 補正は顔ごとに1回だけ renderer で行い（`core/landmark_correction.py` の `LashLineCorrector`）、補正後の座標をすべてのパスで共有する。アイライナー、アイシャドウ（全ゾーン）、涙袋、およびファンデーションの目元の除外範囲が同じまつ毛ラインを使うので、パーツ同士の境界がずれない。補正点は小数精度のまま渡す。
  - これらのパスを含まない顔（唇だけなど）では補正を行わず、モデルも読み込まない。
  - `ProcessorConfig.landmark_correction` を追加（`'ml'` / `'rule'`、デフォルト `'ml'`）。
    - `'ml'`（デフォルト）は機械学習補正を使う。モデルがインストールされていない、または読み込めない場合や、ONNX の入出力契約に合わない場合は、最初の補正時に例外（`RuntimeError` / `ModelContractError`）を送出する。ルールベース補正へのフォールバックはしない。
    - `'rule'` は従来のルールベース補正（2階微分ゼロ交差検出）。
  - モデルは本リポジトリに同梱せず、EyeContourRefiner の `main` の最新版を次のタイミングで取得する（`core/eyeliner_model.py`）。描画時にはダウンロードしない。SHA256 は固定せず、取得時と読み込み時に ONNX の入出力契約を検証する。`pip install` 時は onnxruntime をビルド依存（`[build-system] requires`）に含めているため、インストール前に必ず検証される。onnxruntime が入っていない Python で git hook を実行した場合は、読み込み時にだけ検証される。
    - `pip install`（`git+...`・`pip install -e .`・Community Cloud のデプロイ）: `setup.py` の `build_py` フックで取得し、パッケージデータ（`face_makeup/models/`）としてインストールする。アンインストールや更新では pip が古いモデルも削除するので、利用者がモデルファイルを管理する必要はない。
      - `requirements.txt`（Community Cloud・`run_streamlit_app.bat`）を、face_makeup 自体（`.`）をインストールする内容に変更し、コア依存は `pyproject.toml` に一本化した（OpenCV は `requirements.txt` で使っていた `opencv-python-headless` に揃え、`pyproject.toml` の `opencv-python` から置き換えた）。Streamlit アプリは、face_makeup がインストールされていないときだけ `src/` から import する。
    - `git pull` 後: `.githooks/` の `post-merge` / `post-rewrite` フックで取得する（clone ごとに `git config core.hooksPath .githooks` を一度実行）。取得に失敗しても pull は中断せず、以前のモデルをそのまま使う。
    - 設計判断: 上流のモデル更新に自動で追従するため、commit SHA や SHA256 で固定しない。固定すると、モデルを更新するたびに固定値を書き換える PR が必要になる。固定しないことで生じる次の点は許容する。
      - すり替えのリスク: 上流リポジトリも同じ所有者が管理しており、onnxruntime では ONNX ファイル経由で任意のコードが実行されることはないため、影響は描画結果に限られる。
      - 環境ごとにモデルの版が異なりうる: 同じ face_makeup でも、インストール（または pull）した時点の `main` のモデルが入る。Community Cloud はデプロイごとに環境を作り直すので、最新モデルが入る。
      - 上流の変更で無関係な PR の CI が落ちうる: 入出力契約の破壊をすぐに検知するための意図した挙動。
  - `debug=True` の描画では、機械学習補正の結果を目ごとに1枚の画像（`outputs/debug/<session>/eyeliner_ml_debug_{left,right}.png`）に保存する。拡大した目の周辺に P（青）・Q（緑）・補正点 R（赤）と、ランドマーク ID・t を重ねて描く。描画時の推論結果をキャッシュして使うので、デバッグのために推論をやり直すことはない（`core/eyeliner_ml_debug.py`）。
  - モデル出力の `t` が NaN/inf の場合は `t = 1`（R = P、補正なしの位置。ルールベース補正の失敗時と同じ）に置き換えて警告を出し、さらに学習ラベルの範囲 [-1, 2] に clip する。上流モデルが変わっても、目から大きく外れた点や NaN 座標の線は描かれない。
  - モデルの入力（パッチ切り出し・P/Q・`eye_scale`）には、学習データと同じく丸めていない float の画像座標（新設の `FaceLandmarks.to_image_coords_float`）を使う。補正点 R は `R = t*P + (1-t)*Q` で再構成する。このときの P/Q には、renderer が作る整数の画像座標（`to_image_coords`）を使う。
  - パッチ切り出し処理を `core/eyeliner_patch.py` に移植したため、EyeContourRefiner パッケージには依存しない。
  - モデル入力の `eye_scale` は、モデルの学習データと同じく学習対象の14点（上下7点ずつ）から計算する。
  - 上下のラインで重複する1507/1707は、機械学習補正・ルールベース補正のどちらでも下まぶたの点として1回だけ補正し、両方のラインで同じ点を使う（`core/lash_line_edges.py`）。
  - CI は `pip install` のたびに `main` の最新モデルを取得してテストする（上流のモデル更新に気づけるよう、日次スケジュールでも起動。上流からの `repository_dispatch` は、検知が最大1日早まるだけで PAT の管理が増えるため使わない）。モデルがない場合、実モデルを使うテストはスキップではなく失敗とする。
  - 依存に `onnxruntime` を追加。OpenCV を `<5` に固定（5.x では `warpAffine` の丸め方が変わり、モデル入力のパッチがずれるため）。`cv2` は複数の配布物が同じモジュールとして提供し、後からインストールされたものが使われるので、face_makeup が依存する `opencv-python-headless` と、mediapipe がバージョン指定なしで持ち込む `opencv-contrib-python` の両方を `<5` に固定する。インストールされた `cv2` の配布物がすべて 5 未満であることと、face_makeup が両方の上限を宣言していることをテストで確認する（`tests/test_dependencies.py`）。
- `core/model_files.py` を追加。髪セグメンテーションモデルの配置ディレクトリの解決（`resolve_model_dir`。`processor.py` から移動）と、まつ毛ライン補正モデルのダウンロード処理（`download_file`）を提供する。`ProcessorConfig.model_dir` はこれまでどおり髪セグメンテーションモデルだけに適用される（まつ毛ライン補正モデルはパッケージと一緒にインストールされる）。
- `MaskStrategy.uses_lash_line` を追加。まつ毛ラインのランドマークを使う `MaskStrategy`（アイライナー・アイシャドウ・涙袋）はこれを `True` にする。renderer は、`uses_lash_line` が `True` の `MaskStrategy` か、clip / exclusion にまつ毛ラインのランドマークを含むパス（ファンデーション）があるときだけ補正を行う。

### Changed

- **破壊的変更:** アイライナーの補正対象から目尻のまつ毛ラインのランドマーク1408/1608を外し、上ラインを1407/1607から130/359へ直接つなぐ（ランドマーク定義からも削除）。これらは EyeContourRefiner が意図的に学習対象から外している点であり、機械学習補正・ルールベース補正の両方に適用する。このためルールベース補正の描画も目尻付近がわずかに変わる。
  - `constants.LM_1408` / `LM_1608` を削除したため、これらを import しているコードは `ImportError` になる。
  - `EDGE_LANDMARK_DIRECTIONS[1408]` / `[1608]` は `KeyError` になる。
  - 拡張ランドマーク配列の大きさは変わらないため、インデックス1408/1608は参照してもエラーにはならないが、値は計算されず `(0, 0)` のままとなる。
- **破壊的変更:** まつ毛ラインの補正は常に（`'ml'` / `'rule'` のいずれかで）行われるため、アイライナーに加えてアイシャドウ・涙袋・ファンデーションの描画も、まつ毛ライン付近が 0.1.0 から変わる。補正を無効にする公開 API はない。
  - アイシャドウの `EYELID_*_INDICES` は、まつ毛ライン部分の点として、MediaPipe の点（173〜246/398〜466）ではなくまつ毛ラインのランドマーク（`LM_1401`〜`LM_1407` / `LM_1601`〜`LM_1607`）を使う。補正前のまつ毛ラインのランドマークは元の MediaPipe の点と同じ座標なので、補正しなければアイシャドウの描画は 0.1.0 と変わらない。
  - ファンデーションの目元の除外範囲のうち、アイライナーのラインから作る範囲からは目尻の点（旧1408/1608。MediaPipe の33/263を補正していた点）が外れる。アイシャドウのラインから作る範囲には33/263が残る。
- **破壊的変更:** モデルを取得・読み込みできない環境（オフラインなど）では、デフォルトの `'ml'` は目のメイク・ファンデーションを含む描画で例外を送出する。そうした環境では `landmark_correction='rule'` を指定する。
- ruff のルールセットを `pyproject.toml` で明示的に固定（新しい ruff でデフォルト選択が広がり CI が落ちるのを防ぐ）。

### Fixed

- Streamlit アプリで画像が表示されない問題。starlette 1.8 で Streamlit のメディア配信が失敗するため、`streamlit==1.58.0` と `starlette<1.8` に固定。

## [0.1.0] - 2026-07-18

初回リリース。

### Added

- MediaPipe による顔検出・ランドマーク抽出と fluent API (`MakeupProcessor` / `ImageContext` / `Face`)
- 7つのメイクパーツ:
  - リップ
  - チーク
  - アイシャドウ(水平ゾーン指定・多段重ね・ラメ)
  - アイライナー
  - カラーコンタクト
  - 涙袋ハイライト / シャドウ
  - ファンデーション
- 髪セグメンテーションによる全メイクパスからの髪除外
- Streamlit デモアプリ(`apps/streamlit_app.py`)