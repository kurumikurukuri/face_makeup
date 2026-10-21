# Changelog

本プロジェクトの変更履歴。フォーマットは [Keep a Changelog](https://keepachangelog.com/) に準拠し、バージョニングは [Semantic Versioning](https://semver.org/) に従う。

## [Unreleased]

### Changed

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