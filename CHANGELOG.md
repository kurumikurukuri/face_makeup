# Changelog

本プロジェクトの変更履歴。フォーマットは [Keep a Changelog](https://keepachangelog.com/) に準拠し、バージョニングは [Semantic Versioning](https://semver.org/) に従う。

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