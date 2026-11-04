# Models

## Hair Segmentation

Download the MediaPipe hair segmentation model:

```bash
curl -L -o models/hair_segmenter.tflite \
  https://storage.googleapis.com/mediapipe-assets/hair_segmentation.tflite
```

- **License**: Apache 2.0 (same as MediaPipe)
- **Size**: ~782 KB
- **Source**: [MediaPipe official model assets](https://storage.googleapis.com/mediapipe-assets/hair_segmentation.tflite)

## Lash-Line Correction

The lash-line landmark-correction model (`eye_contour_refiner.onnx`) is published
on the `main` branch of [EyeContourRefiner](https://github.com/kurumikurukuri/EyeContourRefiner)
and is **not** committed to this repository.

- **When it is fetched**: never while rendering. `pip install` (`git+...`,
  `pip install -e .`, a Streamlit Community Cloud deploy) downloads the latest
  `main` model through the `build_py` hook in `setup.py` and installs it as
  package data (`face_makeup/models/`; `src/face_makeup/models/` in a source
  checkout, git-ignored). After `git pull`, the `post-merge` / `post-rewrite`
  hooks in `.githooks/` refresh it; enable them once per clone with
  `git config core.hooksPath .githooks`. A failed refresh keeps the previous model.
- **Missing or unusable model**: `'ml'` raises (there is no fallback) and asks
  to reinstall face_makeup (`pip install -e .` in a source checkout); use
  `landmark_correction='rule'` where the model cannot be installed.
- **Validation**: no checksum is pinned (the model tracks `main`); the model is
  checked against the expected ONNX input/output contract when it is fetched
  and whenever it is loaded. `pip install` always checks it before installing
  (onnxruntime is a build requirement); a git hook running under a Python
  without onnxruntime leaves the check to load time.
- **CI**: `pip install` fetches the latest model from `main` on every run
  and also daily, so an upstream model change is tested even when this
  repository is idle.

- **License**: MIT (training-data attribution: `models/DATA_ATTRIBUTION.csv` in EyeContourRefiner)
- **Size**: ~15 MB
