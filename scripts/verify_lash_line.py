"""
Check the lash-line correction by comparing main and this branch with matching correction modes.

Renders main with off / rule (2 runs) and this branch with off / rule / ml (3 runs), then
diffs the 2 x 3 = 6 pairs (pixels and correction points). Every run folder uses the same
file names, so the folders can also be compared side by side.

Output (default outputs/lash_line_check/; outputs/ is git-ignored):

    main_off/   main without correction (main has no setting; this script disables it)
    main_rule/  main as is (rule-based correction, eyeliner only)
    off/        this branch without correction (no public API; this script removes the corrector)
    rule/       this branch, landmark_correction='rule'
    ml/         this branch, landmark_correction='ml'

    Each run folder:
        <case>_full.png     whole image
        <case>_eyes.png     zoomed eye region (same crop in every folder)
        points_eyes.png     lash-line points: blue = before correction (P),
                            red = coordinates the eyeliner uses
        points.npz          the same coordinates, for numeric comparison

    compare/
        summary.txt             results for the 6 pairs (judge them with the table below)
        <main run>__<branch run>/
                                diff images for images that differ
                                (red = differing pixels, green circle = outer-corner zone)

main is not read from a working tree: --main-ref (default origin/main) is exported with
git archive into a temporary directory, so uncommitted changes in a main checkout never leak in.

Expected results (outer corner = the outer-corner zone; removing 1408/1608 only changes it):

    | Pair                  | Expected                                                      |
    |-----------------------|---------------------------------------------------------------|
    | main_off  vs off      | no differences outside the outer corner in any case           |
    |                       | (changes other than the correction affect the corner only)    |
    | main_rule vs rule     | eyeliner_*: no differences outside the outer corner; points   |
    |                       | identical except 1408/1608 (Issue #1 acceptance criterion).   |
    |                       | Other cases change because the corrected line is now shared   |
    | main_rule vs ml       | rule-based vs ML correction (reference)                       |
    | main_off  vs rule/ml  | effect of the correction (reference)                          |
    | main_rule vs off      | effect of main's correction (reference)                       |

    Also inspect ml/eyeliner_eyeshadow_eyes.png by eye: the eyeliner and eyeshadow
    boundaries should line up.

    Note: eyeliner_* may also differ outside the outer corner. soften_mask caps the feather
    width at half the thickest part of the whole mask (the distanceTransform maximum); when
    removing 1408/1608 thins the overlapping lines at the outer corner, the feather of that
    eye's entire eyeliner changes. Faint differences that only follow the line are this effect.

Usage (from the repository root):
    .venv\\Scripts\\python.exe scripts\\verify_lash_line.py --image path\\to\\face.jpg

    --main-ref selects the main-side commit. --runs limits the rendered folders
    (only pairs with both folders are compared).
"""
from __future__ import annotations

import argparse
import io
import itertools
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "lash_line_check"
BRANCH_SRC = _REPO_ROOT / "src"  # this repository (the branch being checked)

# output folder -> (source, correction)
RUNS: dict[str, tuple[str, str]] = {
    "main_off": ("main", "off"),
    "main_rule": ("main", "rule"),
    "off": ("branch", "off"),
    "rule": ("branch", "rule"),
    "ml": ("branch", "ml"),
}
MAIN_RUNS = [label for label, (src, _) in RUNS.items() if src == "main"]
BRANCH_RUNS = [label for label, (src, _) in RUNS.items() if src == "branch"]

EYE_ZOOM = 4  # zoom factor of the eye crops (nearest neighbour)
# Outer-corner zone: a circle around each outer corner (33/263) whose radius is
# this factor times the eye width (33-133 / 263-362).
DEFAULT_CORNER_RADIUS = 0.6
_OUTER_CORNERS = ((33, 133), (263, 362))  # (outer corner, inner corner)

# Strong colours and opacities so the differences are easy to see.
EYELINER = dict(color=(20, 20, 20), opacity=1.0, thickness=2, softness=0.5)
EYESHADOW = dict(color=(170, 60, 150), opacity=1, softness=0.5, size=0.6)
TEAR_BAG_HIGHLIGHT = dict(color=(255, 255, 255), opacity=0.6, softness=0.3)
TEAR_BAG_SHADOW = dict(color=(120, 70, 50), opacity=0.7, thickness=2)
FOUNDATION = dict(color=(230, 190, 170), opacity=0.6, softness=0.5)

CASES = ["eyeliner", "eyeshadow", "eyeliner_eyeshadow", "tearbag", "foundation", "all"]


# ── image I/O ─────────────────────────────────────────────────────────────────


def _save_rgb(path: Path, rgb) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"PNG encode failed: {path}")
    buf.tofile(str(path))


def _load_rgb(path: Path):
    import cv2
    import numpy as np

    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── rendering (one subprocess per run folder) ─────────────────────────────────


def _apply_case(face, case: str) -> None:
    if case in ("eyeliner", "eyeliner_eyeshadow", "all"):
        face.eyeliner(**EYELINER)
    if case in ("eyeshadow", "eyeliner_eyeshadow", "all"):
        face.eyelid(**EYESHADOW)
    if case in ("tearbag", "all"):
        face.tear_bag_highlight(**TEAR_BAG_HIGHLIGHT)
        face.tear_bag_shadow(**TEAR_BAG_SHADOW)
    if case in ("foundation", "all"):
        face.foundation(**FOUNDATION)


def _eye_bounds(raw, shape) -> tuple[int, int, int, int]:
    """Both eyes and eyelids, from MediaPipe points only, so every folder uses the same crop."""
    from face_makeup.constants import EYELID_CREASE_LEFT_INDICES, EYELID_CREASE_RIGHT_INDICES

    h, w = shape[:2]
    eye_points = [33, 133, 145, 159, 263, 362, 374, 386, 226, 446, 25, 255, 23, 253]
    idx = list(EYELID_CREASE_LEFT_INDICES) + list(EYELID_CREASE_RIGHT_INDICES) + eye_points
    pts = raw[idx]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    pad_x, pad_y = (x1 - x0) * 0.08, (y1 - y0) * 0.4
    return (
        max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y)),
        min(w, int(x1 + pad_x) + 1), min(h, int(y1 + pad_y) + 1),
    )


def _zoom(rgb, bounds):
    import cv2

    x0, y0, x1, y1 = bounds
    crop = rgb[y0:y1, x0:x1]
    return cv2.resize(
        crop, (crop.shape[1] * EYE_ZOOM, crop.shape[0] * EYE_ZOOM),
        interpolation=cv2.INTER_NEAREST,
    )


def _points_overlay(image, raw, edge_ids, used, directions, bounds):
    """Draw the lash-line points before correction (blue) and as used by the eyeliner (red)."""
    import cv2

    x0, y0, _, _ = bounds
    canvas = _zoom(image, bounds).copy()
    shift = 4
    fixed = 1 << shift

    def to_canvas(pt):
        return (
            int(round((pt[0] - x0 + 0.5) * EYE_ZOOM * fixed)),
            int(round((pt[1] - y0 + 0.5) * EYE_ZOOM * fixed)),
        )

    for edge_id, r_pt in zip(edge_ids, used):
        p = to_canvas(raw[directions[edge_id][0]])
        r = to_canvas(r_pt)
        cv2.line(canvas, p, r, (255, 255, 0), 1, cv2.LINE_AA, shift)
        cv2.circle(canvas, p, 3 * fixed, (0, 0, 255), -1, cv2.LINE_AA, shift)
        cv2.circle(canvas, r, 3 * fixed, (255, 0, 0), -1, cv2.LINE_AA, shift)
    return canvas


def _main_rule_points(image, raw):
    """Correction points main's EyelinerCorrector gives the eyeliner (edge id -> xy)."""
    import numpy as np

    from face_makeup.constants import EDGE_LANDMARK_DIRECTIONS, EYELINER_CORRECTION_MAP
    from face_makeup.core.eyeliner_corrector import EyelinerCorrector

    corrector = EyelinerCorrector()
    points: dict[int, np.ndarray] = {}
    for part_name, correction_map in EYELINER_CORRECTION_MAP.items():
        corrected = corrector.correct_points(raw, part_name, eyeliner=image)
        slots = [idx for polyline in correction_map.values() for idx in polyline]
        for slot, idx in enumerate(slots):
            if idx in EDGE_LANDMARK_DIRECTIONS:
                points[idx] = np.asarray(corrected[slot], dtype=np.float64)
    return points


def _disable_main_correction() -> None:
    """Make main's EyelinerMaskStrategy take its uncorrected path (_generate_simple)."""
    import face_makeup.core.passes.eyeliner as eyeliner_pass

    if not hasattr(eyeliner_pass, "EYELINER_CORRECTION_MAP"):
        raise RuntimeError("main's eyeliner pass has no EYELINER_CORRECTION_MAP; cannot disable correction")
    eyeliner_pass.EYELINER_CORRECTION_MAP = {}


def _disable_branch_correction(processor) -> None:
    pipeline = processor._rendering_pipeline
    if not hasattr(pipeline, "_lash_corrector"):
        raise RuntimeError("RenderingPipeline has no _lash_corrector; cannot disable correction")
    pipeline._lash_corrector = None


def _make_processor(source: str, correction: str):
    from face_makeup import MakeupProcessor, ProcessorConfig

    if source == "main":
        if correction == "off":
            _disable_main_correction()
        return MakeupProcessor(ProcessorConfig())
    # 'off' is built as 'rule' (never loads the model) and the corrector is removed before rendering.
    config = ProcessorConfig(landmark_correction="rule" if correction == "off" else correction)
    processor = MakeupProcessor(config)
    if correction == "off":
        _disable_branch_correction(processor)
    return processor


def _used_points(source: str, correction: str, processor, image, face, raw):
    """Lash-line points the eyeliner uses (edge id -> xy)."""
    import numpy as np

    from face_makeup.constants import EDGE_LANDMARK_DIRECTIONS

    if correction == "off":
        return {idx: raw[idx].astype(np.float64) for idx in EDGE_LANDMARK_DIRECTIONS}
    if source == "main":
        return _main_rule_points(image, raw)
    coords = processor._rendering_pipeline._lash_corrector.apply(image, face.landmarks, raw).coords
    return {idx: coords[idx].astype(np.float64) for idx in EDGE_LANDMARK_DIRECTIONS}


def worker(label: str, src: str, image_path: str, output_root: Path) -> None:
    sys.path.insert(0, src)
    import face_makeup

    if not Path(face_makeup.__file__).resolve().is_relative_to(Path(src).resolve()):
        raise RuntimeError(f"{label}: face_makeup was imported from {face_makeup.__file__}")

    import numpy as np

    from face_makeup.constants import EDGE_LANDMARK_DIRECTIONS

    source, correction = RUNS[label]
    out_dir = output_root / label
    with _make_processor(source, correction) as processor:
        ctx = processor.open_path(image_path)
        image = ctx.original_image
        h, w = image.shape[:2]
        face0 = ctx.faces[0]
        raw = face0.landmarks.to_image_coords(w, h)
        bounds = _eye_bounds(raw, image.shape)

        for case in CASES:
            for face in ctx.faces:
                face.clear()
                _apply_case(face, case)
            result = ctx.render()
            _save_rgb(out_dir / f"{case}_full.png", result)
            _save_rgb(out_dir / f"{case}_eyes.png", _zoom(result, bounds))

        used = _used_points(source, correction, processor, image, face0, raw)
    edge_ids = sorted(used)
    used_arr = np.array([used[i] for i in edge_ids])
    _save_rgb(
        out_dir / "points_eyes.png",
        _points_overlay(image, raw, edge_ids, used_arr, EDGE_LANDMARK_DIRECTIONS, bounds),
    )
    np.savez(out_dir / "points.npz", edge_ids=np.array(edge_ids), used=used_arr, raw=raw)
    print(f"[{label}] saved: {out_dir}")


# ── comparison (parent process; never imports face_makeup) ────────────────────


def _corner_zone(raw, shape, radius_factor: float):
    """Boolean mask of the outer-corner zone (a circle around each outer corner)."""
    import cv2
    import numpy as np

    zone = np.zeros(shape[:2], dtype=np.uint8)
    for corner, inner in _OUTER_CORNERS:
        width = float(np.linalg.norm(raw[corner] - raw[inner]))
        center = tuple(int(v) for v in raw[corner])
        cv2.circle(zone, center, max(1, int(round(width * radius_factor))), 1, -1)
    return zone.astype(bool)


def _diff_image(a, diff, zone):
    """Dimmed grayscale of *a* with differing pixels in red and the corner zone outlined in green."""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB) // 2
    gray[diff] = (255, 0, 0)
    contours, _ = cv2.findContours(zone.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(gray, contours, -1, (0, 255, 0), 1)
    return gray


def _compare_points(a_dir: Path, b_dir: Path) -> str:
    import numpy as np

    a, b = np.load(a_dir / "points.npz"), np.load(b_dir / "points.npz")
    pa = dict(zip(a["edge_ids"].tolist(), a["used"]))
    pb = dict(zip(b["edge_ids"].tolist(), b["used"]))
    common = sorted(set(pa) & set(pb))
    differing = [i for i in common if not np.array_equal(pa[i], pb[i])]
    max_diff = max((float(np.abs(pa[i] - pb[i]).max()) for i in differing), default=0.0)
    only = sorted(set(pa) ^ set(pb))
    return (
        f"  points: {len(differing)} of {len(common)} common points differ"
        f" (max {max_diff:.2f} px){' ' + str(differing) if differing else ''}"
        f" / only in one run: {only if only else 'none'}"
    )


def compare(output_root: Path, labels: list[str], radius_factor: float) -> None:
    import numpy as np

    pairs = [
        (m, b) for m, b in itertools.product(MAIN_RUNS, BRANCH_RUNS)
        if m in labels and b in labels
    ]
    if not pairs:
        return
    raw = np.load(output_root / pairs[0][0] / "points.npz")["raw"]
    compare_root = output_root / "compare"
    lines = [
        f"outer-corner zone: circle around 33/263 with radius = eye width x {radius_factor}",
        "differing pixels = total / outside the outer-corner zone",
        "",
    ]
    for a_label, b_label in pairs:
        a_dir, b_dir = output_root / a_label, output_root / b_label
        lines.append(f"=== {a_label} vs {b_label} ===")
        for case in CASES:
            name = f"{case}_full.png"
            a_img, b_img = _load_rgb(a_dir / name), _load_rgb(b_dir / name)
            diff = (a_img != b_img).any(axis=2)
            zone = _corner_zone(raw, a_img.shape, radius_factor)
            n_all, n_out = int(diff.sum()), int((diff & ~zone).sum())
            mark = "  " if n_out == 0 else "!!"
            lines.append(f"  {mark} {name:30s} {n_all:7d} / {n_out:7d}")
            if n_all:
                _save_rgb(
                    compare_root / f"{a_label}__{b_label}" / name,
                    _diff_image(a_img, diff, zone),
                )
        lines.append(_compare_points(a_dir, b_dir))
        lines.append("")
    summary = "\n".join(lines)
    compare_root.mkdir(parents=True, exist_ok=True)
    (compare_root / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


# ── exporting main ────────────────────────────────────────────────────────────


def _export_ref(ref: str, dest: Path) -> tuple[Path, str]:
    """Export src/ of *ref* into *dest* with git archive. Returns (src path, commit)."""
    commit = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    archive = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "archive", "--format=zip", commit, "src"],
        capture_output=True, check=True,
    ).stdout
    zipfile.ZipFile(io.BytesIO(archive)).extractall(dest)
    return dest / "src", commit


# ── arguments and overall flow ────────────────────────────────────────────────


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render main (off / rule) and this branch (off / rule / ml) and compare them.",
    )
    parser.add_argument("--image", required=True, type=Path, help="face image to check")
    parser.add_argument(
        "--main-ref", default="origin/main",
        help="commit exported as main (default: origin/main)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help=f"output directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--runs", nargs="+", choices=list(RUNS), default=list(RUNS),
        help="run folders to render (default: all)",
    )
    parser.add_argument(
        "--corner-radius", type=float, default=DEFAULT_CORNER_RADIUS,
        help=f"outer-corner zone radius as a factor of the eye width (default: {DEFAULT_CORNER_RADIUS})",
    )
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"--image not found: {args.image}")
    return args


def _run_worker(label: str, src: Path, image: Path, output_root: Path) -> None:
    cmd = [sys.executable, __file__, "--worker", label, str(src), str(image), str(output_root)]
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"[{label}] failed")


def main(argv: list[str]) -> None:
    args = _parse_args(argv)
    output_root = args.output_dir.resolve()
    image = args.image.resolve()
    with tempfile.TemporaryDirectory(prefix="lash_line_main_") as tmp:
        main_src = None
        if any(RUNS[label][0] == "main" for label in args.runs):
            main_src, commit = _export_ref(args.main_ref, Path(tmp))
            print(f"main: {args.main_ref} = {commit}")
        # Each source is a different face_makeup module, so every run gets its own process.
        for label in args.runs:
            src = main_src if RUNS[label][0] == "main" else BRANCH_SRC
            _run_worker(label, src, image, output_root)
    compare(output_root, args.runs, args.corner_radius)
    print(f"done: {output_root}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _, _, label, src, image_path, output_root = sys.argv
        worker(label, src, image_path, Path(output_root))
    else:
        main(sys.argv[1:])
