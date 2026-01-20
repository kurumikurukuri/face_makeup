"""
メイクブレンド処理
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from .types import ColorParams, LipsParams, CheekParams, EyeShadowParams,EyeLinerParams,FaceLandmarks


class MakeupBlender:
    """メイクのブレンド処理"""

    def blend(self,
              image: np.ndarray,
              mask: np.ndarray,
              color_params: ColorParams,
              opacity: float) -> np.ndarray:
        """
        メイク色を画像に適用する
        """
        # 入力チェック
        if image.dtype != np.uint8:
            raise ValueError(f"image must be uint8, got {image.dtype}")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3), got {image.shape}")
        if mask.dtype != np.float32:
            raise ValueError(f"mask must be float32, got {mask.dtype}")
        if len(mask.shape) != 2:
            raise ValueError(f"mask must be (H, W), got {mask.shape}")
        if image.shape[:2] != mask.shape:
            raise ValueError(f"image shape {image.shape[:2]} != mask shape {mask.shape}")

        # RGB → HSV変換（float32 0..1）
        image_float = image.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(image_float, cv2.COLOR_RGB2HSV)

        # 色適用
        result_hsv = self._apply_target_color(
            hsv, mask, color_params.target_rgb, opacity
        )

        # HSV → RGB変換
        rgb_modified = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)

        # uint8クリップ & 返却
        result = np.clip(rgb_modified * 255, 0, 255).astype(np.uint8)
        return result

    def _apply_target_color(self,
                            hsv: np.ndarray,
                            mask: np.ndarray,
                            target_rgb: tuple,
                            opacity: float) -> np.ndarray:
        """
        target_rgbで指定された色を適用する
        黒・無彩色はマスクと同化させないよう「明度操作」として扱う
        """
        target_color = np.array(target_rgb, dtype=np.float32) / 255.0
        target_hsv = cv2.cvtColor(
            target_color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV
        )[0, 0]

        # ブレンド重み
        blend_weight = mask[:, :, np.newaxis] * opacity

        # 元画像 RGB
        rgb_original = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # === 黒・無彩色判定 ===
        is_neutral = target_hsv[1] < 0.05  # Saturation がほぼ 0

        if is_neutral:
            # --- 明度を下げるだけ（影・ライン用途） ---
            hsv_dark = hsv.copy()
            hsv_dark[:, :, 2] = np.clip(
                hsv[:, :, 2] * (1.0 - blend_weight[:, :, 0]),
                0.0, 1.0
            )

            return hsv_dark

        # === 通常の色付きメイク ===
        result_hsv = hsv.copy()
        result_hsv[:, :, 0] = target_hsv[0]
        result_hsv[:, :, 1] = target_hsv[1]
        result_hsv[:, :, 2] = hsv[:, :, 2] * 0.8 + target_hsv[2] * 0.2

        rgb_target = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)

        rgb_blended = (
            rgb_original * (1 - blend_weight) +
            rgb_target * blend_weight
        )

        return cv2.cvtColor(rgb_blended, cv2.COLOR_RGB2HSV)


    def blend_lips(self,
                   image: np.ndarray,
                   mask: np.ndarray,
                   params: LipsParams) -> np.ndarray:
        """
        リップにメイクを適用する
        """
        return self.blend(image, mask, params.color_params, params.opacity)

    def blend_cheek(self,
                    image: np.ndarray,
                    mask: np.ndarray,
                    params: CheekParams,
                    landmarks: Optional[FaceLandmarks] = None,
                    cheek_side: str = 'left') -> np.ndarray:
        """
        チークにメイクを適用する（楕円マスク使用）

        Args:
            image: 入力画像
            mask: チークマスク
            params: チークパラメータ
            landmarks: 顔ランドマーク（3D座標を含む）
            cheek_side: 'left' or 'right' - どちらの頬か
        """
        ellipse_mask = self._create_ellipse_mask(
            image.shape[:2], mask, params, landmarks, cheek_side
        )
        combined_mask = mask * ellipse_mask
        return self.blend(image, combined_mask, params.color_params, params.opacity)

    def blend_eyeshadow(self,
                        image: np.ndarray,
                        mask: np.ndarray,
                        params: EyeShadowParams) -> np.ndarray:
        """
        アイシャドウにメイクを適用する

        将来的にグラデーション機能を追加予定
        現状は通常のブレンド処理のみ
        """
        return self.blend(image, mask, params.color_params, params.opacity)

    def blend_eyeliner(self,
                        image: np.ndarray,
                        mask: np.ndarray,
                        params: EyeLinerParams) -> np.ndarray:
        """
        アイライナーにメイクを適用する

        Args:
            image: 入力画像
            mask: アイライナーマスク
            params: アイライナーパラメータ（thickness で太さを調整）
        """
        # 太さに応じてマスクを膨張
        if params.thickness > 1:
            kernel = np.ones((params.thickness, params.thickness), np.uint8)
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
            mask = mask_dilated.astype(np.float32) / 255.0

        return self.blend(image, mask, params.color_params, params.opacity)


    def _create_ellipse_mask(self,
                             image_shape: Tuple[int, int],
                             base_mask: np.ndarray,
                             params: CheekParams,
                             landmarks: Optional[FaceLandmarks] = None,
                             cheek_side: str = 'left',
                             return_debug_info: bool = False) -> np.ndarray:
        """
        楕円グラデーションマスクを生成する

        Args:
            image_shape: 画像サイズ (h, w)
            base_mask: ベースマスク
            params: チークパラメータ
            landmarks: 顔ランドマーク（3D座標を含む）
            cheek_side: 'left' or 'right'

        Returns:
            楕円マスク (h, w) float32
        """
        h, w = image_shape

        mask_binary = (base_mask > 0.1).astype(np.uint8)
        if np.sum(mask_binary) == 0:
            return np.zeros((h, w), dtype=np.float32)

        # 3D多様体ベースの処理（landmarksが提供されている場合）
        if landmarks is not None and landmarks.points_3d is not None:
            return self._create_ellipse_mask_3d(
                image_shape, base_mask, params, landmarks, cheek_side, return_debug_info
            )

        # 従来の2D処理（後方互換性）
        coords = np.column_stack(np.where(mask_binary > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        mask_h = y_max - y_min
        mask_w = x_max - x_min

        cx = x_min + params.center[0] * mask_w
        cy = y_min + params.center[1] * mask_h
        rx = params.radius[0] * mask_w
        ry = params.radius[1] * mask_h

        y_grid, x_grid = np.ogrid[:h, :w]
        distance = (
            ((x_grid - cx) / (rx + 1e-6)) ** 2 +
            ((y_grid - cy) / (ry + 1e-6)) ** 2
        )

        ellipse_mask = np.exp(-distance / (params.falloff + 1e-6))
        result_mask = np.clip(ellipse_mask, 0, 1).astype(np.float32)

        if return_debug_info:
            debug_info = {
                'mode': '2d',
                'cx': cx,
                'cy': cy,
                'rx': rx,
                'ry': ry,
            }
            return result_mask, debug_info

        return result_mask

    def _create_ellipse_mask_3d(self,
                                    image_shape: Tuple[int, int],
                                    base_mask: np.ndarray,
                                    params: CheekParams,
                                    landmarks: FaceLandmarks,
                                    cheek_side: str,
                                    return_debug_info: bool = False) -> np.ndarray:
        """
        頬曲面上に楕円を描画する（局所座標系ベース）

        仕様: docs/cheek_idea.md 参照

        Args:
            image_shape: (height, width)
            base_mask: ベースマスク
            params: チークパラメータ
            landmarks: 顔ランドマーク（3D座標必須）
            cheek_side: 'left' or 'right'
            return_debug_info: デバッグ情報を返すか

        Returns:
            楕円マスク (h, w) float32 (0..1)
        """
        from src.face_makeup.constants import CHEEK_LEFT_INDICES, CHEEK_RIGHT_INDICES

        h, w = image_shape

        # 入力検証
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image_shape: {image_shape}. Both dimensions must be positive.")

        points_3d = landmarks.points_3d

        if points_3d is None:
            # 3D情報がない場合はゼロマスク
            return np.zeros((h, w), dtype=np.float32)

        # 1. 頬ランドマーク取得
        cheek_indices = CHEEK_LEFT_INDICES if cheek_side == 'left' else CHEEK_RIGHT_INDICES
        cheek_pts_3d = points_3d[cheek_indices]

        # 2. 局所座標系を定義するランドマーク取得
        if cheek_side == 'left':
            # 左頬
            p_u1_idx, p_u2_idx = 50, 36    # u方向定義点
            p_v1_idx, p_v2_idx = 118, 205  # v方向定義点
        else:
            # 右頬
            p_u1_idx, p_u2_idx = 280, 266  # u方向定義点
            p_v1_idx, p_v2_idx = 347, 425  # v方向定義点

        p_u1 = points_3d[p_u1_idx]
        p_u2 = points_3d[p_u2_idx]
        p_v1 = points_3d[p_v1_idx]
        p_v2 = points_3d[p_v2_idx]

        # 3. 局所座標系の構築
        # 原点: 頬ランドマークの平均
        p0 = cheek_pts_3d.mean(axis=0)

        # u方向ベクトル
        u_raw = p_u2 - p_u1
        e_u = u_raw / (np.linalg.norm(u_raw) + 1e-6)

        # v方向ベクトル（グラム・シュミット直交化）
        v_raw = p_v2 - p_v1
        v_raw = v_raw - np.dot(v_raw, e_u) * e_u  # e_uに直交化
        e_v = v_raw / (np.linalg.norm(v_raw) + 1e-6)

        # w方向ベクトル（外積）
        e_w = np.cross(e_u, e_v)
        e_w = e_w / (np.linalg.norm(e_w) + 1e-6)

        # 4. 頬ランドマークを局所座標系に変換
        cheek_local = []
        for pt in cheek_pts_3d:
            vec = pt - p0
            u_coord = np.dot(vec, e_u)
            v_coord = np.dot(vec, e_v)
            w_coord = np.dot(vec, e_w)
            cheek_local.append([u_coord, v_coord, w_coord])
        cheek_local = np.array(cheek_local)

        # 5. 曲面近似（2次曲面フィット）
        # w = c0 + c1*u + c2*v + c3*u^2 + c4*v^2 + c5*u*v
        U = cheek_local[:, 0]
        V = cheek_local[:, 1]
        W = cheek_local[:, 2]

        # 計画行列
        A = np.column_stack([
            np.ones_like(U),
            U,
            V,
            U**2,
            V**2,
            U*V
        ])

        # 最小二乗法
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(A, W, rcond=None)
        except np.linalg.LinAlgError:
            # フィッティング失敗時は平面近似
            coeffs = np.zeros(6)
            coeffs[0] = W.mean()

        # 曲面評価関数
        def surface_func(u, v):
            return (coeffs[0] + coeffs[1]*u + coeffs[2]*v +
                    coeffs[3]*u**2 + coeffs[4]*v**2 + coeffs[5]*u*v)

        # 6. 局所UV範囲の計算と正規化空間の定義
        u_min = U.min() if len(U) > 0 else -0.1
        u_max = U.max() if len(U) > 0 else 0.1
        v_min = V.min() if len(V) > 0 else -0.1
        v_max = V.max() if len(V) > 0 else 0.1

        # 局所UV中心・スケール
        u_center = (u_max + u_min) / 2
        v_center = (v_max + v_min) / 2
        u_half = (u_max - u_min) / 2
        v_half = (v_max - v_min) / 2

        # params適用（正規化空間: -1〜1）
        # params.center ∈ (-1, 1), params.radius ∈ (0, 1]
        u0 = u_center + params.center[0] * u_half
        v0 = v_center + params.center[1] * v_half
        a = abs(params.radius[0]) * u_half
        b = abs(params.radius[1]) * v_half

        # パラメトリック表現で楕円点を生成
        N = 50  # サンプリング点数
        theta = np.linspace(0, 2*np.pi, N)
        u_ellipse = u0 + a * np.cos(theta)
        v_ellipse = v0 + b * np.sin(theta)

        # 7. 曲面上への投影・3D点復元
        ellipse_3d_points = []
        for u, v in zip(u_ellipse, v_ellipse):
            w_coord = surface_func(u, v)
            # 3D空間に変換
            pt_3d = p0 + u * e_u + v * e_v + w_coord * e_w
            ellipse_3d_points.append(pt_3d)
        ellipse_3d_points = np.array(ellipse_3d_points)

        # 8. 3D楕円点列から2Dマスクを生成
        # 各画素について、楕円線までの距離を計算
        y_grid, x_grid = np.ogrid[:h, :w]

        # 楕円の中心を2D投影
        center_w = surface_func(u0, v0)
        center_3d = p0 + u0 * e_u + v0 * e_v + center_w * e_w
        cx = center_3d[0] * w
        cy = center_3d[1] * h

        # 局所座標系の基底を2D投影
        P_proj = np.array([p0[0] * w, p0[1] * h])
        P_u_proj = np.array([(p0 + 0.01 * e_u)[0] * w, (p0 + 0.01 * e_u)[1] * h])
        P_v_proj = np.array([(p0 + 0.01 * e_v)[0] * w, (p0 + 0.01 * e_v)[1] * h])

        ex = P_u_proj - P_proj
        ey = P_v_proj - P_proj

        ex_norm = np.linalg.norm(ex)
        ey_norm = np.linalg.norm(ey)

        if ex_norm < 1e-6 or ey_norm < 1e-6:
            return np.zeros((h, w), dtype=np.float32)

        ex = ex / ex_norm
        ey = ey / ey_norm

        # 楕円半径を2D投影スケールに調整
        # 局所座標の単位ベクトル0.01が画像上でどれだけの長さになるかを計算
        scale_u = ex_norm / 0.01  # 0.01の局所座標が何ピクセルか
        scale_v = ey_norm / 0.01
        rx = a * scale_u
        ry = b * scale_v

        # 楕円距離計算（局所座標系ベース）
        dx = x_grid - cx
        dy = y_grid - cy

        ux = dx * ex[0] + dy * ex[1]
        uy = dx * ey[0] + dy * ey[1]

        distance = (ux / (rx + 1e-6)) ** 2 + (uy / (ry + 1e-6)) ** 2

        # 指数減衰でマスク生成
        ellipse_mask = np.exp(-distance / (params.falloff + 1e-6))
        result_mask = np.clip(ellipse_mask, 0, 1).astype(np.float32)

        if return_debug_info:
            debug_info = {
                'p0': p0,
                'e_u': e_u,
                'e_v': e_v,
                'e_w': e_w,
                'cx': cx,
                'cy': cy,
                'rx': rx,
                'ry': ry,
                'ex': ex,
                'ey': ey,
                # 正規化UV空間情報
                'u_min': u_min,
                'u_max': u_max,
                'v_min': v_min,
                'v_max': v_max,
                'u_center': u_center,
                'v_center': v_center,
                'u_half': u_half,
                'v_half': v_half,
                # 楕円パラメータ（局所UV座標）
                'u0': u0,
                'v0': v0,
                'a': a,
                'b': b,
                'coeffs': coeffs,
                'ellipse_3d_points': ellipse_3d_points,
                'cheek_local': cheek_local,
            }
            return result_mask, debug_info

        return result_mask
