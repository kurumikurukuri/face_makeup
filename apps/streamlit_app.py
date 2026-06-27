"""
Makeup Simulator - Streamlit UI

Usage:
    streamlit run apps/streamlit_app.py
"""
import sys
from pathlib import Path

# Add src directory to path for imports when package is not installed
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import ipaddress
import socket
from urllib.parse import urlparse, urlunparse

import streamlit as st
import requests
from PIL import Image, ImageOps
import numpy as np
import io
from datetime import datetime

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False



from face_makeup import MakeupProcessor

# ========================================
# Helper Functions
# ========================================

def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: #{hex_color}")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _on_separate_colors_change():
    """Handle separate colors checkbox change event for eye iris settings."""
    if st.session_state.eyeiris_separate_colors:
        # Propagate unified color to both eyes
        st.session_state.eyeiris_color_left = st.session_state.eyeiris_color
        st.session_state.eyeiris_color_right = st.session_state.eyeiris_color
    else:
        # Revert unified color from left eye
        st.session_state.eyeiris_color = st.session_state.eyeiris_color_left


MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


def resolve_safe_url(url: str) -> tuple:
    """Resolve URL hostname and verify it points to a public IP.

    Returns the resolved IP and parsed URL components so the caller can
    make the HTTP request using the pinned IP, preventing DNS rebinding.

    Args:
        url: The URL to validate.

    Returns:
        Tuple of (resolved_ip, hostname, parsed_url) on success.

    Raises:
        ValueError: If the URL points to a private/internal address or
            cannot be resolved.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URLにホスト名がありません。")
    try:
        resolved_ip = socket.getaddrinfo(hostname, None)[0][4][0]
    except socket.gaierror:
        raise ValueError("ホスト名を解決できません。")
    ip = ipaddress.ip_address(resolved_ip)
    if not ip.is_global:
        raise ValueError("内部ネットワークのURLは許可されていません。")
    return resolved_ip, hostname, parsed


def fetch_image_safe(url: str) -> io.BytesIO:
    """Fetch an image from a URL with SSRF protection and size limit.

    Resolves the hostname once and pins the IP for the actual request,
    preventing DNS rebinding. Streams the response and enforces a max
    size of MAX_IMAGE_SIZE bytes.

    Args:
        url: The image URL to fetch.

    Returns:
        BytesIO containing the image data.

    Raises:
        ValueError: If the URL is unsafe or the response exceeds the size limit.
        requests.exceptions.RequestException: On network errors.
    """
    resolved_ip, hostname, parsed = resolve_safe_url(url)

    # Build a URL with the resolved IP to pin the connection
    pinned_url = urlunparse(parsed._replace(netloc=resolved_ip))

    response = requests.get(
        pinned_url,
        headers={"Host": hostname},
        timeout=10,
        stream=True,
        verify=True,
    )
    response.raise_for_status()

    # Read in chunks with size limit
    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        total += len(chunk)
        if total > MAX_IMAGE_SIZE:
            response.close()
            raise ValueError(
                f"画像サイズが上限（{MAX_IMAGE_SIZE // (1024 * 1024)}MB）を超えています。"
            )
        chunks.append(chunk)

    return io.BytesIO(b"".join(chunks))

@st.cache_resource
def get_processor():
    """Get cached MakeupProcessor instance."""
    return MakeupProcessor()


# ========================================
# Page Configuration
# ========================================

st.set_page_config(
    page_title="メイクシミュレーター",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Eyelid paint-region (zone) label -> API value
_EYELID_ZONE_OPTIONS = {
    "全体": "full",
    "目尻側": "outer",
    "中心部": "center",
    "目頭側": "inner",
}
_EYELID_ZONE_LABELS = list(_EYELID_ZONE_OPTIONS.keys())

_SESSION_DEFAULTS = {
    'foundation_enabled': False,
    'foundation_color': "#FFDCC8",
    'foundation_opacity': 0.3,
    'lips_enabled': False,
    'lips_color': "#FF4D4D",
    'lips_opacity': 0.5,
    'cheek_enabled': False,
    'cheek_color': "#FF4D4D",
    'cheek_opacity': 0.3,
    'cheek_center_x': 0.0,
    'cheek_center_y': 0.0,
    'cheek_radius_x': 0.6,
    'cheek_radius_y': 0.5,
    'cheek_blur_strength': 0.1,
    'eyelid_enabled': False,
    'eyelid_color': "#E0588C",
    'eyelid_opacity': 0.3,
    'eyelid_glitter': 0.0,
    'eyelid_zone': "全体",
    'eyelid_edge_enabled': False,
    'eyelid_edge_color': "#F52929",
    'eyelid_edge_opacity': 0.5,
    'eyelid_edge_glitter': 0.0,
    'eyelid_edge_zone': "全体",
    'eyeliner_enabled': False,
    'eyeliner_color': "#000000",
    'eyeliner_thickness': 2,
    'tear_bag_enabled': False,
    'tear_bag_highlight_color': "#FFF6E6",
    'tear_bag_highlight_opacity': 0.3,
    'tear_bag_shadow_color': "#8C6E5A",
    'tear_bag_shadow_opacity': 0.4,
    'tear_bag_shadow_thickness': 2,
    'tear_bag_size': 1.0,
    'eyeiris_enabled': False,
    'eyeiris_color': "#6496FF",
    'eyeiris_opacity': 0.5,
    'eyeiris_separate_colors': False,
    'eyeiris_color_left': "#6496FF",
    'eyeiris_color_right': "#6496FF",
}
for _key, _default in _SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ========================================
# Sidebar
# ========================================

with st.sidebar:
    st.title("💄 メイク設定")

    st.markdown("---")

    with st.expander("📖 使い方", expanded=True):  
    # Usage instructions
        st.markdown("""
        1. 顔写真をアップロード
        2. 各パラメータを調整（🎨ファンデ、💋リップ、🌸チーク、👁️アイシャドウ、✏️アイライナー、👁️カラコン）
        3. 「メイクを適用」ボタンをクリック
        4. 結果を確認・ダウンロード
        """)

    # Foundation settings
    with st.expander("🎨 ファンデーション", expanded=False):
        foundation_enabled = st.checkbox(
            "ファンデーションを有効化",
            value=st.session_state.foundation_enabled,
            key="foundation_enabled"
        )

        if foundation_enabled:
            foundation_color = st.color_picker(
                "ファンデーションカラー",
                value=st.session_state.foundation_color,
                key="foundation_color"
            )
            st.caption(f"RGB: {hex_to_rgb(foundation_color)}")

            st.slider(
                "不透明度",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.foundation_opacity,
                step=0.05,
                key="foundation_opacity",
                help="メイクの強さ（0: 透明、1: 最大）"
            )

    # Lips settings
    with st.expander("💋 リップ", expanded=False):
        lips_enabled = st.checkbox(
            "リップを有効化",
            value=st.session_state.lips_enabled,
            key="lips_enabled"
        )

        if lips_enabled:
            lips_color = st.color_picker(
                "リップカラー",
                value=st.session_state.lips_color,
                key="lips_color"
            )
            st.caption(f"RGB: {hex_to_rgb(lips_color)}")

            st.slider(
                "不透明度",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.lips_opacity,
                step=0.1,
                key="lips_opacity",
                help="メイクの強さ（0: 透明、1: 最大）"
            )

    # Cheek settings
    with st.expander("🌸 チーク", expanded=False):
        cheek_enabled = st.checkbox(
            "チークを有効化",
            value=st.session_state.cheek_enabled,
            key="cheek_enabled"
        )

        if cheek_enabled:
            cheek_color = st.color_picker(
                "チークカラー",
                value=st.session_state.cheek_color,
                key="cheek_color"
            )
            st.caption(f"RGB: {hex_to_rgb(cheek_color)}")

            st.slider(
                "不透明度",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.cheek_opacity,
                step=0.1,
                key="cheek_opacity"
            )

            st.markdown("**楕円形状**")
            col1, col2 = st.columns(2)
            with col1:
                st.slider(
                    "中心 X",
                    min_value=-0.5,
                    max_value=0.5,
                    value=st.session_state.cheek_center_x,
                    step=0.1,
                    key="cheek_center_x",
                    help="水平方向の中心位置"
                )
            with col2:
                st.slider(
                    "中心 Y",
                    min_value=-0.5,
                    max_value=0.5,
                    value=st.session_state.cheek_center_y,
                    step=0.1,
                    key="cheek_center_y",
                    help="垂直方向の中心位置"
                )

            col1, col2 = st.columns(2)
            with col1:
                st.slider(
                    "半径 X",
                    min_value=0.1,
                    max_value=1.0,
                    value=st.session_state.cheek_radius_x,
                    step=0.1,
                    key="cheek_radius_x",
                    help="水平方向の半径"
                )
            with col2:
                st.slider(
                    "半径 Y",
                    min_value=0.1,
                    max_value=1.0,
                    value=st.session_state.cheek_radius_y,
                    step=0.1,
                    key="cheek_radius_y",
                    help="垂直方向の半径"
                )

            st.slider(
                "ぼかし強度",
                min_value=0.1,
                max_value=0.3,
                value=st.session_state.cheek_blur_strength,
                step=0.05,
                key="cheek_blur_strength",
                help="エッジのぼかし強度（画像サイズに対する割合）"
            )

    # Eye shadow settings
    with st.expander("👁️ アイシャドウ", expanded=False):
        eyelid_enabled = st.checkbox(
            "アイシャドウを有効化",
            value=st.session_state.eyelid_enabled,
            key="eyelid_enabled"
        )

        if eyelid_enabled:
            eyelid_color = st.color_picker(
                "アイシャドウカラー",
                value=st.session_state.eyelid_color,
                key="eyelid_color"
            )
            st.caption(f"RGB: {hex_to_rgb(eyelid_color)}")

            st.selectbox(
                "塗り分け（部位）",
                options=_EYELID_ZONE_LABELS,
                key="eyelid_zone",
                help="まぶたを横方向に分割して塗る部位（全体/目尻側/中心部/目頭側）"
            )

            st.slider(
                "不透明度",
                min_value=0.1,
                max_value=0.4,
                value=st.session_state.eyelid_opacity,
                step=0.05,
                key="eyelid_opacity",
                help="メイクの強さ（0: 透明、1: 最大）"
            )

            st.slider(
                "グリッター（ラメ）",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.eyelid_glitter,
                step=0.05,
                key="eyelid_glitter",
                help="ラメ効果の強さ（0: なし、1: 最大）"
            )

    # Eye shadow (edge) settings
    with st.expander("👁️ アイシャドウ（エッジ）", expanded=False):
        eyelid_edge_enabled = st.checkbox(
            "アイシャドウ（エッジ）を有効化",
            value=st.session_state.eyelid_edge_enabled,
            key="eyelid_edge_enabled",
            help="広範囲に適用されるアイシャドウ"
        )

        if eyelid_edge_enabled:
            eyelid_edge_color = st.color_picker(
                "アイシャドウ（エッジ）カラー",
                value=st.session_state.eyelid_edge_color,
                key="eyelid_edge_color"
            )
            st.caption(f"RGB: {hex_to_rgb(eyelid_edge_color)}")

            st.selectbox(
                "塗り分け（部位）",
                options=_EYELID_ZONE_LABELS,
                key="eyelid_edge_zone",
                help="目の淵を横方向に分割して塗る部位（全体/目尻側/中心部/目頭側）"
            )

            st.slider(
                "不透明度",
                min_value=0.1,
                max_value=0.8,
                value=st.session_state.eyelid_edge_opacity,
                step=0.1,
                key="eyelid_edge_opacity",
                help="メイクの強さ（0: 透明、1: 最大）"
            )

            st.slider(
                "グリッター（ラメ）",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.eyelid_edge_glitter,
                step=0.05,
                key="eyelid_edge_glitter",
                help="ラメ効果の強さ（0: なし、1: 最大）"
            )

    # Eyeliner settings
    with st.expander("✏️ アイライナー", expanded=False):
        eyeliner_enabled = st.checkbox(
            "アイライナーを有効化",
            value=st.session_state.eyeliner_enabled,
            key="eyeliner_enabled"
        )

        if eyeliner_enabled:
            eyeliner_color = st.color_picker(
                "アイライナーカラー",
                value=st.session_state.eyeliner_color,
                key="eyeliner_color"
            )
            st.caption(f"RGB: {hex_to_rgb(eyeliner_color)}")

            st.slider(
                "太さ",
                min_value=1,
                max_value=10,
                value=st.session_state.eyeliner_thickness,
                step=1,
                key="eyeliner_thickness",
                help="アイライナーの線の太さ"
            )

    # Tear bag settings
    with st.expander("👁️ 涙袋", expanded=False):
        tear_bag_enabled = st.checkbox(
            "涙袋を有効化",
            value=st.session_state.tear_bag_enabled,
            key="tear_bag_enabled"
        )

        if tear_bag_enabled:
            st.slider(
                "涙袋のサイズ",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.tear_bag_size,
                step=0.05,
                key="tear_bag_size",
                help="涙袋の膨らみ（0: なし、1: 最大）。ハイライト・シャドウ両方に反映されます"
            )

            st.markdown("**ハイライト**")
            tear_bag_highlight_color = st.color_picker(
                "ハイライトカラー",
                value=st.session_state.tear_bag_highlight_color,
                key="tear_bag_highlight_color"
            )
            st.caption(f"RGB: {hex_to_rgb(tear_bag_highlight_color)}")

            st.slider(
                "ハイライト不透明度",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.tear_bag_highlight_opacity,
                step=0.05,
                key="tear_bag_highlight_opacity",
                help="涙袋の明るさ（0: なし、1: 最大）"
            )

            st.markdown("**シャドウ**")
            tear_bag_shadow_color = st.color_picker(
                "シャドウカラー",
                value=st.session_state.tear_bag_shadow_color,
                key="tear_bag_shadow_color"
            )
            st.caption(f"RGB: {hex_to_rgb(tear_bag_shadow_color)}")

            st.slider(
                "シャドウ不透明度",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.tear_bag_shadow_opacity,
                step=0.05,
                key="tear_bag_shadow_opacity",
                help="シャドウラインの濃さ（0: なし、1: 最大）"
            )

            st.slider(
                "シャドウ太さ",
                min_value=1,
                max_value=10,
                value=st.session_state.tear_bag_shadow_thickness,
                step=1,
                key="tear_bag_shadow_thickness",
                help="シャドウラインの太さ"
            )

    # Eye iris (color contact) settings
    with st.expander("👁️ カラーコンタクト", expanded=False):
        eyeiris_enabled = st.checkbox(
            "カラーコンタクトを有効化",
            value=st.session_state.eyeiris_enabled,
            key="eyeiris_enabled"
        )

        if eyeiris_enabled:
            st.checkbox(
                "左右別々に設定",
                value=st.session_state.eyeiris_separate_colors,
                key="eyeiris_separate_colors",
                on_change=_on_separate_colors_change
            )

            if st.session_state.eyeiris_separate_colors:
                col_left, col_right = st.columns(2)
                with col_left:
                    eyeiris_color_left = st.color_picker(
                        "左目カラー",
                        value=st.session_state.eyeiris_color_left,
                        key="eyeiris_color_left"
                    )
                    st.caption(f"RGB: {hex_to_rgb(eyeiris_color_left)}")
                with col_right:
                    eyeiris_color_right = st.color_picker(
                        "右目カラー",
                        value=st.session_state.eyeiris_color_right,
                        key="eyeiris_color_right"
                    )
                    st.caption(f"RGB: {hex_to_rgb(eyeiris_color_right)}")
            else:
                eyeiris_color = st.color_picker(
                    "虹彩カラー",
                    value=st.session_state.eyeiris_color,
                    key="eyeiris_color"
                )
                st.caption(f"RGB: {hex_to_rgb(eyeiris_color)}")

            st.slider(
                "不透明度",
                min_value=0.25,
                max_value=0.85,
                value=st.session_state.eyeiris_opacity,
                step=0.05,
                key="eyeiris_opacity",
                help="カラーコンタクトの強さ（0: 透明、1: 最大）"
            )

    # Debug mode
    st.markdown("---")
    st.subheader("デバッグ設定")
    debug_mode = st.checkbox(
        "デバッグモード",
        value=False,
        help="有効にすると outputs/debug/ にデバッグ画像を保存します"
    )

    # Reset button
    st.markdown("---")
    if st.button("設定をリセット", use_container_width=True):
        for key, default in _SESSION_DEFAULTS.items():
            st.session_state[key] = default
        if 'result_image' in st.session_state:
            del st.session_state['result_image']
        st.rerun()



# ========================================
# Main Area
# ========================================

st.title("💄 メイクシミュレーター")
st.markdown("**パラメータを調整して、オリジナルメイクを作成しましょう！**")

# Image input method selection
input_method = st.radio(
    "画像の入力方法を選択",
    ["ファイルアップロード", "カメラ", "URL"],
    horizontal=True
)
uploaded_file = None
camera_image = None
url_image = None

if input_method == "ファイルアップロード":
    _accepted_types = ["png", "jpg", "jpeg", "webp"]
    if HEIF_SUPPORTED:
        _accepted_types.extend(["heic", "heif"])
    uploaded_file = st.file_uploader(
        "顔写真をアップロード",
        type=_accepted_types,
        help="推奨: 正面の顔写真、解像度 256x256 ~ 2048x2048"
    )
elif input_method == "カメラ":
    camera_image = st.camera_input("写真を撮影")
elif input_method == "URL":
    image_url = st.text_input(
        "画像のURLを入力",
        placeholder="https://example.com/image.jpg"
    )
    if image_url and st.button("読み込む"):
        try:
            if not image_url.startswith(('http://', 'https://')):
                st.error("有効な HTTP/HTTPS URLを入力してください")
            else:
                url_image = fetch_image_safe(image_url)
        except ValueError as e:
            st.error(str(e))
        except requests.exceptions.Timeout:
            st.error("リクエストがタイムアウトしました。再度お試しください。")
        except requests.exceptions.RequestException:
            st.error("画像の読み込みに失敗しました。URLを確認してください。")
        except Exception:
            st.error("予期しないエラーが発生しました。")


image_source = uploaded_file or camera_image or url_image

if image_source is not None:
    try:
        pil_image = Image.open(image_source)
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")
        image_np = np.array(pil_image)

        # Two column layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("オリジナル")
            st.caption(f"画像サイズ: {image_np.shape[1]} x {image_np.shape[0]} px")
            st.image(pil_image, use_container_width=True)

        with col2:
            st.subheader("メイク後")

        # Apply makeup button - centered
        _btn_left, _btn_center, _btn_right = st.columns([1, 2, 1])
        with _btn_center:
            apply_clicked = st.button("メイクを適用", use_container_width=True, type="primary")

        if apply_clicked:
            with st.spinner("メイクを適用中..."):
                try:
                    # Get processor (cached)
                    processor = get_processor()

                    # Open image and get context
                    ctx = processor.open(image_np)

                    # Check if at least one makeup option is enabled
                    any_enabled = (
                        st.session_state.foundation_enabled or
                        st.session_state.lips_enabled or
                        st.session_state.cheek_enabled or
                        st.session_state.eyelid_enabled or
                        st.session_state.eyelid_edge_enabled or
                        st.session_state.eyeliner_enabled or
                        st.session_state.tear_bag_enabled or
                        st.session_state.eyeiris_enabled
                    )

                    if not any_enabled:
                        st.info("少なくとも1つのメイクオプションを有効にしてください")
                    elif ctx.face_count == 0:
                        st.warning("顔が検出されませんでした。正面顔写真をご使用ください。")
                    else:
                        # Get the first face
                        face = ctx.faces[0]

                        # Apply foundation (base layer, priority 0)
                        if st.session_state.foundation_enabled:
                            face.foundation(
                                color=hex_to_rgb(st.session_state.foundation_color),
                                opacity=st.session_state.foundation_opacity,
                                blur_strength=0.5
                            )

                        # Apply lips
                        if st.session_state.lips_enabled:
                            face.lip(
                                color=hex_to_rgb(st.session_state.lips_color),
                                opacity=st.session_state.lips_opacity,
                                blur_strength=0.5
                            )

                        # Apply cheek
                        if st.session_state.cheek_enabled:
                            face.cheek(
                                color=hex_to_rgb(st.session_state.cheek_color),
                                opacity=st.session_state.cheek_opacity,
                                center=(st.session_state.cheek_center_x, st.session_state.cheek_center_y),
                                radius=(st.session_state.cheek_radius_x, st.session_state.cheek_radius_y),
                                blur_strength=st.session_state.cheek_blur_strength,
                                side='both'
                            )

                        # Apply eye shadow
                        if st.session_state.eyelid_enabled:
                            face.eyelid(
                                color=hex_to_rgb(st.session_state.eyelid_color),
                                opacity=st.session_state.eyelid_opacity,
                                blur_strength=0.25,
                                glitter=st.session_state.eyelid_glitter,
                                zone=_EYELID_ZONE_OPTIONS[st.session_state.eyelid_zone],
                                side='both'
                            )

                        # Apply eye shadow edge
                        if st.session_state.eyelid_edge_enabled:
                            face.eyelid_edge(
                                color=hex_to_rgb(st.session_state.eyelid_edge_color),
                                opacity=st.session_state.eyelid_edge_opacity,
                                blur_strength=0.25,
                                glitter=st.session_state.eyelid_edge_glitter,
                                zone=_EYELID_ZONE_OPTIONS[st.session_state.eyelid_edge_zone],
                                side='both'
                            )

                        # Apply eyeliner
                        if st.session_state.eyeliner_enabled:
                            face.eyeliner(
                                color=hex_to_rgb(st.session_state.eyeliner_color),
                                opacity=0.8,
                                thickness=st.session_state.eyeliner_thickness,
                                blur_strength=0.5,
                                side='both'
                            )

                        # Apply tear bag
                        if st.session_state.tear_bag_enabled:
                            face.tear_bag_highlight(
                                color=hex_to_rgb(st.session_state.tear_bag_highlight_color),
                                opacity=st.session_state.tear_bag_highlight_opacity,
                                blur_strength=0.2,
                                size=st.session_state.tear_bag_size,
                                side='both'
                            )
                            face.tear_bag_shadow(
                                color=hex_to_rgb(st.session_state.tear_bag_shadow_color),
                                opacity=st.session_state.tear_bag_shadow_opacity,
                                thickness=st.session_state.tear_bag_shadow_thickness,
                                blur_strength=0.5,
                                size=st.session_state.tear_bag_size,
                                side='both'
                            )

                        # Apply eye iris (color contact)
                        if st.session_state.eyeiris_enabled:
                            if st.session_state.eyeiris_separate_colors:
                                face.eyeiris(
                                    color=hex_to_rgb(st.session_state.eyeiris_color_left),
                                    opacity=st.session_state.eyeiris_opacity,
                                    blur_strength=0.5,
                                    side='left'
                                )
                                face.eyeiris(
                                    color=hex_to_rgb(st.session_state.eyeiris_color_right),
                                    opacity=st.session_state.eyeiris_opacity,
                                    blur_strength=0.5,
                                    side='right'
                                )
                            else:
                                face.eyeiris(
                                    color=hex_to_rgb(st.session_state.eyeiris_color),
                                    opacity=st.session_state.eyeiris_opacity,
                                    blur_strength=0.5,
                                    side='both'
                                )

                        # Render result
                        result_np = ctx.render(debug=debug_mode)
                        st.session_state['result_image'] = result_np

                        # Display face detection count info
                        if ctx.face_count > 1:
                            st.info(f"{ctx.face_count}つの顔を検出しました。最初の顔のみにメイクを適用しています。")

                except ValueError as e:
                    st.error(f"無効な入力です: {str(e)}")
                except Exception as e:
                    st.error(f"メイク処理に失敗しました: {str(e)}")

        if 'result_image' in st.session_state:
            with col2:
                result_image = st.session_state['result_image']
                result_pil = Image.fromarray(result_image)

                st.image(result_pil, use_container_width=True)

                # Download button
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                buf.seek(0)

                st.download_button(
                    label="結果をダウンロード",
                    data=buf,
                    file_name=f"makeup_result_{st.session_state['session_id']}.png",
                    mime="image/png",
                    use_container_width=True
                )

                # Current settings info
                with st.expander("現在の設定"):
                    if st.session_state.foundation_enabled:
                        st.write("**ファンデーション**")
                        st.write(f"- カラー: {st.session_state.foundation_color} {hex_to_rgb(st.session_state.foundation_color)}")
                        st.write(f"- 不透明度: {st.session_state.foundation_opacity}")

                    if st.session_state.lips_enabled:
                        st.write("**リップ**")
                        st.write(f"- カラー: {st.session_state.lips_color} {hex_to_rgb(st.session_state.lips_color)}")
                        st.write(f"- 不透明度: {st.session_state.lips_opacity}")

                    if st.session_state.cheek_enabled:
                        st.write("**チーク**")
                        st.write(f"- カラー: {st.session_state.cheek_color} {hex_to_rgb(st.session_state.cheek_color)}")
                        st.write(f"- 不透明度: {st.session_state.cheek_opacity}")
                        st.write(f"- 中心: ({st.session_state.cheek_center_x:.2f}, {st.session_state.cheek_center_y:.2f})")
                        st.write(f"- 半径: ({st.session_state.cheek_radius_x:.2f}, {st.session_state.cheek_radius_y:.2f})")
                        st.write(f"- ぼかし強度: {st.session_state.cheek_blur_strength:.2f}")

                    if st.session_state.eyelid_enabled:
                        st.write("**アイシャドウ**")
                        st.write(f"- カラー: {st.session_state.eyelid_color} {hex_to_rgb(st.session_state.eyelid_color)}")
                        st.write(f"- 部位: {st.session_state.eyelid_zone}")
                        st.write(f"- 不透明度: {st.session_state.eyelid_opacity}")
                        st.write(f"- グリッター: {st.session_state.eyelid_glitter:.2f}")

                    if st.session_state.eyelid_edge_enabled:
                        st.write("**アイシャドウ（エッジ）**")
                        st.write(f"- カラー: {st.session_state.eyelid_edge_color} {hex_to_rgb(st.session_state.eyelid_edge_color)}")
                        st.write(f"- 部位: {st.session_state.eyelid_edge_zone}")
                        st.write(f"- 不透明度: {st.session_state.eyelid_edge_opacity}")
                        st.write(f"- グリッター: {st.session_state.eyelid_edge_glitter:.2f}")

                    if st.session_state.eyeliner_enabled:
                        st.write("**アイライナー**")
                        st.write(f"- カラー: {st.session_state.eyeliner_color} {hex_to_rgb(st.session_state.eyeliner_color)}")
                        st.write(f"- 太さ: {st.session_state.eyeliner_thickness}")

                    if st.session_state.tear_bag_enabled:
                        st.write("**涙袋**")
                        st.write(f"- ハイライト: {st.session_state.tear_bag_highlight_color} {hex_to_rgb(st.session_state.tear_bag_highlight_color)}")
                        st.write(f"- シャドウ: {st.session_state.tear_bag_shadow_color} {hex_to_rgb(st.session_state.tear_bag_shadow_color)}")
                        st.write(f"- シャドウ太さ: {st.session_state.tear_bag_shadow_thickness}")
                        st.write(f"- サイズ: {st.session_state.tear_bag_size:.2f}")

                    if st.session_state.eyeiris_enabled:
                        st.write("**カラーコンタクト**")
                        if st.session_state.eyeiris_separate_colors:
                            st.write(f"- 左目カラー: {st.session_state.eyeiris_color_left} {hex_to_rgb(st.session_state.eyeiris_color_left)}")
                            st.write(f"- 右目カラー: {st.session_state.eyeiris_color_right} {hex_to_rgb(st.session_state.eyeiris_color_right)}")
                        else:
                            st.write(f"- カラー: {st.session_state.eyeiris_color} {hex_to_rgb(st.session_state.eyeiris_color)}")
                        st.write(f"- 不透明度: {st.session_state.eyeiris_opacity}")

    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {str(e)}")

else:
    # No image selected
    st.info("画像を選択してください")

    with st.expander("パラメータガイド", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **ファンデーション**
            - カラー / 不透明度

            **リップ**
            - カラー / 不透明度

            **チーク**
            - カラー / 不透明度
            - 中心・半径 / ぼかし
            """)

        with col2:
            st.markdown("""
            **アイシャドウ**
            - カラー / 不透明度

            **アイシャドウ（エッジ）**
            - カラー / 不透明度
            """)

        with col3:
            st.markdown("""
            **アイライナー**
            - カラー / 太さ

            **カラーコンタクト**
            - カラー / 不透明度
            """)

# ========================================
# Footer
# ========================================

st.markdown("---")
st.caption("💄 メイクシミュレーター | Powered by MediaPipe & OpenCV")