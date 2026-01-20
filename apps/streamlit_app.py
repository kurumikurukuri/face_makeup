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

import pillow_heif
pillow_heif.register_heif_opener()
HEIF_SUPPORTED = True



from face_makeup import MakeupProcessor

# ========================================
# Helper Functions
# ========================================

def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


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

_SESSION_DEFAULTS = {
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
    'eyelid_edge_enabled': False,
    'eyelid_edge_color': "#F52929",
    'eyelid_edge_opacity': 0.5,
    'eyeliner_enabled': False,
    'eyeliner_color': "#000000",
    'eyeliner_thickness': 2,
    'eyeiris_enabled': False,
    'eyeiris_color': "#6496FF",
    'eyeiris_opacity': 0.5,
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

    # Lips settings
    with st.expander("💋 リップ", expanded=True):
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
    with st.expander("🌸 チーク", expanded=True):
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

            st.slider(
                "不透明度",
                min_value=0.1,
                max_value=0.4,
                value=st.session_state.eyelid_opacity,
                step=0.05,
                key="eyelid_opacity",
                help="メイクの強さ（0: 透明、1: 最大）"
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

            st.slider(
                "不透明度",
                min_value=0.1,
                max_value=0.8,
                value=st.session_state.eyelid_edge_opacity,
                step=0.1,
                key="eyelid_edge_opacity",
                help="メイクの強さ（0: 透明、1: 最大）"
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

    # Eye iris (color contact) settings
    with st.expander("👁️ カラーコンタクト", expanded=False):
        eyeiris_enabled = st.checkbox(
            "カラーコンタクトを有効化",
            value=st.session_state.eyeiris_enabled,
            key="eyeiris_enabled"
        )

        if eyeiris_enabled:
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

    # Usage instructions
    st.markdown("---")
    st.subheader("📖 使い方")
    st.markdown("""
    1. 顔写真をアップロード
    2. 各パラメータを調整（💋リップ、🌸チーク、👁️アイシャドウ、✏️アイライナー、👁️カラコン）
    3. 「メイクを適用」ボタンをクリック
    4. 結果を確認・ダウンロード
    """)


# ========================================
# Main Area
# ========================================

st.title("💄 メイクシミュレーター")
st.markdown("**パラメータを調整して、オリジナルメイクを作成しましょう！**")

# Image input method selection
input_method = st.radio(
    "Select image input method",
    ["File Upload", "Camera", "URL"],
    horizontal=True
)
uploaded_file = None
camera_image = None
url_image = None

if input_method == "File Upload":
    _accepted_types = ["png", "jpg", "jpeg", "webp"]
    if HEIF_SUPPORTED:
        _accepted_types.extend(["heic", "heif"])
    uploaded_file = st.file_uploader(
        "Upload a face photo",
        type=_accepted_types,
        help="Recommended: frontal face, resolution 256×256 ~ 2048×2048"
    )
elif input_method == "Camera":
    camera_image = st.camera_input("Take a photo")
elif input_method == "URL":
    image_url = st.text_input(
        "Enter image URL",
        placeholder="https://example.com/image.jpg"
    )
    if image_url and st.button("Load"):
        try:
            if not image_url.startswith(('http://', 'https://')):
                st.error("Please enter a valid HTTP/HTTPS URL")
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

        st.info(f"Image size: {image_np.shape[1]} × {image_np.shape[0]} px")

        # Two column layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(pil_image, use_container_width=True)

        with col2:
            st.subheader("With Makeup")

        # Apply makeup button
        if st.button("🎨 Apply Makeup", use_container_width=True, type="primary"):
            with st.spinner("Applying makeup..."):
                try:
                    # Get processor (cached)
                    processor = get_processor()

                    # Open image and get context
                    ctx = processor.open(image_np)

                    # Check if at least one makeup option is enabled
                    any_enabled = (
                        st.session_state.lips_enabled or
                        st.session_state.cheek_enabled or
                        st.session_state.eyelid_enabled or
                        st.session_state.eyelid_edge_enabled or
                        st.session_state.eyeliner_enabled or
                        st.session_state.eyeiris_enabled
                    )

                    if not any_enabled:
                        st.info("Please enable at least one makeup option")
                    else:
                        # Get the first face
                        face = ctx.faces[0]

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
                                side='both'
                            )

                        # Apply eye shadow edge
                        if st.session_state.eyelid_edge_enabled:
                            face.eyelid_edge(
                                color=hex_to_rgb(st.session_state.eyelid_edge_color),
                                opacity=st.session_state.eyelid_edge_opacity,
                                blur_strength=0.25,
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

                        # Apply eye iris (color contact)
                        if st.session_state.eyeiris_enabled:
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
                            st.info(f"Detected {ctx.face_count} faces. Makeup applied to first face only.")

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
                    label="📥 Download Result",
                    data=buf,
                    file_name=f"makeup_result_{st.session_state['session_id']}.png",
                    mime="image/png",
                    use_container_width=True
                )

                # Current settings info
                with st.expander("📊 Current Settings"):
                    if st.session_state.lips_enabled:
                        st.write("**Lips**")
                        st.write(f"- Color: {st.session_state.lips_color} {hex_to_rgb(st.session_state.lips_color)}")
                        st.write(f"- Opacity: {st.session_state.lips_opacity}")

                    if st.session_state.cheek_enabled:
                        st.write("**Cheek**")
                        st.write(f"- Color: {st.session_state.cheek_color} {hex_to_rgb(st.session_state.cheek_color)}")
                        st.write(f"- Opacity: {st.session_state.cheek_opacity}")
                        st.write(f"- Center: ({st.session_state.cheek_center_x:.2f}, {st.session_state.cheek_center_y:.2f})")
                        st.write(f"- Radius: ({st.session_state.cheek_radius_x:.2f}, {st.session_state.cheek_radius_y:.2f})")
                        st.write(f"- Blur Strength: {st.session_state.cheek_blur_strength:.2f}")

                    if st.session_state.eyelid_enabled:
                        st.write("**Eye Shadow**")
                        st.write(f"- Color: {st.session_state.eyelid_color} {hex_to_rgb(st.session_state.eyelid_color)}")
                        st.write(f"- Opacity: {st.session_state.eyelid_opacity}")

                    if st.session_state.eyelid_edge_enabled:
                        st.write("**Eye Shadow (Edge)**")
                        st.write(f"- Color: {st.session_state.eyelid_edge_color} {hex_to_rgb(st.session_state.eyelid_edge_color)}")
                        st.write(f"- Opacity: {st.session_state.eyelid_edge_opacity}")

                    if st.session_state.eyeliner_enabled:
                        st.write("**Eye Liner**")
                        st.write(f"- Color: {st.session_state.eyeliner_color} {hex_to_rgb(st.session_state.eyeliner_color)}")
                        st.write(f"- Thickness: {st.session_state.eyeliner_thickness}")

                    if st.session_state.eyeiris_enabled:
                        st.write("**Color Contact (Eye Iris)**")
                        st.write(f"- Color: {st.session_state.eyeiris_color} {hex_to_rgb(st.session_state.eyeiris_color)}")
                        st.write(f"- Opacity: {st.session_state.eyeiris_opacity}")

    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {str(e)}")

else:
    # No image selected
    st.info("👆 画像を選択してください")

    st.markdown("---")
    st.subheader("📝 Parameters Guide")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **💋 Lips**
        - **Color**: Lip color (RGB)
        - **Opacity**: Makeup strength (0~1)
        """)

    with col2:
        st.markdown("""
        **🌸 Cheek**
        - **Color**: Cheek color (RGB)
        - **Opacity**: Makeup strength (0~1)
        - **Center**: Ellipse center position
        - **Radius**: Ellipse size (horizontal/vertical)
        - **Falloff**: Blur strength
        """)

    with col3:
        st.markdown("""
        **👁️ Eye Shadow**
        - **Color**: Eye shadow color (RGB)
        - **Opacity**: Makeup strength (0~1)
        - Applies to eyelid area

        **👁️ Eye Shadow (Edge)**
        - Extended area eye shadow

        **✏️ Eye Liner**
        - **Color**: Eye liner color (RGB)
        - **Opacity**: Makeup strength (0~1)
        - **Thickness**: Line thickness (1~5)

        **👁️ Color Contact**
        - **Color**: Iris color (RGB)
        - **Opacity**: Contact strength (0~1)
        """)

    # Tech info
    st.markdown("---")
    st.subheader("🔧 Tech Stack")
    st.markdown("""
    - **Face Detection & Landmarks**: MediaPipe Face Detection & Face Mesh (478 points)
    - **Image Processing**: OpenCV & PIL & numpy
    - **UI**: Streamlit
    """)

# ========================================
# Footer
# ========================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💄 メイクシミュレーター | Powered by MediaPipe & OpenCV</p>
</div>
""", unsafe_allow_html=True)