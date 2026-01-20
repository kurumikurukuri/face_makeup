"""
メイクシミュレーター - Streamlit UI

使用方法:
    streamlit run apps/streamlit_app.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
from datetime import datetime


from src.face_makeup.processor import MakeupProcessor

# ========================================
# ヘルパー関数
# ========================================

def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

@st.cache_resource
def get_processor():
    """キャッシュされた MakeupProcessor インスタンスを取得"""
    return MakeupProcessor()


# ========================================
# ページ設定
# ========================================

st.set_page_config(
    page_title="メイクシミュレーター",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# セッションステート初期化
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if 'lips_enabled' not in st.session_state:
    st.session_state.lips_enabled = False
    st.session_state.lips_color = "#FF4D4D"
    st.session_state.lips_opacity = 0.5

    st.session_state.cheek_enabled = False
    st.session_state.cheek_color = "#FF4D4D"
    st.session_state.cheek_opacity = 0.3
    st.session_state.cheek_center_x = 0.0
    st.session_state.cheek_center_y = 0.0
    st.session_state.cheek_radius_x = 0.6
    st.session_state.cheek_radius_y = 0.5
    st.session_state.cheek_falloff = 0.4

    st.session_state.eyelid_enabled = False
    st.session_state.eyelid_color = "#E0588C"
    st.session_state.eyelid_opacity = 0.1

    st.session_state.eyelid_edge_enabled = False
    st.session_state.eyelid_edge_color = "#F52929"
    st.session_state.eyelid_edge_opacity = 0.3

    st.session_state.eyeliner_enabled = False
    st.session_state.eyeliner_color = "#000000"
    st.session_state.eyeliner_opacity = 0.8
    st.session_state.eyeliner_thickness = 2


# ========================================
# サイドバー
# ========================================

with st.sidebar:
    st.title("💄 メイク設定")

    st.markdown("---")

    # リップ設定
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

    # チーク設定
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
                "フェード",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.cheek_falloff,
                step=0.1,
                key="cheek_falloff",
                help="小さいほど中心集中、大きいほど広がる"
            )

    # アイシャドウ設定
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
                min_value=0.0,
                max_value=0.3,
                value=st.session_state.eyelid_opacity,
                step=0.05,
                key="eyelid_opacity",
                help="メイクの強さ（0: 透明、1: 最大）"
            )

    # アイシャドウ（エッジ）設定
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
                min_value=0.0,
                max_value=0.8,
                value=st.session_state.eyelid_edge_opacity,
                step=0.1,
                key="eyelid_edge_opacity",
                help="メイクの強さ（0: 透明、1: 最大）"
            )

    # アイライナー設定
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

    # デバッグモード
    st.markdown("---")
    st.subheader("デバッグ設定")
    debug_mode = st.checkbox(
        "デバッグモード",
        value=False,
        help="有効にすると outputs/debug/ にデバッグ画像を保存します"
    )

    # 使用方法
    st.markdown("---")
    st.subheader("📖 使い方")
    st.markdown("""
    1. 顔写真をアップロード
    2. 各パラメータを調整（💋リップ、🌸チーク、👁️アイシャドウ、✏️ アイライナー）
    3. 「メイクを適用」ボタンをクリック
    4. 結果を確認・ダウンロード
    """)


# ========================================
# メインエリア
# ========================================

st.title("💄 メイク シミュレーター")
st.markdown("**パラメータを調整して、オリジナルメイクを作成しましょう！**")

# 画像入力方法の選択
input_method = st.radio(
    "Select image input method",
    ["File Upload", "Camera", "URL"],
    horizontal=True
)
uploaded_file = None
camera_image = None
url_image = None

if input_method == "File Upload":
    uploaded_file = st.file_uploader(
        "Upload a face photo",
        type=["png", "jpg", "jpeg"],
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
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            url_image = io.BytesIO(response.content)
        except Exception as e:
            raise


image_source = uploaded_file or camera_image or url_image

if image_source is not None:
    try:
        pil_image = Image.open(image_source).convert("RGB")
        image_np = np.array(pil_image)

        st.info(f"Image size: {image_np.shape[1]} × {image_np.shape[0]} px")

        # 2カラムレイアウト
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(pil_image, use_container_width=True)

        with col2:
            st.subheader("With Makeup")

        # メイク適用ボタン
        if st.button("🎨 Apply Makeup", use_container_width=True, type="primary"):
            with st.spinner("Applying makeup..."):
                try:
                    # プロセッサ取得（キャッシュ）
                    processor = get_processor()

                    # 画像を開きコンテキストを取得
                    ctx = processor.open(image_np)

                    # メイクが1つでも有効か確認
                    any_enabled = (
                        st.session_state.lips_enabled or
                        st.session_state.cheek_enabled or
                        st.session_state.eyelid_enabled or
                        st.session_state.eyelid_edge_enabled or
                        st.session_state.eyeliner_enabled
                    )

                    if not any_enabled:
                        st.info("Please enable at least one makeup option")
                    else:
                        # 最初の顔を取得
                        face = ctx.faces[0]

                        # リップ適用
                        if st.session_state.lips_enabled:
                            face.lip(
                                color=hex_to_rgb(st.session_state.lips_color),
                                opacity=st.session_state.lips_opacity
                            )

                        # チーク適用
                        if st.session_state.cheek_enabled:
                            face.cheek(
                                color=hex_to_rgb(st.session_state.cheek_color),
                                opacity=st.session_state.cheek_opacity,
                                center=(st.session_state.cheek_center_x, st.session_state.cheek_center_y),
                                radius=(st.session_state.cheek_radius_x, st.session_state.cheek_radius_y),
                                falloff=st.session_state.cheek_falloff,
                                side='both'
                            )

                        # アイシャドウ適用
                        if st.session_state.eyelid_enabled:
                            face.eyelid(
                                color=hex_to_rgb(st.session_state.eyelid_color),
                                opacity=st.session_state.eyelid_opacity,
                                side='both'
                            )

                        # アイシャドウエッジ適用
                        if st.session_state.eyelid_edge_enabled:
                            face.eyelid_edge(
                                color=hex_to_rgb(st.session_state.eyelid_edge_color),
                                opacity=st.session_state.eyelid_edge_opacity,
                                side='both'
                            )

                        # アイライナー適用
                        if st.session_state.eyeliner_enabled:
                            face.eyeliner(
                                color=hex_to_rgb(st.session_state.eyeliner_color),
                                opacity=st.session_state.eyeliner_opacity,
                                thickness=st.session_state.eyeliner_thickness,
                                side='both'
                            )

                        # 結果をレンダリング
                        result_np = ctx.render(debug=debug_mode)
                        st.session_state['result_image'] = result_np

                        # 顔の検出数情報を表示
                        if ctx.face_count > 1:
                            st.info(f"Detected {ctx.face_count} faces. Makeup applied to first face only.")

                except Exception as e:
                    raise

        if 'result_image' in st.session_state:
            with col2:
                result_image = st.session_state['result_image']
                result_pil = Image.fromarray(result_image)

                st.image(result_pil, use_container_width=True)

                # ダウンロードボタン
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

                # 現在の設定情報
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
                        st.write(f"- Falloff: {st.session_state.cheek_falloff:.2f}")

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
                        st.write(f"- Opacity: {st.session_state.eyeliner_opacity}")
                        st.write(f"- Thickness: {st.session_state.eyeliner_thickness}")

    except Exception as e:
        raise

else:
    # 画像が選択されていない
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
        """)

    # 技術情報
    st.markdown("---")
    st.subheader("🔧 Tech Stack")
    st.markdown("""
    - **Face Detection & Landmarks**: MediaPipe Face Detection & Face Mesh (478 points)
    - **Face segmentation**: https://huggingface.co/jonathandinu/face-parsing
    - **Image Processing**: OpenCV & PIL & numpy
    - **UI**: Streamlit
    """)

# ========================================
# フッター
# ========================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💄 メイクシミュレーター | Powered by MediaPipe & OpenCV</p>
</div>
""", unsafe_allow_html=True)
