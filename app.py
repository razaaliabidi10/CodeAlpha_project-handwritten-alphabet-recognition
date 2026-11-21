 
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="LetterAI | Handwriting Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = Path("model/handwritten_alphabet_model.keras")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 10%, rgba(48, 85, 180, 0.24), transparent 38%),
        radial-gradient(ellipse at 90% 35%, rgba(112, 54, 180, 0.20), transparent 35%),
        linear-gradient(135deg, #080D1B 0%, #10182D 50%, #090F20 100%);
    color: #F4F7FF;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    font-family: 'Inter', sans-serif !important;
}

.hero {
    padding: 35px 10px 28px 10px;
}

.hero-tag {
    display: inline-block;
    padding: 7px 13px;
    border: 1px solid rgba(125, 170, 255, 0.35);
    border-radius: 30px;
    color: #A9C8FF;
    background: rgba(65, 105, 190, 0.13);
    font-size: 12px;
    letter-spacing: 1.4px;
    font-weight: 700;
}

.hero h1 {
    font-size: clamp(36px, 6vw, 62px);
    line-height: 1.12;
    font-weight: 800;
    letter-spacing: -2px;
    color: #FFFFFF;
    margin: 22px 0 12px 0;
}

.gradient-text {
    background: linear-gradient(90deg, #80B7FF, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #A9B7D0;
    font-size: 16px;
    line-height: 1.7;
    max-width: 650px;
}

.status {
    display: inline-block;
    color: #70E4B0;
    border: 1px solid rgba(76, 220, 155, 0.3);
    background: rgba(43, 180, 120, 0.10);
    padding: 8px 13px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.055);
    border: 1px solid rgba(160, 185, 230, 0.18);
    border-radius: 20px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}

[data-testid="stTabs"] button {
    color: #AAB9D2 !important;
    font-weight: 600;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #8DBAFF !important;
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #78AFFF;
}

.stButton > button {
    background: linear-gradient(100deg, #3978E8, #7655DB);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 700;
    min-height: 45px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border: 1px solid #A6C8FF;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(74, 115, 230, 0.3);
}

.stText, .stMarkdown, [data-testid="stMetricLabel"] {
    color: #D9E3F5;
}

[data-testid="stMetricValue"] {
    color: #91BCFF;
}

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.035);
    border: 1px dashed rgba(140, 175, 230, 0.4);
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4389F5, #A17BFA);
}

.footer {
    color: #8291AC;
    font-size: 12px;
    text-align: center;
    padding-top: 30px;
    letter-spacing: 1px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, threshold = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    points = cv2.findNonZero(threshold)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)
    letter = threshold[y:y+h, x:x+w]

    letter = cv2.copyMakeBorder(
        letter, 5, 5, 5, 5,
        cv2.BORDER_CONSTANT,
        value=0
    )

    h, w = letter.shape
    scale = 20 / max(h, w)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    letter = cv2.resize(letter, (new_w, new_h))

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = letter

    return canvas.astype("float32") / 255.0


def show_prediction(processed, model):
    prediction = model.predict(
        processed.reshape(1, 28, 28, 1),
        verbose=0
    )[0]

    top_indices = np.argsort(prediction)[::-1][:3]

    predicted_letter = chr(top_indices[0] + 65)
    confidence = prediction[top_indices[0]] * 100

    st.markdown("### ✨ Recognition Result")

    result_col, details_col = st.columns([1, 2])

    with result_col:
        st.markdown(
            f"""
            <div style="
                background:rgba(78,130,230,0.13);
                border:1px solid rgba(130,175,255,0.3);
                border-radius:20px;
                padding:25px;
                text-align:center;
            ">
                <div style="color:#AABBD7;font-size:12px;">
                    PREDICTED LETTER
                </div>
                <div style="
                    color:#FFFFFF;
                    font-size:75px;
                    font-weight:800;
                    line-height:1.3;
                ">{predicted_letter}</div>
                <div style="color:#91BCFF;font-size:19px;font-weight:700;">
                    {confidence:.2f}%
                </div>
                <div style="color:#AABBD7;font-size:11px;">
                    MODEL CONFIDENCE
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with details_col:
        st.write("**Top 3 predictions**")

        for index in top_indices:
            letter = chr(index + 65)
            score = float(prediction[index])

            st.write(f"**{letter}** — {score * 100:.2f}%")
            st.progress(score)

        st.write("**Image processed for the model**")
        st.image(
            processed,
            caption="28 × 28 grayscale input",
            width=140
        )


st.markdown("""
<div class="hero">
    <span class="hero-tag">✦ MACHINE LEARNING PROJECT</span>
    <h1>Your handwriting,<br>
    <span class="gradient-text">decoded by AI.</span></h1>
    <p>
        An intelligent handwritten alphabet recognition system.
        Draw a letter or upload an image and let the CNN model
        recognize it.
    </p>
    <span class="status">● MODEL READY</span>
</div>
""", unsafe_allow_html=True)


try:
    model = load_model()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


st.markdown("### Explore the recognizer")

tab1, tab2 = st.tabs([
    "✍️  Draw a letter",
    "📁  Upload an image"
])


with tab1:

    with st.container(border=True):
        st.subheader("Draw your letter")
        st.caption("Use your mouse or touchscreen to draw one capital letter.")

        drawing = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=14,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=250,
            width=250,
            drawing_mode="freedraw",
            return_image_data=True,
            key="letter_canvas"
        )

        if st.button("✨ Recognize my letter", key="draw_predict"):

            if drawing.image_data is None:
                st.warning("Please draw a letter first.")

            else:
                rgba = drawing.image_data.astype("uint8")
                image = Image.fromarray(rgba, "RGBA").convert("RGB")

                processed = preprocess_image(image)

                if processed is None:
                    st.warning("No letter detected. Please try again.")
                else:
                    show_prediction(processed, model)


with tab2:

    with st.container(border=True):
        st.subheader("Upload your handwriting")
        st.caption("Upload an image containing one handwritten capital letter.")

        uploaded = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg"],
            key="upload_letter"
        )

        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")

            st.image(
                image,
                caption="Uploaded image",
                width=250
            )

            if st.button("✨ Recognize uploaded letter", key="upload_predict"):

                processed = preprocess_image(image)

                if processed is None:
                    st.warning("No letter detected. Please upload another image.")
                else:
                    show_prediction(processed, model)


st.markdown("""
<div class="footer">
    LETTERAI &nbsp;•&nbsp; POWERED BY CNN, TENSORFLOW & EMNIST
    <br><br>
    Built as a Machine Learning portfolio project.
</div>
""", unsafe_allow_html=True)
