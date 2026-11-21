 
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


# =========================
# 1. PREMIUM UI DESIGN
# =========================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 5% 5%, rgba(48,85,180,0.25), transparent 38%),
        radial-gradient(ellipse at 95% 35%, rgba(112,54,180,0.20), transparent 35%),
        linear-gradient(135deg, #080D1B, #10182D 50%, #090F20);
    color: #F4F7FF;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container,
[data-testid="stMainBlockContainer"] {
    max-width: 1500px !important;
    width: 94% !important;
    padding: 2rem 2rem 3rem 2rem !important;
    margin: auto !important;
}

h1, h2, h3, p, label {
    font-family: 'Inter', sans-serif !important;
}

.hero {
    padding: 35px 10px 30px 10px;
}

.hero-tag {
    display: inline-block;
    padding: 8px 15px;
    border: 1px solid rgba(125,170,255,0.35);
    border-radius: 30px;
    color: #A9C8FF;
    background: rgba(65,105,190,0.13);
    font-size: 12px;
    letter-spacing: 1.4px;
    font-weight: 700;
}

.hero h1 {
    font-size: clamp(38px, 5vw, 65px);
    line-height: 1.12;
    font-weight: 800;
    letter-spacing: -2px;
    color: white;
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
    max-width: 750px;
}

.status {
    display: inline-block;
    color: #70E4B0;
    border: 1px solid rgba(76,220,155,0.3);
    background: rgba(43,180,120,0.10);
    padding: 8px 14px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(24,37,67,0.48) !important;
    border: 1px solid rgba(155,185,255,0.22) !important;
    border-radius: 22px !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.18);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
}

[data-testid="stTabs"] {
    width: 100%;
}

[data-testid="stTabs"] button {
    color: #AAB9D2 !important;
    font-weight: 600;
    font-size: 15px;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #8DBAFF !important;
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #78AFFF;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(100deg, #3978E8, #7655DB);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 700;
    min-height: 48px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border: 1px solid #A6C8FF;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(74,115,230,0.3);
}

[data-testid="stMetricValue"] {
    color: #91BCFF;
}

[data-testid="stMetricLabel"] {
    color: #D9E3F5;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.035);
    border: 1px dashed rgba(140,175,230,0.4);
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4389F5, #A17BFA);
}

.result-card {
    background: rgba(78,130,230,0.13);
    border: 1px solid rgba(130,175,255,0.3);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    margin-bottom: 20px;
}

.result-letter {
    color: white;
    font-size: 85px;
    font-weight: 800;
    line-height: 1.3;
}

.result-confidence {
    color: #91BCFF;
    font-size: 21px;
    font-weight: 700;
}

.result-label {
    color: #AABBD7;
    font-size: 12px;
    letter-spacing: 1px;
}

.section-caption {
    color: #A9B7D0;
    font-size: 13px;
}

.footer {
    color: #8291AC;
    font-size: 12px;
    text-align: center;
    padding-top: 35px;
    letter-spacing: 1px;
}

@media (max-width: 768px) {
    .block-container,
    [data-testid="stMainBlockContainer"] {
        width: 98% !important;
        padding: 1rem 0.5rem 2rem 0.5rem !important;
    }

    .hero {
        padding: 20px 5px;
    }

    .hero h1 {
        font-size: 38px;
    }

    .result-letter {
        font-size: 65px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================
# 2. LOAD TRAINED MODEL
# =========================

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


# =========================
# 3. IMAGE PREPROCESSING
# =========================

def preprocess_image(image):

    image = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    points = cv2.findNonZero(threshold)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)

    letter = threshold[y:y+h, x:x+w]

    letter = cv2.copyMakeBorder(
        letter,
        5, 5, 5, 5,
        cv2.BORDER_CONSTANT,
        value=0
    )

    h, w = letter.shape

    scale = 20 / max(h, w)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    letter = cv2.resize(
        letter,
        (new_w, new_h)
    )

    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = letter

    return canvas.astype("float32") / 255.0


# =========================
# 4. PREDICTION FUNCTION
# =========================

def show_prediction(processed, model):

    prediction = model.predict(
        processed.reshape(1, 28, 28, 1),
        verbose=0
    )[0]

    top_indices = np.argsort(prediction)[::-1][:3]

    predicted_letter = chr(top_indices[0] + 65)

    confidence = prediction[top_indices[0]] * 100

    st.markdown("### ✨ Recognition Result")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">
                PREDICTED LETTER
            </div>
            <div class="result-letter">
                {predicted_letter}
            </div>
            <div class="result-confidence">
                {confidence:.2f}%
            </div>
            <div class="result-label">
                MODEL CONFIDENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("**Top 3 predictions**")

    for index in top_indices:

        letter = chr(index + 65)

        score = float(prediction[index])

        st.write(
            f"**{letter} — {score * 100:.2f}%**"
        )

        st.progress(score)

    st.markdown("**Processed model input**")

    st.image(
        processed,
        caption="28 × 28 grayscale image",
        width=150
    )


# =========================
# 5. HERO SECTION
# =========================

st.markdown("""
<div class="hero">

    <span class="hero-tag">
        ✦ MACHINE LEARNING PROJECT
    </span>

    <h1>
        Your handwriting,<br>
        <span class="gradient-text">
            decoded by AI.
        </span>
    </h1>

    <p>
        An intelligent handwritten alphabet recognition system.
        Draw a letter or upload an image and let the CNN model
        recognize it.
    </p>

    <span class="status">
        ● MODEL READY
    </span>

</div>
""", unsafe_allow_html=True)


# =========================
# 6. MODEL INITIALIZATION
# =========================

try:

    model = load_model()

except Exception as error:

    st.error(f"Model loading failed: {error}")

    st.stop()


# =========================
# 7. MAIN INTERFACE
# =========================

st.markdown("## Explore the Recognizer")

tab1, tab2 = st.tabs([
    "✍️ Draw a Letter",
    "📁 Upload an Image"
])


# =========================
# 8. DRAWING INTERFACE
# =========================

with tab1:

    st.markdown(
        '<p class="section-caption">Draw one capital letter using your mouse or touchscreen.</p>',
        unsafe_allow_html=True
    )

    input_col, result_col = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    with input_col:

        with st.container(border=True):

            st.subheader("✏️ Drawing Canvas")

            drawing = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=14,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=320,
                width=400,
                drawing_mode="freedraw",
                return_image_data=True,
                key="letter_canvas"
            )

            predict_draw = st.button(
                "✨ Recognize My Letter",
                key="draw_predict"
            )

    with result_col:

        with st.container(border=True):

            st.subheader("🔮 Prediction Panel")

            if predict_draw:

                if drawing.image_data is None:

                    st.warning(
                        "Please draw a letter first."
                    )

                else:

                    rgba = drawing.image_data.astype("uint8")

                    image = Image.fromarray(
                        rgba,
                        "RGBA"
                    ).convert("RGB")

                    processed = preprocess_image(image)

                    if processed is None:

                        st.warning(
                            "No letter detected. Please try again."
                        )

                    else:

                        show_prediction(
                            processed,
                            model
                        )

            else:

                st.markdown(
                    """
                    <div style="
                        text-align:center;
                        padding:65px 15px;
                        color:#9BAECC;
                    ">
                        <div style="font-size:50px;">✍️</div>
                        <h3 style="color:white;">
                            Ready to recognize
                        </h3>
                        <p>
                            Draw a letter on the canvas
                            and click the prediction button.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =========================
# 9. IMAGE UPLOAD INTERFACE
# =========================

with tab2:

    st.markdown(
        '<p class="section-caption">Upload an image containing one handwritten capital letter.</p>',
        unsafe_allow_html=True
    )

    upload_col, upload_result_col = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    with upload_col:

        with st.container(border=True):

            st.subheader("📤 Upload Your Image")

            uploaded = st.file_uploader(
                "Choose a handwritten image",
                type=["png", "jpg", "jpeg"],
                key="upload_letter"
            )

            if uploaded is not None:

                image = Image.open(
                    uploaded
                ).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded image",
                    width=300
                )

                predict_upload = st.button(
                    "✨ Recognize Uploaded Letter",
                    key="upload_predict"
                )

            else:

                predict_upload = False

                st.info(
                    "Upload a PNG, JPG or JPEG image to continue."
                )

    with upload_result_col:

        with st.container(border=True):

            st.subheader("🔮 Prediction Panel")

            if uploaded is not None and predict_upload:

                processed = preprocess_image(image)

                if processed is None:

                    st.warning(
                        "No letter detected. Please upload another image."
                    )

                else:

                    show_prediction(
                        processed,
                        model
                    )

            else:

                st.markdown(
                    """
                    <div style="
                        text-align:center;
                        padding:65px 15px;
                        color:#9BAECC;
                    ">
                        <div style="font-size:50px;">🧠</div>
                        <h3 style="color:white;">
                            Awaiting your image
                        </h3>
                        <p>
                            Your prediction and confidence
                            results will appear here.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =========================
# 10. FOOTER
# =========================

st.markdown("""
<div class="footer">

    LETTERAI • POWERED BY CNN, TENSORFLOW & EMNIST

    <br><br>

    Built as a Machine Learning portfolio project.

</div>
""", unsafe_allow_html=True)

