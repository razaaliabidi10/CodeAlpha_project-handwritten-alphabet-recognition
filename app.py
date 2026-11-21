 
import streamlit as st
import numpy as np
import cv2

from PIL import Image
from pathlib import Path
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="LetterAI | Handwriting Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = Path("model/handwritten_alphabet_model.keras")


# ==========================================
# 2. PREMIUM UI DESIGN
# ==========================================

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

.block-container {
    max-width: 1500px !important;
    width: 94% !important;
    padding: 2rem 2rem 3rem 2rem !important;
    margin: auto !important;
}

h1, h2, h3, p, label {
    font-family: 'Inter', sans-serif !important;
}

h1 {
    font-size: clamp(38px, 5vw, 65px) !important;
    font-weight: 800 !important;
    letter-spacing: -2px !important;
    color: #FFFFFF !important;
}

h2 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

h3 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

.hero-subtitle {
    color: #A9B7D0;
    font-size: 17px;
    line-height: 1.7;
    max-width: 750px;
}

.hero-tag {
    color: #A9C8FF;
    background: rgba(65,105,190,0.13);
    border: 1px solid rgba(125,170,255,0.35);
    border-radius: 30px;
    padding: 8px 15px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.3px;
}

.status {
    color: #70E4B0;
    background: rgba(43,180,120,0.10);
    border: 1px solid rgba(76,220,155,0.3);
    border-radius: 30px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 700;
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

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.035);
    border: 1px dashed rgba(140,175,230,0.4);
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4389F5, #A17BFA);
}

.section-caption {
    color: #A9B7D0;
    font-size: 14px;
}

.footer {
    color: #8291AC;
    font-size: 12px;
    text-align: center;
    padding-top: 35px;
    letter-spacing: 1px;
}

@media (max-width: 768px) {
    .block-container {
        width: 98% !important;
        padding: 1rem 0.5rem 2rem 0.5rem !important;
    }

    h1 {
        font-size: 38px !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

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


# ==========================================
# 5. PREDICTION FUNCTION
# ==========================================

def show_prediction(processed, model):

    prediction = model.predict(
        processed.reshape(1, 28, 28, 1),
        verbose=0
    )[0]

    top_indices = np.argsort(prediction)[::-1][:3]

    predicted_letter = chr(top_indices[0] + 65)

    confidence = prediction[top_indices[0]] * 100

    st.markdown("### ✨ Recognition Result")

    st.metric(
        label="Predicted Letter",
        value=predicted_letter,
        delta=f"{confidence:.2f}% model confidence"
    )

    st.markdown("#### Top 3 Predictions")

    for index in top_indices:

        letter = chr(index + 65)

        score = float(prediction[index])

        st.write(f"**{letter} — {score * 100:.2f}%**")

        st.progress(score)

    st.markdown("#### Processed Model Input")

    st.image(
        processed,
        caption="28 × 28 grayscale image",
        width=150
    )


# ==========================================
# 6. HERO SECTION - NATIVE STREAMLIT
# ==========================================

st.markdown(
    '<div class="hero-tag">✦ MACHINE LEARNING PROJECT</div>',
    unsafe_allow_html=True
)

st.title("Your handwriting, decoded by AI.")

st.markdown(
    """
    <p class="hero-subtitle">
    An intelligent handwritten alphabet recognition system.
    Draw a letter or upload an image and let the CNN model
    recognize it.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="status">● MODEL READY</div>',
    unsafe_allow_html=True
)

st.write("")


# ==========================================
# 7. MODEL INITIALIZATION
# ==========================================

try:

    model = load_model()

except Exception as error:

    st.error(f"Model loading failed: {error}")

    st.stop()


# ==========================================
# 8. MAIN INTERFACE
# ==========================================

st.markdown("## Explore the Recognizer")

tab1, tab2 = st.tabs([
    "✍️ Draw a Letter",
    "📁 Upload an Image"
])


# ==========================================
# 9. DRAWING INTERFACE
# ==========================================

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

                    st.warning("Please draw a letter first.")

                else:

                    rgba = drawing.image_data.astype("uint8")

                    image = Image.fromarray(
                        rgba,
                        "RGBA"
                    ).convert("RGB")

                    processed = preprocess_image(image)

                    if processed is None:

                        st.warning("No letter detected. Please try again.")

                    else:

                        show_prediction(processed, model)

            else:

                st.info(
                    "✍️ Draw a letter on the canvas and click "
                    "'Recognize My Letter' to see the prediction."
                )


# ==========================================
# 10. IMAGE UPLOAD INTERFACE
# ==========================================

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

                st.info(
                    "🧠 Your prediction and confidence results "
                    "will appear here after uploading an image."
                )


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.caption(
    "LETTERAI • POWERED BY CNN, TENSORFLOW & EMNIST"
)

st.caption(
    "Built as a Machine Learning portfolio project."
)
