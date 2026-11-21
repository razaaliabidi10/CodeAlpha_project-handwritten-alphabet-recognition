 
import streamlit as st
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Handwritten Alphabet Recognizer",
    page_icon="✍️",
    layout="wide"
)

MODEL_PATH = Path("model/handwritten_alphabet_model.keras")


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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

    st.success(f"Predicted letter: {predicted_letter}")
    st.metric("Confidence", f"{confidence:.2f}%")

    st.write("### Top 3 predictions")

    for index in top_indices:
        letter = chr(index + 65)
        score = prediction[index] * 100

        st.write(f"**{letter}** — {score:.2f}%")
        st.progress(float(prediction[index]))

    st.write("### Processed image")
    st.image(
        processed,
        caption="Model input (28 × 28)",
        width=150
    )


st.title("✍️ Handwritten Alphabet Recognizer")

st.markdown(
    """
    Recognize handwritten English letters from **A to Z**.

    Draw a letter or upload an image to get a prediction.
    """
)

try:
    model = load_model()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


tab1, tab2 = st.tabs([
    "✏️ Draw a Letter",
    "📤 Upload an Image"
])


with tab1:

    st.subheader("Draw your letter")

    drawing = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=14,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=250,
        width=250,
        drawing_mode="freedraw",
        key="letter_canvas"
    )

    if st.button("Recognize drawn letter"):

        if not drawing.image_data_url:
            st.warning("Please draw a letter first.")

        else:
            try:
                image_bytes = base64.b64decode(
                    drawing.image_data_url.split(",")[1]
                )

                image = Image.open(
                    BytesIO(image_bytes)
                ).convert("RGB")

                processed = preprocess_image(image)

                if processed is None:
                    st.warning(
                        "No letter detected. Please draw again."
                    )
                else:
                    show_prediction(processed, model)

            except Exception as error:
                st.error(f"Prediction failed: {error}")


with tab2:

    st.subheader("Upload a handwritten letter")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded is not None:

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            width=250
        )

        if st.button("Recognize uploaded letter"):

            processed = preprocess_image(image)

            if processed is None:
                st.warning(
                    "No letter detected in this image."
                )
            else:
                show_prediction(processed, model)


st.divider()

st.caption(
    "Handwritten Alphabet Recognition | "
    "CNN + TensorFlow + EMNIST"
)
