import streamlit as st
import requests
import io
import numpy as np
import cv2
import os
from PIL import Image

st.set_page_config(
    page_title="MNIST Digit Predictor", page_icon="numbers.png", layout="wide"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")


def preprocess_drawing(image):
    """Preprocess the drawn image for prediction"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (28, 28))

    image = 255 - image

    image = image.astype(np.uint8)

    return image


def predict_digit(image):
    """Send image to API for prediction"""
    try:
        img_buffer = io.BytesIO()
        pil_image = Image.fromarray(image)
        pil_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        files = {"file": ("digit.png", img_buffer, "image/png")}
        response = requests.post(f"{API_URL}/predict", files=files)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}

    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}


def main():
    st.title("MNIST Digit Predictor")
    st.markdown("Draw a digit (0-9) in the canvas below and get an AI prediction!")

    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("API is running and ready!")
        else:
            st.error("API is not responding properly")
            return
    except:
        st.error("Cannot connect to API. Make sure the API service is running.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎨 Drawing Canvas")

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )

        # Buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)

        with col_btn1:
            if st.button("🔍 Predict", type="primary"):
                if canvas_result.image_data is not None:
                    image = canvas_result.image_data

                    processed_image = preprocess_drawing(image)

                    with st.spinner("Predicting..."):
                        result = predict_digit(processed_image)

                    st.session_state.prediction_result = result
                else:
                    st.warning("Please draw a digit first!")

        with col_btn2:
            if st.button("🗑️ Clear"):
                st.session_state.prediction_result = None
                st.rerun()

        with col_btn3:
            if st.button("Show Processed"):
                if canvas_result.image_data is not None:
                    image = canvas_result.image_data
                    processed_image = preprocess_drawing(image)
                    st.image(
                        processed_image, caption="Processed Image (28x28)", width=100
                    )

    with col2:
        st.subheader("Prediction Results")

        if (
            hasattr(st.session_state, "prediction_result")
            and st.session_state.prediction_result
        ):
            result = st.session_state.prediction_result

            if "error" in result:
                st.error(f"{result['error']}")
            else:
                predicted_digit = result["predicted_digit"]
                confidence = result["confidence"]

                st.markdown(f"### Predicted Digit: **{predicted_digit}**")
                st.markdown(f"**Confidence:** {confidence:.2%}")

                st.progress(confidence)

                st.markdown("#### Top 3 Predictions:")
                for i, pred in enumerate(result["top3_predictions"]):
                    digit = pred["digit"]
                    conf = pred["confidence"]

                    # Color coding
                    if i == 0:
                        st.markdown(f"🥇 **{digit}**: {conf:.2%}")
                    elif i == 1:
                        st.markdown(f"🥈 **{digit}**: {conf:.2%}")
                    else:
                        st.markdown(f"🥉 **{digit}**: {conf:.2%}")

                    # Progress bar for each prediction
                    st.progress(conf)
        else:
            st.info("👆 Draw a digit and click 'Predict' to see results!")

    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. Draw a digit (0-9) in the canvas using your mouse or touch")
    st.markdown("2. Click 'Predict' to get the AI prediction")
    st.markdown("3. Use 'Clear' to start over")
    st.markdown("4. Use 'Show Processed' to see how your drawing is preprocessed")


try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error(
        "streamlit-drawable-canvas is not installed. Please install it with: pip install streamlit-drawable-canvas"
    )
    st.stop()

if __name__ == "__main__":
    main()
