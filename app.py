import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import time

# =========================
# MODEL (load once)
# =========================
model = load_model("recycle_model_clean.keras")
class_names = ['glass', 'metal', 'paper', 'plastic', 'shoes']

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="♻️ Recycle AI", layout="centered")

st.title("♻️ Smart Recycle AI")
st.caption("Upload images or use live camera for real-time classification")

# =========================
# PREDICT FUNCTION
# =========================
def predict(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    label = class_names[np.argmax(pred)]
    conf = float(np.max(pred))

    return label, conf


# =========================
# SIDEBAR
# =========================
mode = st.sidebar.radio("Choose Mode", ["📷 Upload Images", "🎥 Live Camera"])


# =========================================================
# 📷 UPLOAD MODE
# =========================================================
if mode == "📷 Upload Images":

    files = st.file_uploader(
        "Upload up to 3 images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if files:

        st.subheader("Results")

        for file in files[:3]:

            image = Image.open(file)
            img = np.array(image)

            label, conf = predict(img)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, use_container_width=True)

            with col2:
                st.markdown(f"### ♻️ {label}")
                st.progress(conf)
                st.write(f"Confidence: {conf:.2f}")

                if label == "plastic":
                    st.warning("⚠️ Plastic detected")
                elif label == "paper":
                    st.info("📄 Paper recyclable")
                elif label == "glass":
                    st.success("🍾 Glass recyclable")
                elif label == "metal":
                    st.success("🔩 Metal detected")
                else:
                    st.write("👟 Other item")


# =========================================================
# 🎥 LIVE CAMERA MODE (REAL TIME)
# =========================================================
else:

    st.subheader("Live Camera")

    start = st.button("▶ Start Camera")

    frame_placeholder = st.image([])

    camera = cv2.VideoCapture(0)

    if start:

        st.info("Camera running...")

        frame_count = 0
        last_label = ""
        last_conf = 0.0

        while True:

            ret, frame = camera.read()
            if not ret:
                st.error("Camera error")
                break

            frame_count += 1

            # predict every 3 frames (faster performance)
            if frame_count % 3 == 0:
                last_label, last_conf = predict(frame)

            # overlay text
            cv2.putText(frame,
                        f"{last_label} ({last_conf:.2f})",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            # convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # update UI
            frame_placeholder.image(frame)

            time.sleep(0.03)