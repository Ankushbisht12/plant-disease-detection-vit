import streamlit as st
import requests
from PIL import Image
import tempfile

BACKEND_URL = "http://127.0.0.1:8000/predict"

st.title("📷 Upload Leaf Image")
st.markdown("Upload a clear image of a plant leaf to detect disease.")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect Disease"):
        with st.spinner("Analyzing image..."):
            try:
                # Save image temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    file_path = tmp.name

                with open(file_path, "rb") as f:
                    files = {"file": f}
                    response = requests.post(BACKEND_URL, files=files)

                if response.status_code == 200:
                    data = response.json()

                    st.success("✅ Prediction Successful")

                    st.subheader("🌿 Prediction Result")
                    st.write(f"**Crop:** {data['crop']}")
                    st.write(f"**Disease:** {data['disease']}")
                    st.write(f"**Confidence:** {data['confidence'] * 100:.2f}%")

                    st.subheader("🧪 Recommended Action")
                    st.info(data["recommendation"])

                    # ✅ Save to history ONLY after success
                    st.session_state.history.append({
                        "crop": data["crop"],
                        "disease": data["disease"],
                        "confidence": data["confidence"],
                        "recommendation": data["recommendation"]
                    })

                else:
                    st.error(f"❌ Backend error: {response.text}")

            except Exception as e:
                st.error(f"⚠️ Failed to connect to backend: {e}")
