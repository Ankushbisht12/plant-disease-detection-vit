import streamlit as st

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Plant Disease Detection & Recommendation System")

st.markdown("""
Welcome to the **Plant Disease Detection System**.

### Supported Crops
- 🍎 Apple
- 🍅 Tomato
- 🥔 Potato

Use the sidebar to:
- Upload a leaf image
- View prediction history
- Learn about the project
""")

st.info("👉 Select a page from the left sidebar to continue.")
