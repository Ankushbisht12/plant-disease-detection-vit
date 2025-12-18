import streamlit as st

st.set_page_config(page_title="Prediction History", page_icon="📜")

st.title("📜 Prediction History")

# Initialize history storage
if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.history:
    st.info("No predictions made yet.")
else:
    for idx, item in enumerate(reversed(st.session_state.history), start=1):
        with st.expander(f"Prediction #{idx}"):
            st.write(f"🌱 **Crop:** {item['crop']}")
            st.write(f"🦠 **Disease:** {item['disease']}")
            st.write(f"📊 **Confidence:** {item['confidence'] * 100:.2f}%")
            st.write(f"💡 **Recommendation:** {item['recommendation']}")
