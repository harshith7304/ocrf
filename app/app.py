import streamlit as st
import joblib
from pathlib import Path
from src.utils.text_cleaning import normalize_text

BASE = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Fake Job Detector", page_icon="🔎", layout="centered")

st.title("🔎 Fake Job Posting Detector")

@st.cache_resource
def load_model():
    return joblib.load(BASE / "models" / "baseline_tfidf_logreg.joblib")

pipe = load_model()

text = st.text_area("Paste a job posting (title + description)", height=220)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please paste some text.")
    else:
        norm = normalize_text(text)
        pred = pipe.predict([norm])[0]
        proba = pipe.predict_proba([norm])[0][pred]
        label = "FAKE" if pred == 1 else "LEGIT"
        st.subheader(f"Prediction: {label}")
        st.progress(float(proba))
        st.caption(f"Confidence: {proba:.3f}")
