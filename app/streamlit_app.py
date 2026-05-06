from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from tokenizer import load_labels, load_tokenizer, tokenize_text
from triton_client import infer_logits, predict_from_logits


TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "emotion_phobert")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return load_tokenizer()


@st.cache_data(show_spinner=False)
def get_labels():
    return load_labels()


def predict(text: str):
    tokenizer = get_tokenizer()
    labels = get_labels()
    encoded = tokenize_text(tokenizer, text, max_length=MAX_LENGTH)
    logits = infer_logits(
        encoded["input_ids"],
        encoded["attention_mask"],
        triton_url=TRITON_URL,
        model_name=MODEL_NAME,
    )
    return predict_from_logits(logits, labels)


st.set_page_config(page_title="Vietnamese Emotion Demo", layout="centered")

st.title("Vietnamese Emotion Classification")
st.caption("Streamlit GUI -> PhoBERT tokenizer -> Triton ONNX model")

text = st.text_area(
    "Nhập câu tiếng Việt",
    value="Tôi rất vui vì hôm nay mọi thứ diễn ra tốt đẹp.",
    height=140,
)

if st.button("Predict", type="primary"):
    cleaned_text = text.strip()
    if not cleaned_text:
        st.warning("Vui lòng nhập một câu tiếng Việt trước khi predict.")
    else:
        try:
            with st.spinner("Đang chạy model trên Triton..."):
                start_time = time.perf_counter()
                prediction = predict(cleaned_text)
                latency_ms = (time.perf_counter() - start_time) * 1000

            st.success(f"Kết quả: {prediction.label}")
            st.write(f"Độ tin cậy: {prediction.score:.2%}")
            st.write(f"Latency: {latency_ms:.2f} ms")

            with st.expander("Xem điểm của tất cả nhãn"):
                for label, score in sorted(
                    prediction.scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                ):
                    st.write(f"{label}: {score:.2%}")
        except Exception as exc:
            st.error("Không thể predict. Hãy kiểm tra Triton Server đã chạy và model đã ready.")
            st.exception(exc)
