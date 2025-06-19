import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np

# Load model + tokenizer
model_path = "models/roberta-bias"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# Set class labels
labels = ['center', 'left', 'right']

# Sidebar
with st.sidebar:
    st.title("📰 Media Bias Detector")
    st.markdown("Detect the political bias in news headlines using a fine-tuned **RoBERTa** model.")
    st.markdown("Trained on real-world news data from multiple sources.")
    st.markdown("Bias labels: **Left**, **Center**, **Right**.")
    st.markdown("Model: `RoBERTa-base` fine-tuned with HuggingFace.")
    st.markdown("Built by **Chahat Verma**")
    st.markdown("---")

# Custom button style
st.markdown("""
    <style>
    .stButton>button {
        background-color: #f63366;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Main App
st.title("🧠 Real-Time Media Bias Classifier")
st.markdown("Enter a news headline and find out if it leans **Left**, **Right**, or is **Neutral**.")

st.markdown("---")
headline = st.text_input("✍️ Enter News Headline:")

if st.button("Detect Bias"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            st.success(f"🎯 **Predicted Bias: `{labels[pred].upper()}`**")
            st.progress(probs[0][pred].item())