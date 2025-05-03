import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Title
st.title("📝Saaransh")

# Model selection
model_choice = st.selectbox("Choose a model", ["mT5 (XLSum)", "IndicBART"])

# Correct checkpoint mapping
model_checkpoints = {
    "mT5 (XLSum)": "D:\Saaranshv2\pythonProject\mT5_multilingual_XLSum",
    "IndicBART": "D:\Saaranshv2\pythonProject\IndicBART"
}

checkpoint = model_checkpoints[model_choice]

# Load tokenizer and model
@st.cache_resource
def load_model_and_tokenizer(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer(checkpoint)

# User input
text = st.text_area("Enter text in Hindi", height=250)

# Generate summary on button click
if st.button("Generate Summary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):
            inputs = tokenizer(
                [text],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            )

            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                max_length=70,
                min_length=40,
                no_repeat_ngram_size=2,
                num_beams=4
            )[0]

            summary = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

        st.subheader("📝 Summary:")
        st.success(summary)
