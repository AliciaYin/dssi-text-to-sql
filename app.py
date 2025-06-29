import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    base_model_id = "Salesforce/codet5p-220m"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_model, "lora_codet5_adapter")
    model.to("cpu")  # Use "cuda" if GPU available
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("🧠 Text-to-SQL Assistant (LoRA Fine-Tuned)")

user_input = st.text_area("Enter your natural language query:")

if st.button("Generate SQL"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating SQL..."):
            prompt = f"Translate to SQL: {user_input.strip()}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cpu")
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
            sql = sql.split("Translate")[0].strip()  # Clean hallucinations
            st.success("Generated SQL:")
            st.code(sql, language="sql")
