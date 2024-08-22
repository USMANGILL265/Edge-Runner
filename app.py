import os
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"
OUTPUT_FILE = "synthetic_data.txt"
PROMPT = "Generate synthetic data for detection of malware and running high memory apps on background:"
NUM_SAMPLES = 100  # Number of synthetic data samples to generate
MAX_LENGTH = 512  # Maximum length of the generated text

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

def generate_synthetic_data(tokenizer, model, prompt, num_samples, max_length):
    synthetic_data = []
    inputs = tokenizer(prompt, return_tensors="pt")
    
    for _ in range(num_samples):
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        synthetic_data.append(generated_text)
    
    return synthetic_data

def save_data_to_file(data, filename):
    with open(filename, 'w') as file:
        for line in data:
            file.write(line + '\n')

def main():
    st.title("Synthetic Data Generator")
    
    # Input fields in Streamlit
    prompt = st.text_area("Enter the prompt:", PROMPT)
    num_samples = st.number_input("Number of samples to generate:", value=NUM_SAMPLES, min_value=1)
    max_length = st.slider("Max length of generated text:", min_value=1, max_value=1024, value=MAX_LENGTH)
    
    if st.button("Generate Data"):
        st.write("Loading model and tokenizer...")
        tokenizer, model = load_model_and_tokenizer()
        
        st.write("Generating synthetic data...")
        data = generate_synthetic_data(tokenizer, model, prompt, num_samples, max_length)
        
        # Display the data
        st.subheader("Generated Data")
        for i, line in enumerate(data, 1):
            st.text(f"{i}: {line}")
        
        # Save the data and provide a download link
        save_data_to_file(data, OUTPUT_FILE)
        st.success(f"Generated data has been saved to {OUTPUT_FILE}")
        st.download_button("Download Synthetic Data", data="\n".join(data), file_name=OUTPUT_FILE)

if __name__ == "__main__":
    main()
