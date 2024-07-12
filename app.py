import streamlit as st
from dotenv import load_dotenv
# from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
# from langchain_core.output_parsers import StrOutputParser
import json
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve variables from the environment
api_token = os.getenv("API_TOKEN")
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize the InferenceClient with the model and the API token
llm_client = InferenceClient(
    model=repo_id,
    token=api_token,
    timeout=120,
)

def call_llm(inference_client: InferenceClient, prompt: str, max_new_tokens: int):
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens
            }
        }
    )
    response_text = response.decode('utf-8')
    try:
        response_json = json.loads(response_text)
        return response_json
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None

# Prompt template
name_template = PromptTemplate(
    input_variables=['number', 'origin', 'gender', 'alphabet'],
    template="Suggest exactly {number} number of {gender} names that are of {origin} along woth their meanings, and starting with the letter {alphabet}."
)

# Streamlit app
st.title("Name Generator")

# Input from the user
x = st.number_input("Number of Names:", min_value=1, step=1)
y = st.text_input("Origin:")
z = st.selectbox("Gender:", ["Male", "Female", "Unisex"])

# Optional starting alphabet selection
alphabet_option = st.checkbox("Starting with the alphabet...")
if alphabet_option:
    alphabet = st.selectbox("Choose an alphabet:", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
else:
    alphabet = ""

if st.button("Generate Names"):
    x_str = str(x)
    prompt = name_template.format(number=x_str, origin=y, gender=z, alphabet=alphabet)

    # Get the response from the API
    response = call_llm(llm_client, prompt, max_new_tokens=500)  # Adjust max_new_tokens as needed

    if response:
        # Extract the generated text
        if isinstance(response, list) and "generated_text" in response[0]:
            generated_text = response[0]["generated_text"]
            # Remove the prompt part from the generated text
            if generated_text.startswith(prompt):
                generated_names = generated_text[len(prompt):].strip()
            else:
                generated_names = generated_text
        else:
            st.error("Unexpected response format.")
            st.write(response)  # For debugging: show the full response

        # Split the response into lines and process
        lines = generated_names.split('\n')

        # Display the output
        st.write("Generated Names:")
        for line in lines:
            if line.strip():  # Only process non-empty lines
                parts = line.split(':', 1)
                if len(parts) > 1:
                    name = parts[0].strip()
                    explanation = parts[1].strip()
                    st.markdown(f"**{name}**: {explanation}")
                else:
                    name = parts[0].strip()
                    st.markdown(f"**{name}**")

    else:
        st.error("Failed to get a valid response from the API.")
