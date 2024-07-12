# NameGeneratorApp

A Streamlit-based Name Generator application that leverages advanced language model capabilities through the Hugging Face Hub. The application allows users to generate a list of names based on specified parameters such as the number of names, origin, gender, and optional starting alphabet. Key features and technologies used in this project include:

Streamlit: Built an interactive web interface for user inputs and displaying results.

Environment Management: Utilized python-dotenv to securely load API tokens from a .env file.

Hugging Face Inference API: Integrated with the Hugging Face Inference Client to interact with the Mistral-7B-Instruct-v0.3 model.

LangChain: Employed LangChain for creating and managing prompt templates.

JSON Processing: Handled API responses and parsed JSON data for extracting and displaying generated names.

Error Handling: Implemented robust error handling mechanisms to ensure smooth user experience.
