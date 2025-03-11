import os
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Fetch Mistral API Key
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    st.error("MISTRAL_API_KEY is missing! Please check your .env file.")
    st.stop()

# Debugging: Show first few characters of API key
st.write(f"Loaded API Key: {mistral_api_key[:5]}********")

# Initialize LLM Model
llm = ChatMistralAI(model="mistral-tiny", mistral_api_key=mistral_api_key)

# Define Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond concisely."),
    ("user", "Question: {question}")
])

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit UI
st.title("🤖 Langchain + Mistral AI Chatbot")
input_text = st.text_input("Enter your question:")

if st.button("Submit"):
    if input_text.strip():
        with st.spinner("Generating response..."):
            try:
                response = chain.invoke({'question': input_text})
                st.success("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid question.")
