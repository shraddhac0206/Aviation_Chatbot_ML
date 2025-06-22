import streamlit as st
from streamlit_chat import message
import os
import json
import pandas as pd
from streamlit_chat import message
import numpy as np
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns
from llm_utils import chat_api, chat_with_data_api




MAX_LENGTH_MODEL_DICT = {
    "gpt-4": 8191,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384
}


def get_text():
    """Input text by the user"""
    # input_text = st.text_input(
    #     label="Ask me your question.",
    #     value="",
    #     key="input"
    # )
    # return input_text
    return st.chat_input("Type your question here...")


def sidebar():
    st.sidebar.title("✈️ Flight AI Assistant")
    st.sidebar.markdown("### Customize your chatbot experience")

    model = st.sidebar.selectbox(
        "🤖 *Choose AI Model*",
        ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"],
        help="Select an AI model for generating responses."
    )

    temperature = st.sidebar.slider(
        "🔥 *Creativity (Temperature*)",
        0.0, 2.0, 0.7, 0.01,
        help="Higher values make responses more creative."
    )

    max_tokens = st.sidebar.slider(
        "📝 *Max Tokens*",
        0, 4096, 256, 1,
        help="Defines the maximum response length."
    )

    top_p = st.sidebar.slider(
        "🎯 *Response Precision*",
        0.0, 1.0, 0.5, 0.01,
        help="Lower values make responses more deterministic."
    )

    st.sidebar.markdown("---")

    return {  
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

def eda(df):
    """Basic EDA on uploaded data"""
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write(df.head())

    if isinstance(df, pd.DataFrame):
        st.write("Dataset Summary:")
        st.write(df.describe())

def load_data(file_path):
    """Load data from CSV, PDF, or TXT files"""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as file:
            return pd.DataFrame({"text": file.readlines()})
    elif ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return pd.DataFrame({"text": [text]})
    else:
        raise ValueError("Unsupported file format")


def chatbot():
    """
    Unified chatbot with file handling, text-based chat, and forecasting
    """

    st.title("Aviation AI Chatbot ✈️")
    st.markdown("<h1 style='text-align: center; color: lightblue;'>Aviation AI Chatbot</h1>", unsafe_allow_html=True)
    st.subheader(
        """
        Welcome to the *Aviation AI Chatbot! This assistant helps you **analyze airline trends, predict flights, and visualize aviation data*.  
        *Features:*
        - 📊 *Upload & Explore Flight Datasets*
        - 🛫 *Generate Charts & Predictions*
        - 💬 *Chat with Aviation AI*
        - 🔍 *Get Flight Insights Instantly!*
        """,
        unsafe_allow_html=True,
    )

    if "model_params" in st.session_state:
       del st.session_state["model_params"]

    with st.sidebar:
        model_params = sidebar()

    # # Upload file (CSV, PDF, or TXT)
    # uploaded_file = st.file_uploader("Upload CSV, PDF, or Text file", type=["csv", "pdf", "txt"])
    # df = None  # Initialize df as None
    # if uploaded_file:
    #     # Save uploaded file
    #     file_path = os.path.join("uploads", uploaded_file.name)
    #     with open(file_path, "wb") as f:
    #         f.write(uploaded_file.getbuffer())

    #     # Load the data
    #     try:
    #         df = load_data(file_path)
    #         st.write(f"Uploaded file: {uploaded_file.name}")
    #         eda(df)
    #     except Exception as e:
    #         st.error(f"Error loading file: {e}")

    # Initialize session state
    if "messages" not in st.session_state or st.session_state.get("reset_chat", False):
        st.session_state["messages"] = [{"role": "system", "content": "You're an AI assistant."}]
        st.session_state["reset_chat"] = False  # Reset the flag

    # greeting_bot_msg = (
    #     "Hi, I am not a just a chatbot. I can plot fetched data for you. "
    #     "Ask me questions like 'What was US, UK and Germany's GDP in 2019 and 2020?'. "
    #     "Once the data is received, ask me to plot it."
    # )

    # # Storing the chat
    # if "generated" not in st.session_state:
    #     st.session_state["generated"] = [greeting_bot_msg]

    if "past" not in st.session_state:
        st.session_state["past"] = []

    prompt = "You are a chatbot that answers questions. You can also plot data if asked"
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": prompt}]

    user_input = get_text()

    # if ((len(st.session_state["past"]) > 0)
    #         and (user_input == st.session_state["past"][-1])):
    #     user_input = ""

    # if user_input:
    #     st.session_state["messages"].append(
    #         {"role": "user", "content": user_input}
    #     )
    #     response = chat_api(st.session_state["messages"], **model_params)
    #     st.session_state.past.append(user_input)
    #     if response is not None:
    #         st.session_state.generated.append(response)
    #         st.session_state["messages"].append({
    #             "role": "assistant",
    #             "content": response
    #         })
    if user_input:
     try:
        formatted_input = json.dumps({"query": user_input})  # Ensure valid JSON format
        st.session_state["messages"].append(
            {"role": "user", "content": formatted_input}
        )
        response = chat_api(st.session_state["messages"], **model_params)
     except json.JSONDecodeError as e:
        st.error(f"JSON formatting error: {e}")
        response = None

     st.session_state.past.append(user_input)
    
     if response is not None:
        st.session_state.generated.append(response)
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response
        })

    if st.session_state["generated"]:
      for i in range(len(st.session_state["generated"])):
        if i < len(st.session_state["past"]):
            message(
                st.session_state["past"][i],
                is_user=True,
                key=str(i) + "_user"
            )
        message(st.session_state["generated"][i], key=str(i))



if __name__ == "__main__":
    chatbot()
