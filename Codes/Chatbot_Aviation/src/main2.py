import streamlit as st
import pandas as pd
from llm_utils2 import chat_api
import os

def get_text():
    """
    Function to get user input from the text area in Streamlit.
    """
    return st.text_input("Ask a question:", "")

def sidebar():
    """
    Defines and returns the sidebar parameters for model configuration.
    """
    st.sidebar.title("⚙️ Chatbot Settings")

    model = st.sidebar.selectbox("Choose AI Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    max_tokens = st.sidebar.slider("Maximum length (tokens)", min_value=0, max_value=4096, value=256, step=1)
    top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }

def load_data(file_path):
    """
    Function to load data from a CSV file.
    """
    try:
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def eda(df):
    """
    Function to perform basic exploratory data analysis.
    """
    if df is not None:
        st.subheader("Dataset Preview")
        st.write(df.head())

        st.subheader("Summary Statistics")
        st.write(df.describe())

        st.subheader("Missing Values")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])

def chatbot():
    """
    Main function to handle chatbot conversation.
    """
    st.title("Aviation AI Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": "You're an AI assistant."}]

    user_input = get_text()

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        response = chat_api([{"role": "user", "content": user_input}])
        st.session_state["messages"].append({"role": "assistant", "content": response})

    if st.session_state["messages"]:
        for message in st.session_state["messages"]:
            if message["role"] == "user":
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**Bot:** {message['content']}")

if __name__ == "__main__":
    chatbot()
