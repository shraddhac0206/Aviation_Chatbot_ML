import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_chat import message
import seaborn as sns
import pandas as pd
from main import get_text, sidebar, eda, load_data, chatbot
from llm_utils import chat_with_data_api, chat_api, extract_python_code
from forecasting import forecasting, time_series_forecasting
import uuid
import plotly.express as px

CHAT_HISTORY_FILE = "chat_history.json"

def chat_with_data():

    # Initialize session state keys if they don't exist
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": "You're an AI assistant."}]
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "reset_chat" not in st.session_state:
        st.session_state["reset_chat"] = False  # Reset flag
    if "plot_generated" not in st.session_state:
        st.session_state["plot_generated"] = False


    st.title("Aviation AI Chatbot")
    st.markdown(
        """
        Welcome to the **Aviation AI Chatbot!** This assistant helps you analyze airline trends, predict flights, and visualize aviation data.  
        *Features:*
        - 📊 *Upload & Explore Flight Datasets*
        - 🛫 *Generate Charts & Predictions*
        - 💬 *Chat with Aviation AI*
        - 🔍 *Get Flight Insights Instantly!*
        """,
        unsafe_allow_html=True
    )


    with st.sidebar:
        model_params = sidebar()
        memory_window = st.slider(
            label="Memory Window",
            value=3,
            min_value=1,
            max_value=10,
            step=1,
            help=(
                """The size of history chats that is kept for context. A value of, say,
                3, keeps the last three pairs of promtps and reponses, i.e. the last
                6 messages in the history."""
            )
        )

    uploaded_file = st.file_uploader("Upload CSV, PDF, or Text file", type=["csv", "pdf", "txt"])
    df = None  # Initialize df as None
    if uploaded_file:
        # Save uploaded file
        upload_folder = "uploads"

        # Ensure the 'uploads' folder exists
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            # Load the data
        try:
            df = load_data(file_path)
            st.write(f"Uploaded file: {uploaded_file.name}")
            eda(df)
        except Exception as e:
            st.error(f"Error loading file: {e}")
        if "messages" not in st.session_state or st.session_state.get("reset_chat", False):
            st.session_state["messages"] = [{"role": "system", "content": "You're an AI assistant."}]
            st.session_state["reset_chat"] = False  # Reset the flag
    else:
        df = pd.DataFrame([])

    # Storing the chat
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Please upload your data"]

    if "past" not in st.session_state:
        st.session_state["past"] = []

    user_input=get_text()
    plot_keywords = ["plot", "graph", "chart", "visualize"]
    if user_input and not any(keyword in user_input.lower() for keyword in plot_keywords):
           st.session_state["plot_generated"] = False  # Reset when asking non-plot questions


    if ((len(st.session_state["past"]) > 0)
            and (user_input == st.session_state["past"][-1])):
        user_input = ""

    if ("messages" in st.session_state) and \
            (len(st.session_state["messages"]) > 2 * memory_window):
        # Keep only the system prompt and the last `memory_window` prompts/answers
        st.session_state["messages"] = (
            # the first one is always the system prompt
            [st.session_state["messages"][0]]
            + st.session_state["messages"][-(2 * memory_window - 2):]
        )

    if user_input:
    # Store user input at the END of both lists
        st.session_state["past"].append(user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})

    # Check if a file is uploaded
        if df.empty:
            response = chat_api([{"role": "user", "content": user_input}], **model_params)  # Use LLM for general queries
        else:
            response = chat_with_data_api(df, **model_params)  # Use LLM with DataFrame
        
        if response:
           if response not in st.session_state["generated"]:
                st.session_state["generated"].append(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})

        # If the response contains a code block for plotting, execute it **only once**
        if "```python" in response:
            code = extract_python_code(response)
            if code:
                unique_key = str(uuid.uuid4())  # Generate a unique ID for the plot
                code = code.replace("fig.show()", f"st.plotly_chart(fig, use_container_width=True, key='{unique_key}')")

                try:
                    exec(code, {"st": st, "plt": plt, "pd": pd, "sns": sns})
                except Exception as e:
                    st.error(f"Error executing the plot: {e}")
        
        if response is not None:
            # Ensure the response is appended only once
            if response not in st.session_state["generated"]:
                st.session_state["generated"].append(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})

           
    # Trim messages to keep only the last 10 interactions
        st.session_state["messages"] = st.session_state["messages"][-10:]
        st.session_state["past"] = st.session_state["past"][-10:]
        st.session_state["generated"] = st.session_state["generated"][-10:]

        if not df.empty:  # Ensure df is not empty before performing forecasting
            if "forecast" in user_input.lower() or "predict" in user_input.lower():
                try:
                    # Run the time series forecasting function
                    forecast = time_series_forecasting(df)
                    st.write("Predictions for the next 3 months:")
                    st.write(forecast)
                except Exception as e:
                    st.error(f"Error in forecasting: {e}")


    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):  # ⬅️ Normal order loop
            if i < len(st.session_state["past"]):
                message(
                   st.session_state["past"][i],
                   is_user=True,
                   key=str(i) + "_user"
                )
        
            response_content = st.session_state["generated"][i]
            message(response_content, key=str(i))

            # If the response contains a code block for plotting, extract and execute it
            if "```python" in response_content:
                code = extract_python_code(response_content)
                if code:
                    unique_key = str(uuid.uuid4())  # Generate a unique ID for each plot
                    code = code.replace("fig.show()", f"st.plotly_chart(fig, use_container_width=True, key='{unique_key}')")

                    try:
                       exec(code, {"st": st, "plt": plt, "pd": pd, "sns": sns, "px":px})
                       st.session_state["plot_generated"] = True
                    except Exception as e:
                        st.error(f"Error executing the plot: {e}")


if __name__ == "__main__":
    chat_with_data()
