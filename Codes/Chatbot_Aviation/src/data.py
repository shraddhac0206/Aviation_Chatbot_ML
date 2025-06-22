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
from forecasting import forecasting
from PIL import Image
import base64
import time
import requests

CHAT_HISTORY_FILE = "chat_history.json"

# Set page configuration
st.set_page_config(page_title="Aviation AI Chatbot", layout="wide")

import time
import plotly.graph_objects as go
import pandas as pd

# # Function to get flight positions
# def get_flight_data():
#     url = "https://opensky-network.org/api/states/all"
#     response = requests.get(url).json()
#     flights = response["states"]
#     df = pd.DataFrame(flights, columns=[
#         "icao24", "callsign", "origin_country", "time_position", "last_contact",
#         "longitude", "latitude", "altitude", "on_ground", "velocity",
#         "heading", "vertical_rate", "sensors", "baro_altitude",
#         "squawk", "spi", "position_source"
#     ])
#     return df.dropna(subset=["longitude", "latitude"])  # Remove missing coordinates

# # Function to plot on a 3D Globe
# def plot_3d_globe():
#     st.subheader("🌍 **3D Global Flight Tracker**")
    
#     df = get_flight_data()

#     fig = go.Figure()

#     # Add world map sphere
#     fig.add_trace(
#         go.Scattergeo(
#             lon=df["longitude"],
#             lat=df["latitude"],
#             mode="markers",
#             marker=dict(size=5, color="red", opacity=0.6),
#             text=df["callsign"],
#         )
#     )

#     fig.update_layout(
#         geo=dict(
#             projection_type="orthographic",  # 3D sphere projection
#             showland=True,
#             showocean=True,
#             landcolor="rgb(200, 200, 200)",
#             oceancolor="rgb(0, 102, 204)",
#         ),
#         margin=dict(l=0, r=0, t=0, b=0),
#     )

#     st.plotly_chart(fig)

# # Add button to Streamlit
# if st.button("Show 3D Flight Map"):
#     plot_3d_globe()

# import folium
# import requests
# import streamlit as st
# from streamlit_folium import folium_static

# # Function to get real-time flight data
# def get_live_flights():
#     url = "https://opensky-network.org/api/states/all"
#     response = requests.get(url)
#     flights = response.json()["states"]
#     return flights

# # Function to plot flights on a map
# def plot_flights():
#     st.subheader("🌍 **Live Flight Tracker**")
    
#     flights = get_live_flights()
#     m = folium.Map(location=[20, 0], zoom_start=2)  # Center map globally

#     for flight in flights[:100]:  # Limit to first 100 flights
#         if flight[5] and flight[6]:  # Ensure latitude & longitude exist
#             folium.Marker(
#                 [flight[6], flight[5]],
#                 popup=f"✈️ Flight {flight[1]}",
#                 tooltip="Click for details",
#                 icon=folium.Icon(color="blue", icon="plane", prefix="fa"),
#             ).add_to(m)

#     folium_static(m)  # Display map in Streamlit

# # Add to Streamlit UI
# if st.button("Track Live Flights"):
#     plot_flights()


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

    
    def get_base64_image(image_path):
       with open(image_path, "rb") as img_file:
         return base64.b64encode(img_file.read()).decode()
       
    image_path = "C:\\Remarkable One\\UTA MSBA\\Spring'25\\Business Symposium\\Pictures\\riku-lu-sjrPOiGWN7A-unsplash.jpg"
    encoded_image = get_base64_image(image_path)

    page_bg_img = f'''
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                    url("data:image/jpg;base64,{encoded_image}");
        background-size: cover;
        background-position: center
        position : relative;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.markdown(
     """
    <style>
     /* 🌙 Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #181818 !important; /* Darker Sidebar */
        border-radius: 15px;
        padding: 20px;
        color: white;
    }

    /* 🎨 Sidebar Headings */
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
        color: #00c8ff !important; /* Light Blue Titles */
        font-size: 20px !important;
    }

    /* 🔘 Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b, #ff7878) !important; /* Gradient Effect */
        color: white !important;
        border-radius: 12px !important;
        padding: 10px 15px !important;
        font-weight: bold !important;
        transition: all 0.3s ease-in-out !important;
    }

    /* 🔄 Button Hover Effect */
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff7878, #ff4b4b) !important;
        transform: scale(1.05) !important; /* Slight pop-up effect */
    }

    /* 💬 Chat Bubble Styles */
    div[data-testid="stChatMessage"] {
        background-color: rgba(0, 123, 255, 0.1) !important; /* Light Blue */
        padding: 12px !important;
        border-radius: 15px !important;
        margin: 8px 0 !important;
        width: fit-content !important; /* Ensure width is correct */
    }

    /* 🧑‍💻 User Message */
    div[data-testid="stChatMessage"]:nth-of-type(odd) {
        background-color: rgba(255, 255, 255, 0.1) !important; /* Light Gray for User */
        text-align: right !important;
        margin-left: auto !important; /* Push to right */
    }

    /* 🤖 AI Message */
    div[data-testid="stChatMessage"]:nth-of-type(even) {
        background-color: rgba(0, 255, 180, 0.2) !important; /* Greenish for AI */
        text-align: left !important;
        margin-right: auto !important; /* Push to left */
    }

    </style>
    """,
    unsafe_allow_html=True
    )
    
    st.markdown(
    """
    <style>
    /* 🌙 Make Chat Background Transparent */
    div[data-testid="stChatMessage"] {
        background: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        color: white !important;
    }

    /* 🧑‍💻 User Message */
    div[data-testid="stChatMessage"]:nth-of-type(odd) {
        background: none !important;
        background-color: rgba(255, 255, 255, 0.1) !important; /* Light Gray */
        text-align: right !important;
        margin-left: auto !important;
        border-radius: 15px !important;
    }

    /* 🤖 AI Message */
    div[data-testid="stChatMessage"]:nth-of-type(even) {
        background: none !important;
        background-color: rgba(0, 255, 180, 0.2) !important; /* Light Green */
        text-align: left !important;
        margin-right: auto !important;
        border-radius: 15px !important;
    }

    /* 📜 Remove Default Chat Box Background */
    [data-testid="stChatContainer"] {
        background: none !important;
        background-color: transparent !important;
    }

    /* 🔄 Remove Chat Input Box Background */
    [data-testid="stChatInputContainer"] {
        background: none !important;
        background-color: transparent !important;
    }

    </style>
    """,
    unsafe_allow_html=True
    )

 

    st.markdown('''
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap');

        .title-text {
           font-family: 'Great Vibes', cursive;
           font-size: 60px;
           font-weight: bold;
           color: white;
           text-align: center;
           text-shadow: 4px 4px 6px rgba(0, 0, 0, 0.8);
           padding-top: 20px; /* Add spacing from the top */
        }

        </style>
    ''', unsafe_allow_html=True)

    st.markdown('<h1 class="title-text">✈️ Aviation  AI  Chatbot</h1>', unsafe_allow_html=True)



 
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
            label="*Memory Window*",
            value=3,
            min_value=1,
            max_value=10,
            step=1,
            help=(
                """The size of history chats that is kept for context. A value of, say,
                3, keeps the last three pairs of prompts and responses, i.e. the last
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

    user_input = get_text()

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
        st.session_state["past"].append(user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})

        if df.empty:
            response = chat_api([{"role": "user", "content": user_input}], **model_params)  # Use LLM for general queries
        else:
            response = chat_with_data_api(df, **model_params)  # Use LLM with DataFrame
        
        import re

        def clean_response(text):
            """
            Removes code blocks and markdown formatting from responses.
            """
            text = re.sub(r"```python.*?```", "", text, flags=re.DOTALL)  # Remove Python code blocks
            text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # Remove any other code blocks
            text = re.sub(r"#.*?\n", "", text)  # Remove comments (lines starting with #)
            text = re.sub(r"\n+", "\n", text).strip()  # Remove extra newlines
            return text
        
        if response is not None:
            st.session_state["generated"].append(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})

        st.session_state["messages"] = st.session_state["messages"][-10:]
        st.session_state["past"] = st.session_state["past"][-10:]
        st.session_state["generated"] = st.session_state["generated"][-10:]

        if not df.empty:
            if "forecast" in user_input.lower():
                try:
                    target_column, predictions = forecasting(df)  
                    st.write(f"Predictions for **{target_column}**:")
                    st.write(predictions)
                except Exception as e:
                    st.error(f"Error in forecasting: {e}")

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            if i < len(st.session_state["past"]):
                message(
                   st.session_state["past"][i],
                   is_user=True,
                   key=str(i) + "_user"
                )
        
            response_content = st.session_state["generated"][i]
            message(response_content, key=str(i))

            if "```python" in response_content:
                code = extract_python_code(response_content)
                if code:
                    try:
                        exec(code)  
                    except Exception as e:
                        st.error(f"Error executing the plot: {e}")


if __name__ == "__main__":
    chat_with_data()
