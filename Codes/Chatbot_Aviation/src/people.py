import streamlit as st
import os
import pandas as pd
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Aviation AI Chatbot", layout="wide")

# Load background image
page_bg_img = '''
<style>
.stApp {
    background: url("https://source.unsplash.com/1600x900/?aviation,technology");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .chat-bubble {
        background-color: rgba(0, 123, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .sidebar .block-container {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 8px 15px;
    }
    .stButton>button:hover {
        background-color: #ff7878;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for chatbot settings
st.sidebar.title("🚀 Flight AI Assistant")
st.sidebar.write("Customize your chatbot experience:")

model_choice = st.sidebar.selectbox("Choose AI Model", ["gpt-3.5-turbo", "gpt-4"]) 
temp_slider = st.sidebar.slider("Creativity (Temperature)", 0.0, 2.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 0, 4096, 256)
response_precision = st.sidebar.slider("Response Precision", 0.0, 1.0, 0.5)
memory_window = st.sidebar.slider("Memory Window", 1, 10, 3)

# Main Chatbot UI
st.title("✈️ Aviation AI Chatbot")
st.write("Welcome to the **Aviation AI Chatbot**! This assistant helps you analyze airline trends, predict flights, and visualize aviation data.")

# File Upload Section
uploaded_file = st.file_uploader("📂 Upload CSV, PDF, or Text file", type=["csv", "pdf", "txt"])
df = None  # Initialize dataframe
if uploaded_file:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        df = pd.read_csv(file_path) if uploaded_file.name.endswith(".csv") else None
        st.success(f"✅ File {uploaded_file.name} uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")

# Chatbox
st.subheader("💬 Chat with Aviation AI")
user_input = st.text_input("Type your question here...")
if st.button("Ask AI ✨"):
    if user_input:
        st.markdown(f'<div class="chat-bubble">🧑‍💼 You: {user_input}</div>', unsafe_allow_html=True)
        response = "🔍 AI: Analyzing your query..."  # Placeholder AI response
        st.markdown(f'<div class="chat-bubble">{response}</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a question!")

# Quick Actions
st.subheader("🚀 Quick Actions")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📊 Generate Charts"):
        st.write("Generating charts...")
with col2:
    if st.button("🔍 Analyze Data"):
        st.write("Analyzing uploaded data...")
with col3:
    if st.button("✈️ Predict Flights"):
        st.write("Predicting flight trends...")

# Footer
st.markdown("---")
st.markdown("🛫 Developed for smarter aviation insights. 💡")
