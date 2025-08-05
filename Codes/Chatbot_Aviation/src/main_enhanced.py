import streamlit as st
from streamlit_chat import message
import os
import json
import pandas as pd
import numpy as np
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns
from llm_utils import chat_api, chat_with_data_api
import time
from datetime import datetime

# Enhanced styling and configuration
st.set_page_config(
    page_title="Aviation AI Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: #f0f0f0;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Chat container styling */
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
    }
    
    /* Status indicators */
    .status-online {
        color: #28a745;
        font-weight: bold;
    }
    
    .typing-indicator {
        color: #6c757d;
        font-style: italic;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

def get_text():
    """Enhanced input with placeholder text"""
    return st.chat_input("Ask me anything about aviation, flight data, or request visualizations...")

def enhanced_sidebar():
    """Enhanced sidebar with better styling and features"""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">✈️ Flight AI Assistant</h2>
        <p style="color: #f0f0f0; margin: 0; font-size: 0.9rem;">Your Aviation Intelligence Partner</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🎛️ AI Configuration")
    
    model = st.sidebar.selectbox(
        "🤖 AI Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"],
        help="Select the AI model for generating responses"
    )
    
    temperature = st.sidebar.slider(
        "🔥 Creativity Level",
        0.0, 2.0, 0.7, 0.01,
        help="Higher values make responses more creative and varied"
    )
    
    max_tokens = st.sidebar.slider(
        "📝 Response Length",
        50, 4096, 512, 50,
        help="Maximum length of AI responses"
    )
    
    top_p = st.sidebar.slider(
        "🎯 Response Focus",
        0.0, 1.0, 0.9, 0.01,
        help="Lower values make responses more focused and deterministic"
    )
    
    st.sidebar.markdown("---")
    
    # Status indicator
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 0.5rem;">
        <span class="status-online">🟢 AI Assistant Online</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    if st.sidebar.button("🔄 Clear Chat History"):
        st.session_state["messages"] = []
        st.session_state["past"] = []
        st.session_state["generated"] = []
        st.rerun()
    
    if st.sidebar.button("📊 Sample Aviation Query"):
        st.session_state["sample_query"] = "Show me the top 10 busiest airports by passenger traffic"
    
    # Aviation facts sidebar
    st.sidebar.markdown("### ✈️ Aviation Fact")
    aviation_facts = [
        "The Boeing 747 can carry up to 660 passengers",
        "Commercial aircraft cruise at about 35,000-42,000 feet",
        "The busiest airport in the world is Hartsfield-Jackson Atlanta",
        "A typical commercial flight uses about 1 gallon of fuel per second",
        "There are over 100,000 flights per day worldwide"
    ]
    fact_index = int(time.time()) % len(aviation_facts)
    st.sidebar.info(f"💡 {aviation_facts[fact_index]}")
    
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

def display_welcome_message():
    """Enhanced welcome message with better styling"""
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">✈️ Aviation AI Assistant</h1>
        <p class="header-subtitle">Your Intelligent Companion for Aviation Data Analysis & Flight Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Data Analysis</h3>
            <p>Upload and analyze aviation datasets with AI-powered insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🛫 Flight Intelligence</h3>
            <p>Get real-time flight information and aviation industry insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Visualizations</h3>
            <p>Generate interactive charts and graphs from your aviation data</p>
        </div>
        """, unsafe_allow_html=True)

def get_aviation_greeting():
    """Generate a more genuine aviation-focused greeting"""
    greetings = [
        "Welcome aboard! I'm your Aviation AI Assistant, ready to help you navigate through flight data and aviation insights. What would you like to explore today?",
        "Greetings, aviation enthusiast! I'm here to assist you with flight analysis, airport statistics, and airline data. How can I help you soar to new insights?",
        "Hello! As your dedicated Aviation AI, I can help you analyze flight patterns, airport performance, and airline operations. What aviation topic interests you today?",
        "Welcome to your personal aviation command center! I'm equipped to handle flight data analysis, route optimization insights, and industry trends. What's your flight plan for today's analysis?"
    ]
    return greetings[int(time.time()) % len(greetings)]

def display_typing_indicator():
    """Show typing indicator for better UX"""
    with st.empty():
        for i in range(3):
            st.markdown('<p class="typing-indicator">AI is thinking' + '.' * (i + 1) + '</p>', unsafe_allow_html=True)
            time.sleep(0.5)
        st.empty()

def format_ai_response(response):
    """Enhanced response formatting"""
    # Add aviation context to responses
    if any(keyword in response.lower() for keyword in ['airport', 'flight', 'airline', 'aircraft']):
        response = f"✈️ **Aviation Insight**: {response}"
    elif 'plot' in response.lower() or 'chart' in response.lower():
        response = f"📊 **Data Visualization**: {response}"
    elif any(keyword in response.lower() for keyword in ['forecast', 'predict', 'trend']):
        response = f"🔮 **Predictive Analysis**: {response}"
    
    return response

def enhanced_chatbot():
    """Enhanced chatbot with improved UI and genuine aviation focus"""
    
    # Load custom CSS
    load_custom_css()
    
    # Display welcome message
    display_welcome_message()
    
    # Sidebar configuration
    with st.sidebar:
        model_params = enhanced_sidebar()
    
    # Initialize session state with aviation-focused greeting
    if "messages" not in st.session_state or st.session_state.get("reset_chat", False):
        aviation_greeting = get_aviation_greeting()
        st.session_state["messages"] = [
            {"role": "system", "content": """You are an expert Aviation AI Assistant with deep knowledge of:
            - Flight operations and air traffic management
            - Airport operations and logistics
            - Airline industry trends and analysis
            - Aircraft specifications and performance
            - Aviation safety and regulations
            - Flight data analysis and visualization
            
            Always provide helpful, accurate, and aviation-focused responses. When users ask for data visualization,
            generate appropriate code using plotly. Be conversational but professional, and show genuine interest
            in helping users understand aviation data and concepts."""},
            {"role": "assistant", "content": aviation_greeting}
        ]
        st.session_state["reset_chat"] = False
    
    if "past" not in st.session_state:
        st.session_state["past"] = []
    
    if "generated" not in st.session_state:
        st.session_state["generated"] = [get_aviation_greeting()]
    
    # Handle sample query from sidebar
    if "sample_query" in st.session_state:
        user_input = st.session_state["sample_query"]
        del st.session_state["sample_query"]
    else:
        user_input = get_text()
    
    # Process user input
    if user_input:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["past"].append(user_input)
        
        # Show typing indicator
        with st.container():
            display_typing_indicator()
        
        try:
            # Get AI response
            response = chat_api(st.session_state["messages"], **model_params)
            
            if response:
                # Format the response
                formatted_response = format_ai_response(response)
                
                # Add to session state
                st.session_state["generated"].append(formatted_response)
                st.session_state["messages"].append({"role": "assistant", "content": formatted_response})
                
        except Exception as e:
            error_response = f"I apologize, but I encountered an issue: {str(e)}. Please try rephrasing your question or check your API configuration."
            st.session_state["generated"].append(error_response)
    
    # Display chat history with enhanced styling
    if st.session_state["generated"]:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for i in range(len(st.session_state["generated"])):
            # Display AI message
            message(st.session_state["generated"][i], key=str(i))
            
            # Display user message if it exists
            if i < len(st.session_state["past"]):
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + "_user"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 1rem;">
        <p>🛡️ <strong>Privacy Notice:</strong> Your conversations are not stored permanently. 
        | 🔧 <strong>Support:</strong> For technical issues, please check your API configuration.
        | ✈️ <strong>Aviation Data:</strong> Powered by advanced AI models</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    enhanced_chatbot()