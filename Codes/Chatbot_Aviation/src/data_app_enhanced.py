import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_chat import message
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from main_enhanced import get_text, load_custom_css, display_welcome_message
from llm_utils import chat_with_data_api, chat_api, extract_python_code
from forecasting import forecasting, time_series_forecasting
import uuid
import time
from datetime import datetime

# Enhanced styling and configuration
st.set_page_config(
    page_title="Aviation Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHAT_HISTORY_FILE = "chat_history.json"

def enhanced_data_sidebar():
    """Enhanced sidebar for data analysis features"""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">📊 Aviation Analytics</h2>
        <p style="color: #f0f0f0; margin: 0; font-size: 0.9rem;">Data-Driven Flight Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🎛️ AI Configuration")
    
    model = st.sidebar.selectbox(
        "🤖 AI Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"],
        help="Select the AI model for data analysis"
    )
    
    temperature = st.sidebar.slider(
        "🔥 Analysis Creativity",
        0.0, 2.0, 0.3, 0.01,
        help="Lower values for more analytical responses"
    )
    
    max_tokens = st.sidebar.slider(
        "📝 Response Detail",
        100, 4096, 1024, 50,
        help="Maximum detail level for analysis responses"
    )
    
    top_p = st.sidebar.slider(
        "🎯 Analysis Focus",
        0.0, 1.0, 0.8, 0.01,
        help="Focus level for data analysis responses"
    )
    
    memory_window = st.sidebar.slider(
        "🧠 Memory Window",
        value=5,
        min_value=1,
        max_value=15,
        step=1,
        help="Number of previous interactions to remember for context"
    )
    
    st.sidebar.markdown("---")
    
    # Data analysis quick actions
    st.sidebar.markdown("### ⚡ Quick Analysis")
    if st.sidebar.button("📈 Generate Summary Stats"):
        st.session_state["quick_analysis"] = "summary"
    
    if st.sidebar.button("📊 Create Visualization"):
        st.session_state["quick_analysis"] = "visualization"
    
    if st.sidebar.button("🔮 Forecast Trends"):
        st.session_state["quick_analysis"] = "forecast"
    
    if st.sidebar.button("🔄 Clear Analysis"):
        for key in ["messages", "past", "generated", "plot_generated"]:
            if key in st.session_state:
                st.session_state[key] = []
        st.rerun()
    
    # Data insights
    st.sidebar.markdown("### 💡 Analysis Tips")
    analysis_tips = [
        "Ask for correlations between different variables",
        "Request time-series analysis for temporal data",
        "Try asking for statistical significance tests",
        "Request outlier detection and analysis",
        "Ask for predictive modeling insights"
    ]
    tip_index = int(time.time()) % len(analysis_tips)
    st.sidebar.info(f"📋 {analysis_tips[tip_index]}")
    
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }, memory_window

def enhanced_file_uploader():
    """Enhanced file upload section with better styling"""
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #667eea;">
        <h3 style="margin-top: 0; color: #333;">📁 Upload Your Aviation Dataset</h3>
        <p style="color: #666; margin-bottom: 1rem;">Upload CSV, PDF, or text files containing flight data, airport statistics, or airline information.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your aviation data file",
        type=["csv", "pdf", "txt"],
        help="Supported formats: CSV (preferred for analysis), PDF, and TXT files"
    )
    
    return uploaded_file

def enhanced_eda(df, filename):
    """Enhanced exploratory data analysis with aviation-specific insights"""
    if df is None or df.empty:
        return
    
    st.markdown("### 📊 Dataset Overview")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Basic Info", "📈 Statistics", "🔍 Data Quality", "✈️ Aviation Insights"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("File Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(col_info, use_container_width=True)
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        st.subheader("Statistical Summary")
        if not df.select_dtypes(include=[np.number]).empty:
            st.dataframe(df.describe(), use_container_width=True)
        else:
            st.info("No numerical columns found for statistical analysis.")
        
        # Distribution plots for numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.subheader("Distribution Analysis")
            selected_col = st.selectbox("Select column for distribution plot", numeric_cols)
            if selected_col:
                fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Data Quality Assessment")
        
        # Missing values heatmap
        if df.isnull().sum().sum() > 0:
            st.subheader("Missing Values Pattern")
            missing_data = df.isnull().sum()
            missing_df = pd.DataFrame({'Column': missing_data.index, 'Missing Count': missing_data.values})
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            
            fig = px.bar(missing_df, x='Column', y='Missing Count', title="Missing Values by Column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values detected!")
        
        # Duplicate detection
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates} duplicate rows")
        else:
            st.success("✅ No duplicate rows detected!")
    
    with tab4:
        st.subheader("Aviation-Specific Analysis")
        
        # Detect aviation-related columns
        aviation_keywords = {
            'airport': ['airport', 'origin', 'destination', 'dep', 'arr'],
            'flight': ['flight', 'airline', 'carrier', 'aircraft'],
            'time': ['time', 'date', 'delay', 'duration'],
            'location': ['city', 'country', 'state', 'latitude', 'longitude']
        }
        
        detected_categories = {}
        for category, keywords in aviation_keywords.items():
            matching_cols = [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in keywords)]
            if matching_cols:
                detected_categories[category] = matching_cols
        
        if detected_categories:
            st.success("✈️ Aviation data structure detected!")
            for category, cols in detected_categories.items():
                st.write(f"**{category.title()} columns:** {', '.join(cols)}")
        else:
            st.info("💡 Upload aviation-specific data for specialized insights")
        
        # Quick aviation insights
        if 'delay' in ' '.join(df.columns).lower():
            st.subheader("Flight Delay Insights")
            delay_cols = [col for col in df.columns if 'delay' in col.lower()]
            if delay_cols and not df[delay_cols[0]].empty:
                avg_delay = df[delay_cols[0]].mean()
                st.metric("Average Delay", f"{avg_delay:.1f} minutes" if not pd.isna(avg_delay) else "N/A")

def get_aviation_data_greeting():
    """Generate aviation-focused data analysis greeting"""
    greetings = [
        "Welcome to Aviation Data Analytics! I'm ready to help you uncover insights from your flight data. Upload your dataset and let's start exploring!",
        "Hello, data analyst! I'm your Aviation AI specialized in flight data analysis. Whether it's passenger traffic, delays, or route optimization - I'm here to help!",
        "Greetings! I'm equipped with advanced aviation analytics capabilities. Upload your data and ask me to analyze trends, create visualizations, or generate forecasts!",
        "Welcome aboard the data analysis flight! I can help you analyze airport performance, airline efficiency, flight patterns, and much more. What would you like to explore?"
    ]
    return greetings[int(time.time()) % len(greetings)]

def handle_quick_analysis(df, analysis_type):
    """Handle quick analysis requests from sidebar"""
    if df is None or df.empty:
        return "Please upload a dataset first to perform analysis."
    
    if analysis_type == "summary":
        return f"Here's a quick summary of your dataset:\n- **Rows:** {len(df)}\n- **Columns:** {len(df.columns)}\n- **Numerical columns:** {len(df.select_dtypes(include=[np.number]).columns)}\n- **Text columns:** {len(df.select_dtypes(include=['object']).columns)}\n\nWhat specific analysis would you like me to perform?"
    
    elif analysis_type == "visualization":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            return f"I can create various visualizations with your data. Try asking: 'Create a scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}' or 'Show me a correlation heatmap of all numerical variables'."
        else:
            return "I can create visualizations with your data. Try asking for bar charts, histograms, or time series plots based on your columns."
    
    elif analysis_type == "forecast":
        date_cols = df.select_dtypes(include=['datetime']).columns
        if len(date_cols) > 0:
            return f"I can help with time series forecasting using your date column '{date_cols[0]}'. Ask me to 'forecast the trend' or 'predict future values'."
        else:
            return "For forecasting, I'll need a date/time column in your data. I can still help with statistical predictions and trend analysis!"

def enhanced_chat_with_data():
    """Enhanced data chat interface with improved UI and functionality"""
    
    # Load custom CSS
    load_custom_css()
    
    # Enhanced header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
        <h1 style="color: white; font-size: 3rem; font-weight: bold; text-align: center; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">📊 Aviation Data Analytics</h1>
        <p style="color: #f0f0f0; font-size: 1.2rem; text-align: center; margin-bottom: 1rem;">Advanced AI-Powered Flight Data Analysis & Visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    for key in ["messages", "past", "generated", "reset_chat", "plot_generated"]:
        if key not in st.session_state:
            if key == "reset_chat" or key == "plot_generated":
                st.session_state[key] = False
            else:
                st.session_state[key] = []
    
    # Sidebar configuration
    with st.sidebar:
        model_params, memory_window = enhanced_data_sidebar()
    
    # File upload section
    uploaded_file = enhanced_file_uploader()
    
    df = None
    if uploaded_file:
        # Create uploads directory if it doesn't exist
        upload_folder = "uploads"
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        file_path = os.path.join(upload_folder, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Load data based on file type
            ext = os.path.splitext(uploaded_file.name)[-1].lower()
            if ext == ".csv":
                df = pd.read_csv(file_path)
            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as file:
                    df = pd.DataFrame({"text": file.readlines()})
            elif ext == ".pdf":
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                df = pd.DataFrame({"text": [text]})
            
            st.success(f"✅ Successfully loaded: {uploaded_file.name}")
            
            # Enhanced EDA
            enhanced_eda(df, uploaded_file.name)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    
    # Initialize chat with aviation data greeting
    if not st.session_state["generated"] or st.session_state.get("reset_chat", False):
        greeting = get_aviation_data_greeting()
        st.session_state["generated"] = [greeting]
        st.session_state["messages"] = [
            {"role": "system", "content": """You are an expert Aviation Data Analyst AI with specialized knowledge in:
            - Flight operations analysis and airport performance metrics
            - Airline industry KPIs and operational efficiency
            - Passenger traffic patterns and seasonal trends
            - Aircraft utilization and route optimization
            - Aviation safety statistics and delay analysis
            - Air traffic management and capacity planning
            
            When analyzing data:
            1. Always provide context-aware insights specific to aviation
            2. Suggest relevant visualizations using plotly
            3. Identify key performance indicators and trends
            4. Offer actionable recommendations based on the data
            5. Use aviation terminology and industry knowledge
            
            Be conversational, insightful, and always relate findings back to aviation operations."""},
            {"role": "assistant", "content": greeting}
        ]
        st.session_state["reset_chat"] = False
    
    # Handle quick analysis from sidebar
    if "quick_analysis" in st.session_state:
        analysis_type = st.session_state["quick_analysis"]
        quick_response = handle_quick_analysis(df, analysis_type)
        st.session_state["generated"].append(quick_response)
        st.session_state["messages"].append({"role": "assistant", "content": quick_response})
        del st.session_state["quick_analysis"]
    
    # Chat input
    user_input = get_text()
    
    # Process user input
    if user_input:
        # Check for duplicate input
        if (len(st.session_state["past"]) > 0 and 
            user_input == st.session_state["past"][-1]):
            user_input = ""
        
        if user_input:
            # Add user input to session state
            st.session_state["past"].append(user_input)
            st.session_state["messages"].append({"role": "user", "content": user_input})
            
            # Manage memory window
            if len(st.session_state["messages"]) > 2 * memory_window + 1:
                st.session_state["messages"] = (
                    [st.session_state["messages"][0]] +  # Keep system message
                    st.session_state["messages"][-(2 * memory_window):]  # Keep recent messages
                )
            
            # Show typing indicator
            with st.empty():
                st.markdown('<p style="color: #6c757d; font-style: italic;">🤖 Analyzing your request...</p>', unsafe_allow_html=True)
                time.sleep(1)
            
            try:
                # Get AI response
                if df.empty:
                    response = chat_api([{"role": "user", "content": user_input}], **model_params)
                else:
                    response = chat_with_data_api(df, **model_params)
                
                if response and response not in st.session_state["generated"]:
                    # Format response with aviation context
                    if any(keyword in response.lower() for keyword in ['airport', 'flight', 'airline', 'aircraft']):
                        formatted_response = f"✈️ **Aviation Analysis**: {response}"
                    elif 'plot' in response.lower() or 'chart' in response.lower() or '```python' in response:
                        formatted_response = f"📊 **Data Visualization**: {response}"
                    elif any(keyword in response.lower() for keyword in ['forecast', 'predict', 'trend']):
                        formatted_response = f"🔮 **Predictive Insights**: {response}"
                    else:
                        formatted_response = response
                    
                    st.session_state["generated"].append(formatted_response)
                    st.session_state["messages"].append({"role": "assistant", "content": formatted_response})
                    
            except Exception as e:
                error_response = f"I apologize, but I encountered an issue while analyzing your request: {str(e)}. Please try rephrasing your question or check your API configuration."
                st.session_state["generated"].append(error_response)
    
    # Display chat history
    if st.session_state["generated"]:
        st.markdown("### 💬 Analysis Conversation")
        st.markdown('<div style="background: #f8f9fa; border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem;">', unsafe_allow_html=True)
        
        for i in range(len(st.session_state["generated"])):
            # Display AI message
            message(st.session_state["generated"][i], key=str(i))
            
            # Display user message if exists
            if i < len(st.session_state["past"]):
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + "_user"
                )
            
            # Handle code execution for plots
            response_content = st.session_state["generated"][i]
            if "```python" in response_content:
                code = extract_python_code(response_content)
                if code:
                    try:
                        # Create unique key for each plot
                        unique_key = str(uuid.uuid4())
                        code = code.replace("fig.show()", f"st.plotly_chart(fig, use_container_width=True, key='{unique_key}')")
                        
                        # Execute the code
                        exec(code, {
                            "st": st, 
                            "plt": plt, 
                            "pd": pd, 
                            "sns": sns, 
                            "px": px,
                            "go": go,
                            "df": df
                        })
                        st.session_state["plot_generated"] = True
                        
                    except Exception as e:
                        st.error(f"❌ Error executing visualization: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Trim session state to prevent memory issues
        max_history = 20
        if len(st.session_state["generated"]) > max_history:
            st.session_state["messages"] = st.session_state["messages"][:1] + st.session_state["messages"][-max_history:]
            st.session_state["past"] = st.session_state["past"][-max_history:]
            st.session_state["generated"] = st.session_state["generated"][-max_history:]
    
    # Forecasting section
    if not df.empty and any(keyword in ' '.join(st.session_state["past"]).lower() for keyword in ["forecast", "predict"]):
        try:
            with st.expander("🔮 Advanced Forecasting", expanded=False):
                st.markdown("### Time Series Forecasting")
                forecast_result = time_series_forecasting(df)
                if forecast_result is not None:
                    st.write("**Forecast Results:**")
                    st.write(forecast_result)
                else:
                    st.info("💡 For time series forecasting, ensure your data has a datetime column and numerical values to predict.")
        except Exception as e:
            st.warning(f"Forecasting unavailable: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 1rem;">
        <p>📊 <strong>Aviation Analytics:</strong> Powered by advanced AI models 
        | 🔐 <strong>Data Security:</strong> Your data is processed locally and not stored
        | 💡 <strong>Tip:</strong> Try asking for specific aviation metrics or industry benchmarks</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    enhanced_chat_with_data()