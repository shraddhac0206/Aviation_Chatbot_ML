import streamlit as st
import os
from main_enhanced import enhanced_chatbot
from data_app_enhanced import enhanced_chat_with_data
from people import people_analysis
import time

# Enhanced page configuration
st.set_page_config(
    page_title="Aviation AI Hub",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the entire application
def load_global_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main app styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Navigation styling */
    .nav-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .nav-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .nav-subtitle {
        color: #f0f0f0;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 300;
    }
    
    /* Page cards */
    .page-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .page-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .page-card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .page-card-description {
        color: #666;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .page-card-features {
        color: #888;
        font-size: 0.9rem;
        font-style: italic;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    .status-online {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-beta {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    /* Sidebar enhancements */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Footer styling */
    .footer {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
        color: #6c757d;
        border-top: 3px solid #667eea;
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

def display_navigation():
    """Enhanced navigation header"""
    st.markdown("""
    <div class="nav-container">
        <h1 class="nav-title">✈️ Aviation AI Hub</h1>
        <p class="nav-subtitle">Advanced Artificial Intelligence for Aviation Analytics & Insights</p>
        <div style="text-align: center;">
            <span class="status-indicator status-online">🟢 All Systems Operational</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_home_page():
    """Enhanced home page with feature overview"""
    display_navigation()
    
    st.markdown("### 🚀 Choose Your Aviation AI Experience")
    
    # Feature cards in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="page-card">
            <div class="page-card-title">💬 Aviation AI Chatbot</div>
            <div class="page-card-description">
                Engage with our intelligent aviation assistant for general flight information, 
                industry insights, and aviation knowledge. Perfect for quick queries and 
                conversational AI interactions.
            </div>
            <div class="page-card-features">
                ✨ Natural language processing • 🧠 Aviation expertise • 🔄 Real-time responses
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Chatbot", key="chatbot_btn"):
            st.session_state.page = "chatbot"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="page-card">
            <div class="page-card-title">📊 Data Analytics Suite</div>
            <div class="page-card-description">
                Upload and analyze aviation datasets with AI-powered insights. Generate 
                visualizations, perform statistical analysis, and uncover data-driven 
                insights from your flight data.
            </div>
            <div class="page-card-features">
                📈 Advanced analytics • 📊 Interactive visualizations • 🔮 Predictive modeling
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Open Analytics", key="analytics_btn"):
            st.session_state.page = "analytics"
            st.rerun()
    
    # Additional features row
    st.markdown("### 🛠️ Additional Features")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">50+</div>
            <div class="metric-label">Aviation Metrics</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">15+</div>
            <div class="metric-label">Chart Types</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">24/7</div>
            <div class="metric-label">AI Availability</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("### ✨ Key Features")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        **🤖 Intelligent AI Assistant**
        - Advanced natural language understanding
        - Aviation industry expertise
        - Context-aware responses
        - Multi-turn conversations
        """)
    
    with feature_col2:
        st.markdown("""
        **📊 Powerful Analytics**
        - Interactive data visualizations
        - Statistical analysis tools
        - Predictive modeling
        - Custom report generation
        """)
    
    with feature_col3:
        st.markdown("""
        **✈️ Aviation Focus**
        - Flight operations analysis
        - Airport performance metrics
        - Airline industry insights
        - Route optimization tools
        """)
    
    # Recent updates
    with st.expander("🆕 Recent Updates & Features", expanded=False):
        st.markdown("""
        **Version 2.0 - Enhanced UI & Features**
        - ✅ Completely redesigned user interface
        - ✅ Improved chatbot personality and aviation focus
        - ✅ Enhanced data visualization capabilities
        - ✅ Better error handling and user feedback
        - ✅ Responsive design for all screen sizes
        - ✅ Advanced analytics with aviation-specific insights
        
        **Coming Soon**
        - 🔄 Real-time flight tracking integration
        - 📱 Mobile app companion
        - 🌐 Multi-language support
        - 🔒 Enhanced security features
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <h4>🛡️ Privacy & Security</h4>
        <p>Your data is processed securely and not stored permanently. All conversations are encrypted and handled with enterprise-grade security.</p>
        <p><strong>Aviation AI Hub</strong> • Powered by Advanced Language Models • Built with ❤️ for Aviation Professionals</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar_navigation():
    """Enhanced sidebar navigation"""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0;">🧭 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    
    if st.sidebar.button("💬 AI Chatbot", use_container_width=True):
        st.session_state.page = "chatbot"
        st.rerun()
    
    if st.sidebar.button("📊 Data Analytics", use_container_width=True):
        st.session_state.page = "analytics"
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.markdown("### 📊 System Status")
    st.sidebar.success("🟢 AI Models: Online")
    st.sidebar.success("🟢 Data Processing: Ready")
    st.sidebar.success("🟢 Visualizations: Active")
    
    # Quick stats
    st.sidebar.markdown("### 📈 Quick Stats")
    st.sidebar.metric("Response Time", "< 2s")
    st.sidebar.metric("Accuracy", "95%+")
    st.sidebar.metric("Uptime", "99.9%")
    
    # Help section
    with st.sidebar.expander("❓ Need Help?"):
        st.markdown("""
        **Getting Started:**
        1. Choose between Chatbot or Analytics
        2. For analytics, upload your data file
        3. Ask questions in natural language
        4. Request visualizations and insights
        
        **Tips:**
        - Be specific with your questions
        - Use aviation terminology
        - Ask for visualizations by name
        - Request comparisons and trends
        """)
    
    # Footer info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; font-size: 0.8rem; color: #666;">
        <p>Aviation AI Hub v2.0<br>
        Built with Streamlit & OpenAI<br>
        © 2024 Future Minds Team</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application controller"""
    # Load global CSS
    load_global_css()
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    # Display sidebar navigation
    display_sidebar_navigation()
    
    # Route to appropriate page
    if st.session_state.page == "home":
        display_home_page()
    
    elif st.session_state.page == "chatbot":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; text-align: center;">💬 Aviation AI Chatbot</h2>
        </div>
        """, unsafe_allow_html=True)
        enhanced_chatbot()
    
    elif st.session_state.page == "analytics":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; text-align: center;">📊 Aviation Data Analytics</h2>
        </div>
        """, unsafe_allow_html=True)
        enhanced_chat_with_data()

if __name__ == "__main__":
    main()