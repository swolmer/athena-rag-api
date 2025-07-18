"""
Athen.ai Healthcare RAG Platform - Dashboard
RunPod Deployment Dashboard
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from typing import Dict, List

# Configuration
API_BASE = "http://localhost:8000"
API_TOKEN = os.getenv("ATHEN_JWT_TOKEN", "kilment1234")

# Page configuration
st.set_page_config(
    page_title="Athen.ai Healthcare RAG Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None):
    """Make API request with authentication"""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        if method == "GET":
            response = requests.get(f"{API_BASE}{endpoint}", headers=headers)
        elif method == "POST":
            response = requests.post(f"{API_BASE}{endpoint}", headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Athen.ai Healthcare RAG Platform</h1>', unsafe_allow_html=True)
    st.markdown("### RunPod Deployment Dashboard")
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/200x100/2E86AB/FFFFFF?text=Athen.ai", width=200)
    st.sidebar.markdown("## Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Dashboard", "📁 Document Upload", "🤖 Training", "💬 Chat", "⚙️ Settings"]
    )
    
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "📁 Document Upload":
        show_upload_page()
    elif page == "🤖 Training":
        show_training_page()
    elif page == "💬 Chat":
        show_chat_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_dashboard():
    """Show main dashboard with metrics"""
    
    st.header("📊 Platform Overview")
    
    # Get metrics
    metrics = make_api_request("/dashboard/metrics")
    
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🏥 Organizations",
                value=metrics.get("total_organizations", 0)
            )
        
        with col2:
            st.metric(
                label="🤖 Active Trainings",
                value=metrics.get("active_trainings", 0)
            )
        
        with col3:
            st.metric(
                label="✅ Completed Trainings",
                value=metrics.get("completed_trainings", 0)
            )
        
        with col4:
            st.metric(
                label="📄 Total Documents",
                value=metrics.get("total_documents", 0)
            )
    
    # Projects overview
    st.subheader("📁 Active Projects")
    projects = make_api_request("/projects")
    
    if projects and projects.get("projects"):
        df = pd.DataFrame(projects["projects"])
        st.dataframe(df, use_container_width=True)
        
        # Training status chart
        if not df.empty:
            fig = px.pie(
                df, 
                names="training_status", 
                title="Training Status Distribution",
                color_discrete_map={
                    "completed": "#28a745",
                    "training": "#ffc107",
                    "not_started": "#6c757d",
                    "failed": "#dc3545"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No projects found. Upload documents to get started!")

def show_upload_page():
    """Show document upload page"""
    
    st.header("📁 Document Upload")
    
    # Organization selection
    org_id = st.text_input("Organization ID", placeholder="Enter organization ID (e.g., hospital_1)")
    
    if org_id:
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Medical Documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Upload medical documents for RAG training"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Upload {uploaded_file.name}"):
                    with st.spinner(f"Uploading {uploaded_file.name}..."):
                        files = {"file": uploaded_file.getvalue()}
                        headers = {"Authorization": f"Bearer {API_TOKEN}"}
                        
                        try:
                            response = requests.post(
                                f"{API_BASE}/rag/{org_id}/upload",
                                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                                headers=headers
                            )
                            
                            if response.status_code == 200:
                                st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                            else:
                                st.error(f"❌ Upload failed: {response.text}")
                        except Exception as e:
                            st.error(f"❌ Upload error: {str(e)}")

def show_training_page():
    """Show model training page"""
    
    st.header("🤖 Model Training")
    
    org_id = st.text_input("Organization ID", placeholder="Enter organization ID for training")
    
    if org_id:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Training"):
                with st.spinner("Starting training..."):
                    result = make_api_request(f"/rag/{org_id}/training/start", "POST")
                    if result:
                        st.success("✅ Training started successfully!")
        
        with col2:
            if st.button("📊 Check Status"):
                status = make_api_request(f"/rag/{org_id}/training/status")
                if status:
                    st.json(status)
        
        # Auto-refresh training status
        if st.checkbox("Auto-refresh status"):
            placeholder = st.empty()
            import time
            
            while True:
                status = make_api_request(f"/rag/{org_id}/training/status")
                if status:
                    with placeholder.container():
                        if status.get("status") == "training":
                            progress = status.get("progress", 0)
                            st.progress(progress / 100)
                            st.write(f"Training Progress: {progress}%")
                        elif status.get("status") == "completed":
                            st.success("🎉 Training completed!")
                            break
                        elif status.get("status") == "failed":
                            st.error("❌ Training failed!")
                            break
                        else:
                            st.info("Training not started")
                            break
                
                time.sleep(2)

def show_chat_page():
    """Show chat interface"""
    
    st.header("💬 Healthcare AI Chat")
    
    org_id = st.text_input("Organization ID", placeholder="Enter trained organization ID")
    
    if org_id:
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a medical question..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = make_api_request(
                        f"/rag/{org_id}/chat",
                        "POST",
                        {"question": prompt}
                    )
                    
                    if response:
                        answer = response.get("answer", "No response received")
                        st.markdown(answer)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Show additional info
                        with st.expander("📊 Response Details"):
                            st.json(response)
                    else:
                        st.error("Failed to get response from AI")

def show_settings_page():
    """Show settings page"""
    
    st.header("⚙️ Platform Settings")
    
    # API Configuration
    st.subheader("🔧 API Configuration")
    st.code(f"API Base URL: {API_BASE}")
    st.code(f"Authentication Token: {API_TOKEN[:8]}...")
    
    # Medical Templates
    st.subheader("🏥 Medical Templates")
    templates = make_api_request("/templates")
    
    if templates:
        for template in templates.get("templates", []):
            with st.expander(f"📋 {template['name']}"):
                st.write(template["description"])
                st.write("**Document Types:**")
                for doc_type in template["document_types"]:
                    st.write(f"- {doc_type}")
    
    # System Health
    st.subheader("💚 System Health")
    health = make_api_request("/health")
    
    if health:
        st.success("✅ System is healthy")
        st.json(health)
    else:
        st.error("❌ System health check failed")

if __name__ == "__main__":
    main()
