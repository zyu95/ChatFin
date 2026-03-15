
# config.py
import os
import streamlit as st  # <-- 导入 streamlit 库
from dotenv import load_dotenv

# --- 1. Load local .env (only for local development) ---
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, ".env"))

# --- 2. Smart Key Retrieval ---
# First, try to get from Streamlit Cloud Secrets
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    print("✅ Successfully loaded API Key from Streamlit Secrets")
else:
    # If not on cloud, fall back to environment variables (local)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    print("🏠 Successfully loaded API Key from Local Environment")