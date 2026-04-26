
import streamlit as st
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agent import run_agent

st.set_page_config(page_title="GlassBox Agent", page_icon="🔬")
st.title("🔬 GlassBox AutoML Agent")
st.caption("Upload a CSV. Ask in plain English. Get a trained model.")

# Save uploaded file permanently in the project folder
csv_path = None
uploaded = st.file_uploader("Upload your dataset (CSV)", type="csv")
if uploaded:
    save_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, uploaded.name)
    with open(csv_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Loaded: {uploaded.name}")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
if prompt := st.chat_input("e.g. Build a model to predict whether a loan gets approved"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Running AutoML pipeline..."):
        reply = run_agent(prompt, csv_path=csv_path)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)