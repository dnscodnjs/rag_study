import streamlit as st
from openai import OpenAI
from llm_util import *
import base64
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import sys

# This retrieves all command line arguments as a list
arguments = sys.argv
if len(sys.argv) != 2:
    print("Please specify the llm to use as the first argument")
    st.stop()
else:
    profile = sys.argv[1]

st.title("Chat with Image Support")

if "chat" not in st.session_state:
    client = open_llm(profile)
    st.session_state.chat = client  # Assume this returns a ChatOpenAI instance

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a form for user input
with st.form("chat_form"):
    prompt = st.text_input("Enter your message:")
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
    submit_button = st.form_submit_button("Send")

if submit_button:
    # Build the user message content
    message_content = []
    if prompt:
        message_content.append({"type": "text", "text": prompt})
    if uploaded_file is not None:
        # Read the image data and encode it in base64
        image_bytes = uploaded_file.read()
        image_type = uploaded_file.type  # e.g., 'image/jpeg'
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        # Include the image data in the message content
        message_content.append(
            {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_data}"}}
        )
    # Create the HumanMessage
    message = HumanMessage(content=message_content)
    # Append user's message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        if prompt:
            st.markdown(prompt)
        if uploaded_file is not None:
            st.image(uploaded_file)
    # Get response from the LLM
    response = st.session_state.chat.invoke([message])
    # Append assistant's response to messages
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.markdown(response.content)