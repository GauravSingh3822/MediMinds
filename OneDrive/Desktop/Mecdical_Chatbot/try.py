import streamlit as st
import datetime
import requests

st.set_page_config(page_title="Medical Chatbot", layout="centered")

st.markdown(
    """
    <style>
    body, html {
        height: 100%;
        margin: 0;
        background: linear-gradient(to right, rgb(38, 51, 61), rgb(50, 55, 65), rgb(33, 33, 78));
        color: white;
        text-align: center;
    }
    .chat-container {
        max-width: 600px;
        margin: auto;
        padding: 20px;
        background-color: rgba(0,0,0,0.4);
        border-radius: 15px;
    }
    .user-message {
        background-color: #58cc71;
        padding: 10px;
        border-radius: 15px;
        margin: 5px;
        text-align: right;
    }
    .bot-message {
        background-color: rgb(82, 172, 255);
        padding: 10px;
        border-radius: 15px;
        margin: 5px;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h2>Medical Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p>Ask me anything!</p>", unsafe_allow_html=True)

chat_history = st.session_state.get("chat_history", [])

for message in chat_history:
    role, text, timestamp = message
    css_class = "user-message" if role == "user" else "bot-message"
    st.markdown(f'<div class="{css_class}">{text} <span style="font-size:10px; color:gray;">{timestamp}</span></div>', unsafe_allow_html=True)

msg = st.text_input("Type your message...")

if st.button("Send") and msg:
    timestamp = datetime.datetime.now().strftime("%H:%M")
    chat_history.append(("user", msg, timestamp))
    st.session_state.chat_history = chat_history
    
    response = requests.post("http://localhost:5000/get", data={"msg": msg}).text
    chat_history.append(("bot", response, timestamp))
    st.session_state.chat_history = chat_history
    st.experimental_rerun()

