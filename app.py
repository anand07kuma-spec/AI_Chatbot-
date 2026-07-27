import streamlit as st
from chatbot import get_response

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat Input
prompt = st.chat_input("Ask me anything...")

if "quick_prompt" in st.session_state:
    prompt = st.session_state.quick_prompt
    del st.session_state.quick_prompt    


# Sidebar
with st.sidebar:
    st.title("🤖 AI Chatbot")
    st.write("Built with Python + Streamlit + Gemini")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.info("Type your question in the chat box below.")

    st.markdown("---")
    conversation_count = len(st.session_state.messages) // 2

    if prompt:
        conversation_count += 1

    st.metric("💬 Conversations", conversation_count)


    st.markdown("---")

    chat_text = ""

    if "messages" in st.session_state:
        for msg in st.session_state.messages:
           role = "You" if msg["role"] == "user" else "AI"
           chat_text += f"{role}: {msg['content']}\n\n"

    st.download_button(
        label="📥 Download Chat",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
   )

# Main Title
st.title("🤖 AI Chatbot")
st.markdown(
    "### Your Personal AI Assistant powered by Google Gemini 🚀"
)


# Show Previous Chats
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


if len(st.session_state.messages) == 0:
# Welcome Screen    
    st.markdown("## 👋 Welcome Anand!")
    st.write("### Your Personal AI Assistant is ready.")

    st.info("💡 Try asking one of these questions:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🐍 Explain Python"):
            st.session_state.quick_prompt = "Explain Python for beginners."

        if st.button("🤖 What is AI?"):
            st.session_state.quick_prompt = "What is Artificial Intelligence?"

    with col2:
        if st.button("💻 Write HTML"):
            st.session_state.quick_prompt = "Write a simple HTML webpage."

        if st.button("😂 Tell me a Joke"):
            st.session_state.quick_prompt = "Tell me a funny joke."


if prompt:
    # User Message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # AI Reply
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤖 Thinking..."):
            reply = get_response(prompt)
            st.markdown(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

st.markdown("---")
st.caption("Made by Anand Kumar ❤️ | Powered by Python + Streamlit + Gemini")