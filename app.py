
import streamlit as st
import json
from datetime import datetime
from chatbot import get_response

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------- SESSION STATE ----------
if "conversations" not in st.session_state:
    st.session_state.conversations = [
        {
            "id": 1,
            "title": "New Chat",
            "messages": []
        }
    ]
    st.session_state.active_chat_id = 1
    st.session_state.next_chat_id = 2

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


def get_active_chat():
    for chat in st.session_state.conversations:
        if chat["id"] == st.session_state.active_chat_id:
            return chat
    return st.session_state.conversations[0]


def create_new_chat():
    chat_id = st.session_state.next_chat_id
    st.session_state.next_chat_id += 1

    st.session_state.conversations.insert(
        0,
        {
            "id": chat_id,
            "title": "New Chat",
            "messages": []
        }
    )
    st.session_state.active_chat_id = chat_id


def download_chat(messages):
    lines = []
    for message in messages:
        role = "You" if message["role"] == "user" else "AI"
        lines.append(f'{role}:\n{message["content"]}\n')

    return "\n".join(lines)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("🤖 AI Assistant")
    st.caption("Powered by Google Gemini")

    if st.button("＋ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.divider()
    st.subheader("💬 Your Conversations")

    for chat in st.session_state.conversations:
        label = (
            f'● {chat["title"]}'
            if chat["id"] == st.session_state.active_chat_id
            else chat["title"]
        )

        if st.button(
            label,
            key=f'chat_{chat["id"]}',
            use_container_width=True
        ):
            st.session_state.active_chat_id = chat["id"]
            st.rerun()

    st.divider()

    active_chat = get_active_chat()

    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        active_chat["messages"] = []
        active_chat["title"] = "New Chat"
        st.rerun()

    if active_chat["messages"]:
        st.download_button(
            "⬇️ Download Current Chat",
            data=download_chat(active_chat["messages"]),
            file_name="ai_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.caption(
        f'Conversations: {len(st.session_state.conversations)}'
    )

    st.caption("Your chats are stored in this browser session.")


# ---------- MAIN UI ----------
st.title("🤖 AI Chatbot")
st.caption("Ask questions, learn new things, and explore ideas.")

active_chat = get_active_chat()
messages = active_chat["messages"]

# Welcome screen and quick prompts
if not messages:
    st.markdown("### 👋 Hello! How can I help you today?")
    st.write("Choose a prompt or type your own question below.")

    prompts = [
        "Explain Python in simple words",
        "What is Artificial Intelligence?",
        "Write a simple HTML page",
        "Give me a fun fact"
    ]

    columns = st.columns(2)

    for index, prompt_text in enumerate(prompts):
        with columns[index % 2]:
            if st.button(
                prompt_text,
                key=f"prompt_{active_chat['id']}_{index}",
                use_container_width=True
            ):
                st.session_state.pending_prompt = prompt_text
                st.rerun()

# Display conversation history
for index, message in enumerate(messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Retry button for failed AI responses
        if (
            message["role"] == "assistant"
            and message.get("is_error", False)
            and index == len(messages) - 1
        ):
            if st.button(
                "🔄 Retry",
                key=f'retry_{active_chat["id"]}_{index}'
            ):
                last_prompt = message.get("prompt", "")

                with st.spinner("Trying again..."):
                    try:
                        answer = get_response(last_prompt)

                        if not isinstance(answer, str) or not answer.strip():
                            raise ValueError("The AI returned an empty response.")

                        messages[index]["content"] = answer
                        messages[index]["is_error"] = False
                    except Exception:
                        messages[index]["content"] = (
                            "I couldn't get a response. Please try again "
                            "or check your API configuration."
                        )
                        messages[index]["is_error"] = True

                st.rerun()

# Chat input
typed_prompt = st.chat_input("Message your AI assistant...")

prompt = typed_prompt

if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt and prompt.strip():
    prompt = prompt.strip()

    # Add the user's message
    messages.append({
        "role": "user",
        "content": prompt
    })

    # Set conversation title from the first question
    if active_chat["title"] == "New Chat":
        active_chat["title"] = prompt[:35]
        if len(prompt) > 35:
            active_chat["title"] += "..."

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate the AI response
    with st.chat_message("assistant"):
        try:
            with st.spinner("🤔 Thinking..."):
                answer = get_response(prompt)

            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("The AI returned an empty response.")

            # Detect common error strings returned by the chatbot module
            error_words = (
                "error:",
                "api error",
                "failed to",
                "exception:"
            )
            is_error = answer.strip().lower().startswith(error_words)

            st.markdown(answer)

            messages.append({
                "role": "assistant",
                "content": answer,
                "is_error": is_error,
                "prompt": prompt
            })

        except Exception:
            error_message = (
                "Sorry, I couldn't get a response. Please check your "
                "internet connection and API configuration, then retry."
            )

            st.error(error_message)

            messages.append({
                "role": "assistant",
                "content": error_message,
                "is_error": True,
                "prompt": prompt
            })

    st.rerun()
