import streamlit as st
from rag_app import chat_with_video, summarize_video

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="YouTube AI Assistant",
    page_icon="",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* General App Styling */
body {
    background-color: #0f172a;
}

/* Title */
.title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 10px;
}

/* Chat container */
.chat-container {
    height: 60vh;
    overflow-y: auto;
    padding: 10px;
    border-radius: 10px;
    background-color: #111827;
}

/* Sticky input */
.stChatInput {
    position: fixed;
    bottom: 20px;
    left: 25%;
    width: 50%;
}

/* Buttons */
.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 45px;
    font-weight: 600;
}

/* Summary box */
.summary-box {
    background-color: #111827;
    padding: 15px;
    border-radius: 10px;
    min-height: 200px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">YouTube AI Assistant Note: Only use video links that are 5–7 minutes long; otherwise, the API quota may get exhausted.</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
video_url = st.text_input("🔗 Enter YouTube URL")

if video_url:

    st.video(video_url)

    col1, col2 = st.columns([1, 1.5])

    # ---------------- SUMMARY ----------------
    with col1:
        st.subheader("📄 Video Summary")

        if st.button("✨ Generate Summary"):
            with st.spinner("Generating summary..."):
                summary = summarize_video(video_url)
                st.session_state.summary = summary

        # Persist summary
        if "summary" in st.session_state:
            st.markdown(f'<div class="summary-box">{st.session_state.summary}</div>', unsafe_allow_html=True)

    # ---------------- CHAT ----------------
    with col2:
        st.subheader("💬 Chat with Video")

        # Session state init
        if "last_video" not in st.session_state:
            st.session_state.last_video = ""

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Reset on new video
        if video_url != st.session_state.last_video:
            st.session_state.messages = []
            st.session_state.last_video = video_url

        # Chat container
        chat_container = st.container()

        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat input (always bottom)
        query = st.chat_input("Ask anything about this video...")

        if query:
            # Store user msg
            st.session_state.messages.append({"role": "user", "content": query})

            # Display user msg
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(query)

            # Get response
            with st.spinner("Thinking..."):
                answer = chat_with_video(video_url, query)

            # Display assistant msg
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(answer)

            # Store response
            st.session_state.messages.append({"role": "assistant", "content": answer})
