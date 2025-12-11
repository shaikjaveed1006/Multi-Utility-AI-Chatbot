import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from datetime import datetime

from final_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    save_thread_title,
    get_thread_title,
    generate_conversation_title,
    get_thread_message_count,
)

# =========================== Page Config ===========================
st.set_page_config(
    page_title="Multi Utility Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================== Custom CSS ===========================
st.markdown("""
<style>
    /* Main chat container */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    /* Thread button styling */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(100, 150, 255, 0.5) !important;
        box-shadow: 0 2px 8px rgba(100, 150, 255, 0.3) !important;
    }
    
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* PDF upload section */
    .pdf-info {
        background: rgba(52, 211, 153, 0.15);
        border-left: 4px solid #34d399;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
        color: #d1fae5 !important;
    }
    
    [data-testid="stSidebar"] .stAlert {
        background: rgba(59, 130, 246, 0.15) !important;
        color: #bfdbfe !important;
    }
    
    [data-testid="stSidebar"] code {
        background: rgba(0, 0, 0, 0.3) !important;
        color: #93c5fd !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* New chat button */
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stSidebar"] .stExpander {
        background: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Thread title in sidebar */
    .thread-title {
        font-size: 14px;
        color: #495057;
        margin-bottom: 2px;
    }
    
    .thread-id {
        font-size: 11px;
        color: #6c757d;
        font-family: monospace;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .status-success {
        background: #d1e7dd;
        color: #0f5132;
    }
    
    .status-info {
        background: #cfe2ff;
        color: #084298;
    }
</style>
""", unsafe_allow_html=True)


# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["conversation_started"] = False
    st.session_state["initialized"] = False


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def get_or_create_title(thread_id: str, first_message: str = None) -> str:
    """Get existing title or create a new one if needed."""
    title = get_thread_title(thread_id)
    if not title and first_message:
        title = generate_conversation_title(first_message)
        save_thread_title(thread_id, title)
    return title or "New Conversation"


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "conversation_started" not in st.session_state:
    st.session_state["conversation_started"] = False

if "initialized" not in st.session_state:
    # On first load, check if current thread has messages
    thread_key = str(st.session_state["thread_id"])
    existing_messages = load_conversation(thread_key)
    if existing_messages:
        temp_messages = []
        for msg in existing_messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                if msg.content:
                    temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages
        st.session_state["conversation_started"] = len(temp_messages) > 0
    st.session_state["initialized"] = True

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
with st.sidebar:
    st.markdown("### 🤖 Multi Utility Chatbot")
    
    # Current conversation info
    current_title = get_or_create_title(thread_key)
    st.markdown(f"**Current:** {current_title}")
    
    with st.expander("📋 Thread Details", expanded=False):
        st.code(thread_key, language=None)
    
    # New chat button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()
    
    st.divider()
    
    # PDF Upload Section
    st.markdown("### 📄 Document Upload")
    
    if thread_docs:
        latest_doc = list(thread_docs.values())[-1]
        st.markdown(f"""
        <div class="pdf-info">
            <strong>✅ {latest_doc.get('filename')}</strong><br>
            <small>{latest_doc.get('chunks')} chunks • {latest_doc.get('documents')} pages</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📤 No PDF uploaded yet")
    
    uploaded_pdf = st.file_uploader(
        "Upload PDF", 
        type=["pdf"],
        label_visibility="collapsed"
    )
    
    if uploaded_pdf:
        if uploaded_pdf.name in thread_docs:
            st.success(f"✓ Already indexed: `{uploaded_pdf.name}`")
        else:
            with st.status("🔄 Processing PDF...", expanded=True) as status_box:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(
                    label="✅ PDF indexed successfully", 
                    state="complete", 
                    expanded=False
                )
                st.rerun()
    
    st.divider()
    
    # Past Conversations
    st.markdown("### 💬 Conversations")
    
    if not threads:
        st.markdown("*No conversations yet*")
    else:
        for idx, thread_id in enumerate(threads):
            thread_title = get_or_create_title(str(thread_id))
            is_current = str(thread_id) == thread_key
            msg_count = get_thread_message_count(str(thread_id))
            
            # Create a container for each thread
            col1, col2 = st.columns([4, 1])
            
            with col1:
                button_type = "primary" if is_current else "secondary"
                button_label = f"{'📌 ' if is_current else '💭 '}{thread_title}"
                if msg_count > 0:
                    button_label += f" ({msg_count})"
                
                if st.button(
                    button_label,
                    key=f"thread-{thread_id}",
                    use_container_width=True,
                    disabled=is_current,
                    type=button_type if is_current else "secondary"
                ):
                    selected_thread = thread_id
            
            with col2:
                # Show document indicator if thread has docs
                doc_meta = thread_document_metadata(str(thread_id))
                if doc_meta:
                    st.markdown("📄", help=f"Has document: {doc_meta.get('filename')}")

# ============================ Main Layout ========================
st.title("🤖 Multi Utility Chatbot")

# Feature badges
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<span class="status-badge status-info">📄 PDF Analysis</span>', unsafe_allow_html=True)
with col2:
    st.markdown('<span class="status-badge status-info">🔍 Web Search</span>', unsafe_allow_html=True)
with col3:
    st.markdown('<span class="status-badge status-info">🧮 Calculator</span>', unsafe_allow_html=True)
with col4:
    st.markdown('<span class="status-badge status-info">💾 Persistent Memory</span>', unsafe_allow_html=True)

st.markdown("---")

# Show loaded message count if conversation exists
if st.session_state["message_history"]:
    st.caption(f"💾 {len(st.session_state['message_history'])} messages loaded from memory")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("💬 Ask about your document or use tools...")

if user_input:
    # Generate title for first message
    if not st.session_state["conversation_started"]:
        try:
            title = generate_conversation_title(user_input)
            save_thread_title(thread_key, title)
        except Exception as e:
            print(f"Error generating title: {e}")
            # Use a simple fallback
            title = " ".join(user_input.split()[:4]) + "..."
            save_thread_title(thread_key, title)
        st.session_state["conversation_started"] = True
    
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_holder = {"box": None}
        full_response = ""

        try:
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Handle tool messages
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    tool_icon = {
                        "rag_tool": "📄",
                        "calculator": "🧮",
                        "duckduckgo_search": "🔍",
                    }.get(tool_name, "🔧")
                    
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"{tool_icon} Using {tool_name}...", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"{tool_icon} Using {tool_name}...",
                            state="running",
                            expanded=True,
                        )

                # Stream assistant tokens
                if isinstance(message_chunk, AIMessage):
                    if message_chunk.content:
                        full_response += message_chunk.content
                        message_placeholder.markdown(full_response + "▌")

            # Final update without cursor
            message_placeholder.markdown(full_response)

            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool execution complete", 
                    state="complete", 
                    expanded=False
                )
        except Exception as e:
            error_msg = f"⚠️ An error occurred: {str(e)}"
            message_placeholder.error(error_msg)
            full_response = "I apologize, but I encountered an error. Please try again."

    st.session_state["message_history"].append(
        {"role": "assistant", "content": full_response}
    )

    # Show document info at bottom if available
    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📄 Using: **{doc_meta.get('filename')}** "
            f"({doc_meta.get('chunks')} chunks, {doc_meta.get('documents')} pages)"
        )

# Handle thread switching
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    st.session_state["initialized"] = False
    
    # Load conversation from database
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            if msg.content:  # Only add messages with content
                temp_messages.append({"role": role, "content": msg.content})
    
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.session_state["conversation_started"] = len(temp_messages) > 0
    st.rerun()