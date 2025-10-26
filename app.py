# import os
# import streamlit as st
# from dotenv import load_dotenv
# from rag import answer_query
# from storage.logger import JsonlLogger
# from pathlib import Path

# load_dotenv()

# LOGS_DIR = os.getenv("LOGS_DIR", "./storage/logs")
# DATA_DIR = Path(os.getenv("DATA_DIR", "data/pdfs"))
# DATA_DIR.mkdir(parents=True, exist_ok=True)

# logger = JsonlLogger(LOGS_DIR)

# st.set_page_config(page_title="RAG Chatbot (Open-Source)", page_icon="🤖", layout="wide")

# st.title("🤖 RAG Chatbot (Open-Source)")
# st.caption("Ask questions about your uploaded PDFs. Answers will cite sources.")

# with st.expander("Upload & Index PDFs", expanded=True):
#     up = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
#     saved = []
#     if up:
#         for file in up:
#             dest = DATA_DIR / file.name
#             with open(dest, "wb") as f:
#                 f.write(file.getbuffer())
#             saved.append(dest.name)
#         if saved:
#             st.success("Saved: " + ", ".join(saved))
#     if st.button("Index now"):
#         with st.spinner("Indexing PDFs (this can take a minute)..."):
#             from ingest import ingest_folder
#             n_pdfs, n_chunks = ingest_folder(str(DATA_DIR))
#         st.toast(f"Indexed {n_pdfs} PDFs → {n_chunks} chunks.")

# st.divider()

# if "history" not in st.session_state:
#     st.session_state.history = []

# st.info("Welcome! Ask a question about the uploaded PDFs. If I can’t find it, I’ll tell you.")

# query = st.chat_input("Type your question...")

# for role, msg, sources in st.session_state.history:
#     with st.chat_message(role):
#         st.markdown(msg)
#         if role == "assistant" and sources:
#             st.caption("Sources: " + ", ".join(sources))

# if query:
#     with st.chat_message("user"):
#         st.markdown(query)
#     logger.log("user_query", {"text": query})

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             ans, sources = answer_query(query)
#         if ans is None:
#             msg = "Sorry, I couldn't find anything related to your query."
#             st.markdown(msg)
#             st.session_state.history.append(("assistant", msg, []))
#             logger.log("assistant_answer", {"text": msg, "sources": []})
#         else:
#             st.markdown(ans)
#             if sources:
#                 st.caption("Sources: " + ", ".join(sources))
#             st.session_state.history.append(("assistant", ans, sources))
#             logger.log("assistant_answer", {"text": ans, "sources": sources})




import os
import streamlit as st
from dotenv import load_dotenv
from rag import answer_query
from storage.logger import JsonlLogger
from pathlib import Path

load_dotenv()

LOGS_DIR = os.getenv("LOGS_DIR", "./storage/logs")
DATA_DIR = Path(os.getenv("DATA_DIR", "data/pdfs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger = JsonlLogger(LOGS_DIR)

st.set_page_config(page_title="RAG Chatbot (Open-Source)", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot (Open-Source)")
st.caption("Ask questions about your uploaded PDFs. Answers will cite sources.")

# --- Upload & Index panel
with st.expander("Upload & Index PDFs", expanded=True):
    up = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    saved = []
    if up:
        for file in up:
            dest = DATA_DIR / file.name
            with open(dest, "wb") as f:
                f.write(file.getbuffer())
            saved.append(dest.name)
        if saved:
            st.success("Saved: " + ", ".join(saved))
    if st.button("Index now"):
        with st.spinner("Indexing PDFs (this can take a minute)..."):
            from ingest import ingest_folder
            n_pdfs, n_chunks = ingest_folder(str(DATA_DIR))
        st.toast(f"Indexed {n_pdfs} PDFs → {n_chunks} chunks.")

st.divider()

# --- Session memory (used both for display & prompting)
if "chat_turns" not in st.session_state:
    # list[dict]: {"role":"user"|"assistant", "content": str}
    st.session_state.chat_turns = []

# --- Replay chat
for t in st.session_state.chat_turns:
    with st.chat_message("user" if t["role"] == "user" else "assistant"):
        st.markdown(t["content"])

st.info("Welcome! Ask a question about the uploaded PDFs. If I can’t find it, I’ll tell you.")
query = st.chat_input("Type your question...")

if query:
    # Show user message and store to memory
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_turns.append({"role": "user", "content": query})
    logger.log("user_query", {"text": query})

    # Generate answer with memory-aware RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans, sources = answer_query(query, history=st.session_state.chat_turns)
        if ans is None:
            msg = "Sorry, I couldn't find anything related to your query."
            st.markdown(msg)
            st.session_state.chat_turns.append({"role": "assistant", "content": msg})
            logger.log("assistant_answer", {"text": msg, "sources": []})
        else:
            # append citations at the end for display
            display = ans
            if sources:
                display += "\n\n" + "_Sources: " + ", ".join(sources) + "_"
            st.markdown(display)
            st.session_state.chat_turns.append({"role": "assistant", "content": display})
            logger.log("assistant_answer", {"text": ans, "sources": sources})
