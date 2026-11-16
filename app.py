import streamlit as st
from rag_engine import extract_and_chunk_pdf, build_vector_index, answer_question

st.set_page_config(page_title="Local Doc Q&A (RAG)", page_icon="📄", layout="centered")
st.title("📄 Document Q&A Chatbot")
st.caption("Upload a PDF and ask questions grounded strictly in its content.")

with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file and st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Chunking and indexing PDF..."):
            chunks = extract_and_chunk_pdf(uploaded_file)
            if not chunks:
                st.error("No extractable text found (scanned image PDFs need OCR).")
            else:
                st.session_state["chunks"] = chunks
                st.session_state["index"] = build_vector_index(chunks)
                st.session_state["file_name"] = uploaded_file.name
                st.session_state["messages"] = []
                st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("View Retrieved Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**Page {s['page']}** (Similarity: `{s['score']:.2f}`): {s['text'][:250]}...")

if prompt := st.chat_input("Ask something about your PDF..."):
    if "index" not in st.session_state:
        st.warning("Please upload a PDF in the sidebar first.")
    else:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                answer, sources = answer_question(
                    prompt, st.session_state["index"], st.session_state["chunks"]
                )
                st.markdown(answer)
                with st.expander("View Retrieved Sources"):
                    for s in sources:
                        st.markdown(f"**Page {s['page']}** (Similarity: `{s['score']:.2f}`): {s['text'][:250]}...")

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )