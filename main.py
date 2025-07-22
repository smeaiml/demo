import streamlit as st
import json
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title=" JSON QA", layout="wide")
st.title("Testing Question Answering ")

# File uploader
uploaded_file = st.file_uploader("📄 Upload your JSON file", type="json")

if uploaded_file is not None:
    st.success("File uploaded successfully")
    data = json.load(uploaded_file)

    # Extract documents
    documents = []
    for entry in data:
        content = entry.get("content", "").strip()
        if content:
            metadata = {
                "title": entry.get("title", "Untitled"),
                "url": entry.get("url", "")
            }
            documents.append(Document(page_content=content, metadata=metadata))

    # Split documents into chunks
    st.info("🔧 Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)

    # Embedding
    st.info("🧠 Generating embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # FAISS index (reuse if exists)
    index_path = "faiss_index"
    if os.path.exists(index_path):
        db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(split_docs, embedding_model)
        db.save_local(index_path)

    # LLM setup
    llm = Ollama(model="mistral:7b-instruct-q4_0")

    # RAG pipeline setup
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )

    # Question input
    query = st.text_input("💬 Ask a question based on the uploaded JSON:")

    if query:
        st.info("🧠 Thinking...")

        # Instruction prompt
        instruction = (
            "You are an assistant that only answers questions based on the uploaded JSON data. "
            "Do not use outside knowledge or make assumptions. If the answer is not found, reply with 'I don't know.'\n\n"
            f"Question: {query}"
        )

        result = qa_chain(instruction)

        st.subheader("✅ Answer")
        st.write(result["result"])

        with st.expander("📜 Prompt sent to the model"):
            st.code(instruction)

        with st.expander("📚 Source Documents"):
            for doc in result["source_documents"]:
                st.markdown(f"**{doc.metadata.get('title', 'No Title')}**\n[{doc.metadata.get('url', 'No URL')}]\n\n---")
