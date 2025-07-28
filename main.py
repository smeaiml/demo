<<<<<<< HEAD
import os
import json
import time
=======
import streamlit as st
import json
import os
>>>>>>> 1153e09ddd2a304ef0eb398104519895b632fa2d
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.llms import Ollama
<<<<<<< HEAD
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# === CONFIG ===
INDEX_DIR = "faiss_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

# === Input ===
json_file_paths = [
    "indiankanoon.json",
    "itat.json",
    "taxmann_output_fixed.json"
]
query = "What does the website tell about Article 19 (1) ?."

# === Init models ===
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "cpu"}
)
llm = Ollama(model="phi3:mini")

# === Process ===
all_answers = []
meta_data = []

for file_path in json_file_paths:
    try:
        filename = os.path.basename(file_path).replace(".json", "")
        index_path = os.path.join(INDEX_DIR, filename)

        if os.path.exists(index_path):
            db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents = []
            for entry in data:
                content = entry.get("content", "").strip()
                if content:
                    metadata = {
                        "title": entry.get("title", filename),
                        "url": entry.get("url", "")
                    }
                    documents.append(Document(page_content=content, metadata=metadata))

            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
            split_docs = splitter.split_documents(documents)
            db = FAISS.from_documents(split_docs, embedding_model)
            db.save_local(index_path)

        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = (
            "You are an AI trained helpful assisstant to extract and summarize legal and tax document data. Use only the context below.\n"
            "If multiple relevant pieces of text are found, summarize them in bullet points.\n"
            "If context is ambiguous or insufficient, state clearly: 'The answer is not available in the given context.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer (prefer bullet points or numbered list if multiple clauses apply):"
        )

        start_time = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - start_time

        # Metadata from docs
        for doc in docs:
            meta_data.append({
                "filename": doc.metadata.get("title", filename),
                "page_num": doc.metadata.get("page", 0),
                "url": doc.metadata.get("url", "")
            })

        all_answers.append(response.strip())

    except Exception as e:
        print(f"Skipped {file_path}: {str(e)}")

# === Final Summary ===
summary_prompt = (
    "You are a legal assistant AI. Given multiple answers from different legal and tax documents, "
    "summarize the combined insights clearly and concisely in bullet points only. Avoid repetition. "
    "If some documents don’t address the question, mention that in a bullet too.\n\n"
    + "\n\n".join(all_answers)
)

final_response = llm.invoke(summary_prompt).strip()

# === Extract bold/important phrases ===
bold_words = list(set(re.findall(r"\*\*(.*?)\*\*", final_response)))

# === Output Format ===
output = {
    "bold_words": bold_words,
    "meta_data": meta_data,
    "response": final_response,
    "table_data": [],
    "ucid": "99_18"
}

# === Print Final Output ===
print(json.dumps(output, indent=4))
=======
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
>>>>>>> 1153e09ddd2a304ef0eb398104519895b632fa2d
