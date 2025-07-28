import os
import json
import re
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter


INDEX_DIR = "faiss_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

json_file_paths = [
    "indiankanoon.json",
    "itat.json",
    "taxmann_output_fixed.json"
]

query = "What does the website tell about Article 19 (1)?"


print("Loading embedding model and LLM...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "cpu"}
)
llm = Ollama(model="phi3:mini")


per_file_responses = []
meta_data = []


for file_path in json_file_paths:
    filename = os.path.basename(file_path).replace(".json", "")
    index_path = os.path.join(INDEX_DIR, filename)
    print(f"\nProcessing {filename}...")

    try:
        if os.path.exists(index_path):
            print(f"Loading existing FAISS index for {filename}")
            db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        else:
            print(f"Creating FAISS index for {filename}")
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
            print(f"Saved FAISS index for {filename}")

        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = (
            "You are an AI trained assistant to extract and summarize legal and tax document data. Use only the context below.\n"
            "If multiple relevant pieces of text are found, summarize them in bullet points.\n"
            "If context is ambiguous or insufficient, state clearly: 'The answer is not available in the given context.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer (prefer bullet points or numbered list if multiple clauses apply):"
        )

        print(f"Querying LLM for {filename}...")
        response = llm.invoke(prompt).strip()
        print("Done.")

        per_file_responses.append({
            "filename": filename,
            "response": response
        })

        for doc in docs:
            meta_data.append({
                "filename": doc.metadata.get("title", filename),
                "page_num": doc.metadata.get("page", 0),
                "url": doc.metadata.get("url", "")
            })

    except Exception as e:
        print(f"Skipped {filename}: {e}")


print("\nGenerating final summary...")
all_answer_text = "\n\n".join([f"{item['filename']}:\n{item['response']}" for item in per_file_responses])
summary_prompt = (
    "You are a AI legal assistant. Given multiple answers from different legal and tax documents, "
    "summarize the combined insights clearly and concisely in bullet points only. Avoid repetition. "
    "If some documents don’t address the question, mention that in a bullet too.\n\n"
    f"{all_answer_text}"
)
summary = llm.invoke(summary_prompt).strip()


response_string = "\n\n".join([f"**{item['filename']}**\n{item['response']}" for item in per_file_responses])
response_string += "\n\n**Summary**\n" + summary

bold_words = list(set(re.findall(r"\*\*(.*?)\*\*", response_string)))


output = {
    "bold_words": bold_words,
    "meta_data": meta_data,
    "response": response_string,
    "table_data": [],
    "ucid": "99_18"
}


print(output)
