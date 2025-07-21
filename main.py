
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import json

# Load your JSON
with open("indiankanoon.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare documents
documents = []
for entry in data:
    content = entry.get("content", "").strip()
    if content:
        metadata = {"title": entry["title"], "url": entry["url"]}
        documents.append(Document(page_content=content, metadata=metadata))

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embedding_model)
db.save_local("faiss_index")

# Mistral via Ollama
llm = Ollama(model="mistral")

# RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Ask question
question = "What is the mission of the website?"
result = qa_chain(question)

print("\nQ:", question)
print("\nAnswer:", result['result'])

# Optional: print sources
for doc in result['source_documents']:
    print("Source:", doc.metadata['title'], "|", doc.metadata['url'])
