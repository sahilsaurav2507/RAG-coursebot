import os
import json
from flask import Flask, request, jsonify
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document

app = Flask(__name__)

# Keep existing initialization code
os.environ["GROQ_API_KEY"] = "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"

# Initialize components
loader = WebBaseLoader("https://brainlox.com/courses/category/technical")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

texts = [doc.page_content for doc in docs]
metadata = [{"source": doc.metadata.get("source", "unknown")} for doc in docs]
vector_store.add_texts(texts, metadatas=metadata)

llm = ChatGroq(model_name="gemma2-9b-it")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    query_embedding = embedding_model.embed_query(query)
    results = vector_store.similarity_search_by_vector(query_embedding, k=5)
    
    context = "\n".join(doc.page_content for doc in results)
    prompt = f"Context: {context}\n\nUser Query: {query}\nResponse: "
    
    response = llm.invoke(prompt)
    return jsonify({"response": response.content})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
