# RAG-coursebot
### Course Information Chatbot

A RAG-based chatbot that provides information about courses using LangChain, Flask API, and Groq.

## Features

- Real-time course information retrieval from brainlox.com
- Vector similarity search using ChromaDB
- Advanced embedding using HuggingFace's sentence-transformers
- LLM integration with Groq's Gemma 2-9b model
- REST API endpoints using Flask

## Tech Stack

- **Framework**: Flask
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (sentence-transformers/all-mpnet-base-v2)
- **LLM**: Groq (gemma2-9b-it)
- **Document Processing**: LangChain

## API Endpoints

### POST /chat
Processes user queries about courses
- Input: JSON with "query" field
- Output: AI-generated response based on course context

### GET /health
Health check endpoint
- Output: System health status

Test Health check upt 
Output :> { status : healthy }

## Setup

1. Set your Groq API key in environment variables
2. Install required dependencies
3. Run the Flask application

## How it Works

1. Loads course data from brainlox.com
2. Splits content into manageable chunks
3. Creates and stores embeddings in ChromaDB
4. Processes user queries through:
   - Embedding generation
   - Similarity search
   - Context-aware response generation using Groq

## Test input and the output
Test Query =  "Tell me about courses on AI."
Test output ===
![image](https://github.com/user-attachments/assets/fa218846-ce26-46dc-8617-b2652a668311)



```bash
GROQ_API_KEY=your_groq_api_key

