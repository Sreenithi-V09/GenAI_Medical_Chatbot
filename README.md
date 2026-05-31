# Medical Chatbot using RAG

## Overview

The Medical Chatbot is an AI-powered healthcare assistant that provides responses to medical queries using Retrieval-Augmented Generation (RAG). The chatbot retrieves relevant information from medical PDF documents and generates context-aware responses using a biomedical Large Language Model.

## Features

- Medical question answering system
- Retrieval-Augmented Generation (RAG)
- PDF document processing
- Semantic search using embeddings
- Chroma vector database integration
- Biomedical Llama model support
- Responsive web-based chatbot interface
- Flask backend API

## Technologies Used

- Python
- Flask
- LangChain
- ChromaDB
- Hugging Face Embeddings
- LlamaCpp
- HTML
- CSS
- JavaScript

## Project Structure

MedicalChatbot/

├── app.py

├── templates/

│ └── index.html

├── static/

│ └── style.css

├── data/

│ └── Medical_PDFs

├── model/

│ └── Bio-Medical-Llama-3.1-8B.gguf

└── README.md

## Installation

### Clone Repository

git clone <repository-url>

cd MedicalChatbot

### Install Dependencies

pip install flask

pip install langchain

pip install langchain-community

pip install langchain-core

pip install langchain-huggingface

pip install langchain-text-splitters

pip install sentence-transformers

pip install chromadb

pip install pypdf

pip install llama-cpp-python

## Usage

1. Place medical PDF documents inside the data folder.

2. Place the GGUF model file inside the model folder.

3. Run the Flask application:

py app.py

4. Open the browser and visit:

http://127.0.0.1:5000

5. Enter a medical query and receive AI-generated responses.

## Working Process

1. Load PDF documents.
2. Split documents into text chunks.
3. Generate embeddings using PubMedBERT.
4. Store embeddings in ChromaDB.
5. Retrieve relevant chunks based on user queries.
6. Generate responses using the biomedical Llama model.
7. Display responses through the web interface.

## Future Enhancements

- Voice-based interaction
- Multi-language support
- User authentication
- Medical report analysis
- Cloud deployment
- Real-time symptom checker

## Author

V. Sreenithi

B.Tech Computer Science and Business Systems

K. Ramakrishnan College of Engineering

## License

This project is developed for educational and research purposes.
