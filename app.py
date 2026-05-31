from flask import Flask, render_template, request, jsonify

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# =========================
# Flask App
# =========================

app = Flask(__name__)

# =========================
# Load PDF Documents
# =========================

loader = PyPDFDirectoryLoader("data/")
docs = loader.load()

# =========================
# Split Documents
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)

chunks = text_splitter.split_documents(docs)

# =========================
# Embeddings
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)

# =========================
# Vector Database
# =========================

vectorstore = Chroma.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# =========================
# Load LLM Model
# =========================

llm = LlamaCpp(
    model_path="model/Bio-Medical-Llama-3.1-8B.i1-IQ1_M.gguf",
    temperature=0.7,
    max_tokens=512,
    top_p=0.9,
    n_ctx=2048,
    verbose=True,
)

# =========================
# Prompt Template
# =========================

template = """
<|context|>
You are a Medical Assistant.
Answer the question accurately based on the context.
</s>

<|User|>
{query}
</s>

<|Assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)

# =========================
# RAG Chain
# =========================

rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# =========================
# Routes
# =========================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot():
    user_input = request.json.get("message")

    if not user_input:
        return jsonify({"response": "Please enter a message."})

    result = rag_chain.invoke(user_input)

    return jsonify({"response": result})

# =========================
# Run App
# =========================

if __name__ == "__main__":
    app.run(debug=True)