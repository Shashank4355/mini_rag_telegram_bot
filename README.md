**Mini RAG Telegram Bot (Retrieval-Augmented Generation)**
=====================================================================

**1\. Project Description**
---------------------------

This project implements a lightweight Retrieval-Augmented Generation (RAG) system integrated into a Telegram bot. The bot accepts user queries, retrieves relevant information from a collection of local documents, and generates concise answers using a locally running LLM through Ollama.

The system fulfills the specifications of **Option A — Mini-RAG**, including document storage, chunking, embedding, retrieval, prompt construction, and generation.

**2\. Features**
----------------

### **Bot Interface**

The Telegram bot supports the following commands:

*   /ask — Searches documents and returns an answer.
    
*   /help — Displays available commands.
    
*   /summarize — Shows the last three user questions.
    

### **Mini-RAG Pipeline**

*   3–5 Markdown documents stored locally.
    
*   Document chunking for accurate retrieval.
    
*   Embeddings created using all-MiniLM-L6-v2 (SentenceTransformers).
    
*   Embedding storage in SQLite (embeddings.db).
    
*   Top-k retrieval using cosine similarity (Nearest Neighbors).
    
*   Context construction and prompt creation.
    
*   Local LLM summarization using **Ollama model: phi3:mini**.
    

**3\. System Architecture**
---------------------------
Telegram User

│

▼

Telegram Bot (app.py)

│

▼

RAG Engine (rag.py)

│

├── Query Embedding (MiniLM)

├── SQLite Vector Database

├── Top-k Chunk Retrieval

├── Prompt Construction

└── LLM Summarization (Ollama: phi3:mini)

▼

Formatted Response Returned to Telegram

**4\. Project Structure**
-------------------------

mini\_rag\_telegram\_bot/

│

├── app.py # Telegram bot logic

├── rag.py # RAG pipeline implementation

├── index\_docs.py # Indexer for creating embeddings.db

├── requirements.txt

├── README.md

│

├── docs/ # Local knowledge documents (Markdown)

│ ├── doc1.md

│ ├── doc2.md

│ ├── doc3.md

│

├── screenshots/ # Demonstration screenshots (for submission)

│

└── embeddings.db # Auto-generated vector database


### **5.1. Virtual Environment Setup**

python -m venv venv

venv\\Scripts\\activate

### **5.2. Install Dependencies**

pip install -r requirements.txt

**6\. Installing and Configuring Ollama**
-----------------------------------------

Download Ollama for your OS:[https://ollama.com/download](https://ollama.com/download)

Pull the required LLM:

ollama pull phi3:mini

Verify installation:

ollama list

**7\. Document Indexing**
-------------------------

Place your documents inside the docs/ folder.

Run the indexer:

python index\_docs.py

This script:

*   Splits documents into chunks
    
*   Generates embeddings
    
*   Stores them in a SQLite database (embeddings.db)
    

**8\. Telegram Bot Configuration**
----------------------------------

Create a .env file in the project root:

TG\_TOKEN=your\_bot\_token\_here

OLLAMA\_MODEL=phi3:mini

The .env file should not be committed to GitHub.

**9\. Running the Bot**
-----------------------

Start the Telegram bot:

python app.py

Example usage in Telegram:

/ask What is the shipping policy?

The bot will retrieve relevant chunks, construct a prompt, summarize using the local LLM, and reply with a concise answer including sources.


**10\. Screenshots**

<img width="853" height="147" alt="image" src="https://github.com/user-attachments/assets/1fbdb31b-75a6-4139-ace5-71fba02c612a" />

<img width="861" height="110" alt="image" src="https://github.com/user-attachments/assets/904ac121-8b51-4cd2-b2bf-62415adeb243" />


--------------------
