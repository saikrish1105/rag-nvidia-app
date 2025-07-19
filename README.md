# NVIDIA LangChain RAG App

A Streamlit-based Retrieval-Augmented Generation (RAG) application powered by NVIDIA Mixtral-8x7B and LangChain.

## Features
- 📄 Document upload (PDF/images)
- 🖼️ OCR & image-to-text conversion
- 🔍 Semantic search with ChromaDB
- 📊 MongoDB logging
- 💬 Context-aware LLM responses

## 🔧 Setup

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Set up your `.env` file**:
    ```bash
    cp .env.example .env
    # Add your NVIDIA_API_KEY to .env
    ```

3. **Create MongoDB container (one-time)**:
    ```bash
    docker run -d --name rag-mongo -p 27018:27017 mongo
    ```

4. **Start MongoDB container (after creation)**:
    ```bash
    docker start rag-mongo
    ```

5. **Run the app**:
    ```bash
    streamlit run app.py
    ```

## 📂 Documents
Upload your file (PDF or image) using the app. It will be saved in the `Documents` directory automatically.

## 🧠 Prompt Example
> What are all the different ways I can apply for the minority quota and what are the certificates needed?

## 📝 License