# NVIDIA LangChain RAG App

This project is a Retrieval-Augmented Generation (RAG) application using:
- LangChain
- NVIDIA's Mixtral-8x7B model
- MongoDB logging
- Chroma vector database
- OCR/image-to-text PDF conversion

## 💾 Features
- Drag & drop document support (PDF or images)
- Converts image to PDF if needed
- Extracts and chunks text
- Stores vectors using NVIDIA embeddings + Chroma
- Logs query, context, and answer to MongoDB
- LLM answers based strictly on context

## 🔧 Setup

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Set up your `.env` file**:
    ```bash
    cp .env.example .env
    ```

3. **Start MongoDB (Docker example)**:
    ```bash
    docker run -d --name rag-mongo -p 27017:27017 mongo
    ```

4. **Run the app**:
    ```bash
    python main.py
    ```

## 📂 Documents
Place your input documents (images or PDFs) inside the `Documents/` folder.

## 🧠 Prompt Example
> What are all the different ways I can apply for the minority quota and what are the certificates needed?

## 📝 License
MIT
