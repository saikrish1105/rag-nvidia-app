import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
import img2pdf
from datetime import datetime
from pymongo import MongoClient
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

nim_api_key = os.getenv("NVIDIA_API_KEY") 

def convert_to_pdf(img_path):
    if img_path.lower().endswith('.pdf'):
        return img_path
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    pdf_path = f"Documents\\{base_name}.pdf"

    image = Image.open(img_path)
    pdf_bytes = img2pdf.convert(image.filename)

    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    image.close()
    return pdf_path 

def load_chunk_documents(file_path):
    pdf_path = convert_to_pdf(file_path)
    loader = PyMuPDF4LLMLoader(
        pdf_path,
        extract_images=True,
        images_parser=RapidOCRBlobParser(),
    )
    documents = list(loader.lazy_load())  
    
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    result = char_splitter.split_documents(documents)
    return result

def VectorDB(docs, file_path):
    embeddings = NVIDIAEmbeddings(
        model="NV-Embed-QA",
        nvidia_api_key=nim_api_key
    )
    doc_name = os.path.splitext(os.path.basename(file_path))[0]
    collection_name = f"nvidia_{doc_name}"

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory='chroma_db3',
        collection_name=collection_name
    )
    return vector_store, collection_name

def load_vector_store_for_doc(file_path):
    doc_name = os.path.splitext(os.path.basename(file_path))[0]
    collection_name = f"nvidia_{doc_name}"
    
    embeddings = NVIDIAEmbeddings(
        model="NV-Embed-QA",
        nvidia_api_key=nim_api_key
    )
    
    return Chroma(
        persist_directory='chroma_db3',
        collection_name=collection_name,
        embedding_function=embeddings
    )

class MongoLogger:
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/") 
        try:
            self.client.admin.command('ping')
            st.sidebar.success("✅ Connected to MongoDB")
        except Exception as e:
            st.sidebar.error(f"❌ MongoDB connection failed: {e}")
            raise
        
        self.db = self.client["rag_logs"]
        self.queries = self.db["queries"]
        
    def log(self, query, context, answer, doc_name):
        record = {
            "timestamp": datetime.utcnow(),
            "document": doc_name,
            "query": query,
            "context": context,
            "answer": answer,
            "model": "mixtral_8x7b"
        }
        self.queries.insert_one(record)
        return record

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def query_document(file_path, question, logger):
    vector_store = load_vector_store_for_doc(file_path)
    
    retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={"k": 5, "lambda_mult": 0.5}
    )

    llm = ChatNVIDIA(
        model="mistralai/mixtral-8x7b-instruct-v0.1",
        temperature=1.3,
        api_key=nim_api_key
    )

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context:
        {context}

        Using this context, answer the question appropriately.

        Question:
        {question}
        """,
        input_variables=['context', 'question']
    )

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    final_chain = parallel_chain | prompt | llm | parser

    answer = final_chain.invoke(question)
    context = format_docs(retriever.invoke(question))
    
    doc_name = os.path.basename(file_path)
    logger.log(question, context, answer, doc_name)

    return answer, context
