from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pymongo import MongoClient
from datetime import datetime,timezone
import os
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import img2pdf
import time

load_dotenv()

nim_api_key = os.getenv("NVIDIA_API_KEY") 

# convert input files to pdf for loader to work
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

# load your documents and create multiple smaller chunks of data
def load_chunk_documents(file_path):
    pdf_path = convert_to_pdf(file_path)
    loader = PyMuPDF4LLMLoader(
        pdf_path,
        extract_images=True,
        images_parser=RapidOCRBlobParser(),
    )
    documents = list(loader.lazy_load()) # 
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    result = char_splitter.split_documents(documents)

    return result

# convert all these chunks to embeddings and storing it in a Vector Database
def VectorDB(docs):
    embeddings = NVIDIAEmbeddings(
        model="NV-Embed-QA",  # Choose from table above
        nvidia_api_key= nim_api_key
    )

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding = embeddings,
        persist_directory='chroma_db3',
        collection_name='nvidia_embeddings'
    )
    return vector_store

# create a MongoDB logger class to store your query answer and context
class MongoLogger:
    def __init__(self):
        # self.client = MongoClient(
        #     "mongodb://admin:password@localhost:27017/",  # Added credentials
        #     authSource="admin",  # Specify auth database
        #     authMechanism="SCRAM-SHA-256",  # Explicit authentication method
        #     serverSelectionTimeoutMS=5000
        # )
        self.client = MongoClient("mongodb://localhost:27017/") 
        try:
            # Verify connection works
            self.client.admin.command('ping')
            print("Successfully connected to MongoDB")
        except Exception as e:
            print("MongoDB connection failed:", e)
            raise
        
        self.db = self.client["rag_logs"]
        self.queries = self.db["queries"]
        
    def log(self, query, context, answer):
        record = {
            "timestamp": "timestamp": datetime.now(timezone.utc), # datetime.utcnow(), fixed depcriciation warning
            "query": query,
            "context": context,
            "answer": answer,
            "model": "mixtral_8x7b"
        }
        self.queries.insert_one(record)
        return record

# format the retrive documents together to send as context to LLM
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

# Start timer
start_time = time.time()

# Enter the path to your file
documents_loaded = load_chunk_documents(r"Documents\1-Definition and Macroeconomic Issues-16-07-2024.pdf")

#creating a vector and storing it in a Vector Database
vector_store = VectorDB(documents_loaded)

#Open MongoDB logger for storing all data
logger = MongoLogger()

# Retrieve the context close to your query
retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs=({"k":5,"lambda_mult":0.5}) #lambda_mult is relevance-div balance
    )

#name of the model you're using
llm = ChatNVIDIA(
    model = "mistralai/mixtral-8x7b-instruct-v0.1",
    temperature = 1.3,
    api_key=nim_api_key
)

#predefine prompt - change it to your need
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.
    Context : 
    {context}
    Question: 
    {question}
    """,
    input_variables = ['context', 'question']
)

# Parallel chainning the entire RAG model
parallel_chain = RunnableParallel({
    'context' : retriever | RunnableLambda(format_docs),
    'question' : RunnablePassthrough()
}) 
parser = StrOutputParser()
final_chain = parallel_chain | prompt | llm | parser

# enter your query 
question = "Explain me the core idea in 1 line"
answer = final_chain.invoke(question)
context = format_docs(retriever.invoke(question))

#Log the query answer and context to MongoDB
logger.log(question,context,answer)

# End timer and calculate execution time
end_time = time.time()
execution_time = end_time - start_time

print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"\nRetrieved Context:\n{context[:200]}...")
print(f"\nExecution Time: {execution_time:.2f} seconds")

