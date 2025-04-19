import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize FastAPI app
app = FastAPI()

# Allow all origins and all methods/headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Shared LLM and embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="Llama3-70b-8192")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Global retriever to reuse
global_retriever = None

# Upload PDF Endpoint
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global global_retriever
    try:
        # Save file temporarily
        file_id = uuid.uuid4().hex
        file_path = f"temp_{file_id}.pdf"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load and process PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs[:50])
        vector_store = FAISS.from_documents(split_docs, embeddings)
        global_retriever = vector_store.as_retriever()

        # Cleanup temp file
        os.remove(file_path)

        return {"message": "PDF uploaded and processed successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Ask Question Endpoint
class QueryModel(BaseModel):
    query: str

@app.post("/ask_question/")
async def ask_question(data: QueryModel):
    global global_retriever

    if global_retriever is None:
        return JSONResponse(status_code=400, content={"error": "Please upload a PDF first."})

    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(global_retriever, document_chain)
        response = retrieval_chain.invoke({'input': data.query})
        return {"answer": response['answer']}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
