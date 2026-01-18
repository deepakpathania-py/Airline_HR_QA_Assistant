import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class HR_RAG_Engine:
    def __init__(self, google_api_key):
        self.google_api_key = google_api_key
        # Initialize Google Gemini Config
        if not google_api_key:
            raise ValueError("Google API Key is required")
        
        os.environ["GOOGLE_API_KEY"] = google_api_key

    def ingest_data(self, pdf_path):
        """
        Ingests the PDF file, splits it, and stores in Vector DB.
        """
        if not os.path.exists(pdf_path):
            return "File not found!"
            
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        
        # Create Embeddings and Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Persist directory for Chroma
        persist_directory = "./data/db"
        
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        return f"Successfully processed {len(texts)} chunks."

    def get_qa_chain(self):
        """
        Returns the QA Chain for querying
        """
        persist_directory = "./data/db"
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        vector_store = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always cite the source section or page number if available in the context.
        
        Context:
        {context}
        
        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return chain
