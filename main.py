import os
from dotenv import load_dotenv
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

load_dotenv()

st.title("News Research App 📰")

st.sidebar.title("News Article URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url = st.sidebar.button("Process the URLs")

main_placefolder = st.empty()

if process_url:

    # Load documents
    loader = WebBaseLoader(urls)
    main_placefolder.text("Data Loading...Started...*️⃣*️⃣*️⃣")
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=200,
        # length_function=len
    )
    main_placefolder.text("Text Splitting...Started...*️⃣*️⃣*️⃣")
    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Create vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)

    main_placefolder.text("Vectorstore Creation...Started...*️⃣*️⃣*️⃣")


    # Save vectorstore
    vectorstore.save_local("faiss_index_")

query = main_placefolder.text_input("Question:")

if query:


    # Load vectorstore
    # persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings,allow_dangerous_deserialization=True)

    # # Create retriever
    # retriever = persisted_vectorstore.as_retriever()

    # # Create LLM
    # llm = Ollama(model="llama3.1")

    # # Create QA
    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # while True:
    #     query = input("Type your query if you want to exit type Exit: \n")
    #     if query == "Exit":
    #         break
    #     result = qa.invoke(query)
    #     print(result.get('result'))