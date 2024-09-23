import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from llama_index.llms.groq import Groq
import logging
import os

def main():
    # Load environment variables
    load_dotenv()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Set up Groq LLM
    api_key = os.getenv("GROQ_API_KEY")  # Store your Groq API key in a .env file
    llm = Groq(model="llama3-70b-8192", api_key=api_key)

    # Load the SentenceTransformer model for embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Upload PDF
    pdf = st.file_uploader("**Upload your PDF**", type='pdf')

    if pdf is not None:
        # Read and extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Create or load FAISS vector store
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            # Generate embeddings using SentenceTransformer
            embeddings = embedding_model.encode(chunks)
            
            # Create FAISS vector store from the embeddings
            VectorStore = FAISS.from_embeddings(embeddings, chunks)
            
            # Save the vector store for future use
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user query
        query = st.text_input("**Ask questions about your PDF file:**")

        if query:
            # Search for similar documents
            docs = VectorStore.similarity_search(query=query, k=3)

            # Create the prompt for Groq
            prompt = "Answer the following question based on these documents: "
            prompt += "\n\n".join([doc.page_content for doc in docs])
            prompt += f"\n\nQuestion: {query}"

            # Generate the response using Groq
            try:
                response = llm.complete(prompt)
                answer = response.text.strip()

                # Display the response
                st.markdown(f"**Answer:**", unsafe_allow_html=True)
                st.markdown(f"{answer}", unsafe_allow_html=True)

            except Exception as e:
                logging.error(f"Error occurred while generating response: {e}")
                st.write("An error occurred while processing your request.")
                  
    else:
        st.write("No file uploaded")

if __name__ == '__main__':
     main()
