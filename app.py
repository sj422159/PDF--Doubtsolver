import streamlit as st
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import os

# Initialize Streamlit UI
st.title("📄 RAG-powered PDF Q&A")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    # Extract text
    pdf_reader = PdfReader(uploaded_file)
    raw_text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

    # Split text for embedding
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    text_chunks = text_splitter.split_text(raw_text)

    # Initialize embeddings and FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.from_texts(text_chunks, embeddings)

    # QA model
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    # User Question
    query = st.text_input("Ask a question about the PDF:")
    if query:
        # Retrieve relevant chunks
        relevant_chunks = index.similarity_search(query, k=3)
        context = " ".join([chunk.page_content for chunk in relevant_chunks])

        if context.strip():
            # Answer the question
            qa_input = {"question": query, "context": context}
            answer = qa_pipeline(qa_input)

            # Display Answer
            st.subheader("📌 Answer:")
            st.write(answer["answer"])

            # Explanation
            st.subheader("📖 Brief Explanation:")
            st.write("This is the core concept behind your question. The extracted context helped in forming this answer.")

            # Example Generation
            st.subheader("🔹 Example:")
            st.write(f"For instance, consider this scenario: {context[:200]} ...")

        else:
            st.warning("No relevant information found in the PDF.")
