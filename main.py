# Import necessary modules for various tasks including PDF handling, text splitting, embeddings,
# vector storage, chat modeling, and question answering in Streamlit.
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os
import tempfile
import streamlit as st
import sys

# Fix for SQLite issue in certain environments.
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Initialize the Streamlit web application with a title and separator lines.
st.title("ChatPDF")
st.write("---")

# Create an uploader widget to allow users to upload PDF files.
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
st.write("---")

# Define a function to handle the uploaded PDF file.


def pdf_to_document(uploaded_file):
    # Create a temporary directory to store the uploaded PDF file.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        # Load the PDF and split it into pages.
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
        return pages


# Check if the user has uploaded a PDF file.
if uploaded_file is not None:
    # Process the uploaded PDF to get its pages.
    pages = pdf_to_document(uploaded_file)

    # Configure a text splitter to divide the PDF into smaller text chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=20, length_function=len, is_separator_regex=False)

    # Split the PDF into smaller segments for processing.
    texts = text_splitter.split_documents(pages)

    # Initialize the OpenAI Embeddings model for text representation.
    embedding_model = OpenAIEmbeddings()

    # Create a Chroma vector store with the text segments and the embedding model.
    db = Chroma.from_documents(texts, embedding_model)

    # Add an input section in the Streamlit app for users to type their questions.
    st.header("Ask the PDF!")
    question = st.text_input("Type your question")

    # Add a button in the app for submitting the question.
    if st.button("Ask"):
        # Ensure the question is not empty before processing.
        if question:
            # Show a spinner while the question is being processed.
            with st.spinner('Processing...'):
                # Initialize the ChatOpenAI model with specific configurations.
                llm = ChatOpenAI(model_name="gpt-4", temperature=0)
                # Create a QA chain with the model and Chroma retriever for answering the question.
                qa_chain = RetrievalQA.from_chain_type(
                    llm, retriever=db.as_retriever())
                # Retrieve the answer to the user's question.
                result = qa_chain({"query": question})
                st.success('Done!')
                # Display the answer.
                st.write(result["result"])
        else:
            # Display an error message if no question is entered.
            st.error("Please enter a question.")
