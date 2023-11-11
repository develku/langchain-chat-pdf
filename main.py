import os
import tempfile
import time
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Fix for SQLite issue in certain environments.


# Set up the title and a separator for the Streamlit web app interface.
st.title("ChatPDF")
st.write("---")

# Create an uploader in the Streamlit interface for PDF files.
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
st.write("---")

# Define a function to read and process the uploaded PDF file.


def pdf_to_document(uploaded_file):
    # Create a temporary directory to store the uploaded file.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
        return pages


# Check if a file has been uploaded.
if uploaded_file is not None:
    try:
        # Process the uploaded PDF and get its pages.
        pages = pdf_to_document(uploaded_file)

        # Configure a text splitter for smaller, manageable chunks from the PDF.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=20, length_function=len, is_separator_regex=False)

        # Split the PDF into smaller text segments.
        texts = text_splitter.split_documents(pages)

        # Instantiate OpenAI embeddings for numerical text representation.
        embedding_model = OpenAIEmbeddings()

        # Set up a Chroma vector store with the text segments and embedding model.
        db = Chroma.from_documents(texts, embedding_model)

        # Interface for asking questions to the PDF.
        st.header("Ask the PDF!")
        question = st.text_input("Type your question")

        # Button to trigger the question-answering process.
        if st.button("Ask"):
            if question:
                # add spinner for waiting
                with st.spinner('Wait for it...'):
                    time.sleep(5)
                st.success('Done!')

                # Initialize a ChatOpenAI model with specific configurations.
                llm = ChatOpenAI(model_name="gpt-4", temperature=0)

                # Create a question-answering chain using the ChatOpenAI model and Chroma retriever.
                qa_chain = RetrievalQA.from_chain_type(
                    llm, retriever=db.as_retriever())
                result = qa_chain({"query": question})

                # Display the result.
                st.write(result["result"])
            else:
                st.error("Please enter a question.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
