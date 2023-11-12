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
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# Workaround for SQLite issue in specific deployment environments.
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# button for buy me a coffee
button(username="develku", floating=True, width=221)

# Set up the Streamlit app's title and a separator for layout.
st.title("ChatPDF")
st.write("---")

# Getting OpenAI KEY from User
openai_key = st.text_input("Enter your OpenAI Key", type="password")

# Create a file uploader in the Streamlit interface for users to upload PDF files.
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
st.write("---")

# Define a function to process the uploaded PDF file.


def pdf_to_document(uploaded_file):
    # Create a temporary directory to store and handle the uploaded file.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        # Load and split the PDF file into individual pages.
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
        return pages


# Check if a PDF file has been uploaded.
if uploaded_file is not None:
    try:
        # Process the uploaded PDF and retrieve its pages.
        pages = pdf_to_document(uploaded_file)

        # Configure the text splitter for segmenting the PDF into smaller chunks.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=20, length_function=len, is_separator_regex=False)

        # Split the PDF into manageable text segments.
        texts = text_splitter.split_documents(pages)

        # Initialize OpenAI embeddings for numerical representation of text.
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)

        # Create a Chroma vector store with the text segments and the embedding model.
        db = Chroma.from_documents(texts, embedding_model)

        # Add an interface element for users to ask questions about the PDF.
        st.header("Ask the PDF!")
        question = st.text_input("Type your question")

        # Add a button to submit the question for processing.
        if st.button("Ask"):
            if question:
                # Show a spinner and pause for a short duration while processing.
                with st.spinner('Wait for it...'):
                    time.sleep(5)

                # Initialize the ChatOpenAI model with specified configurations.
                llm = ChatOpenAI(model_name="gpt-4",
                                 temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

                # Create a question-answering chain with the ChatOpenAI model and Chroma retriever.
                qa_chain = RetrievalQA.from_chain_type(
                    llm, retriever=db.as_retriever())
                result = qa_chain({"query": question})

                st.success('Done!')
                # Display the result from the question-answering process.
                st.write(result["result"])
            else:
                # Display an error if no question is entered.
                st.error("Please enter a question.")
    except Exception as e:
        # Display an error message if any exception occurs during the process.
        st.error(f"An error occurred: {e}")
