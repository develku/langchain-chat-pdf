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

# Adding a 'Buy Me a Coffee' button for user donations
button(username="develku", floating=True, width=221)

# Set up the Streamlit app's title and a separator for layout.
st.title("ChatPDF")
st.write("---")

# Users to enter their OpenAI API key securely
openai_key = st.text_input("Enter your OpenAI Key", type="password")

# Allow the user to select a GPT model
model_choices = ["gpt-4", "gpt-3.5-turbo"]  # List of available models
selected_model = st.selectbox(
    "Choose a GPT model", model_choices, index=0)  # Default to GPT-4

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
            chunk_size=1000, chunk_overlap=100, length_function=len, is_separator_regex=False)

        # Split the PDF into manageable text segments.
        texts = text_splitter.split_documents(pages)

        # Initialize OpenAI embeddings for numerical representation of text.
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)

        # Create a Chroma vector store with the text segments and the embedding model.
        db = Chroma.from_documents(texts, embedding_model)

        # Handler for Stream
        from langchain.callbacks.base import BaseCallbackHandler

        class StreamHandler(BaseCallbackHandler):
            def __init__(self, container, initial_text=""):
                self.container = container
                self.text = initial_text

            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.text += token
                self.container.markdown(self.text)

        # Add an interface element for users to ask questions about the PDF.
        st.header("Ask the PDF!")
        question = st.text_input("Type your question")

        # Add a button to submit the question for processing.
        if st.button("Ask"):
            with st.spinner('Wait for it...'):
                chat_box = st.empty()
                stream_hander = StreamHandler(chat_box)

                # Initialize the ChatOpenAI model with specified configurations.
                llm = ChatOpenAI(model_name=selected_model,
                                 temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_hander])

                # Create a question-answering chain with the ChatOpenAI model and Chroma retriever.
                qa_chain = RetrievalQA.from_chain_type(
                    llm, retriever=db.as_retriever())
                qa_chain({"query": question})

    except Exception as e:
        # Display an error message if any exception occurs during the process.
        st.error(f"An error occurred: {e}")
