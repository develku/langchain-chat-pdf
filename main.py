import os
import tempfile
import time
import streamlit as st
# Importing necessary libraries and modules from langchain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Workaround for SQLite issue in specific deployment environments.
# This is a fix for environments where the default SQLite setup causes issues.
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Adding a 'Buy Me a Coffee' button as a support/donation option for users
button(username="develku", floating=True, width=221)

# Setting up the title and layout for the Streamlit application
st.title("ChatPDF")
st.write("---")

# Creating a field for users to input their OpenAI API key securely
openai_key = st.text_input("Enter your OpenAI Key", type="password")

# Allowing the user to select between different GPT models (GPT-4 and GPT-3.5-turbo)
model_choices = ["gpt-4", "gpt-3.5-turbo"]  # List of available models
selected_model = st.selectbox(
    "Choose a GPT model", model_choices, index=0)  # Default set to GPT-4

# Creating a file uploader for users to upload PDF files
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
st.write("---")

# Function to process and split the uploaded PDF file into individual pages


def pdf_to_document(uploaded_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
        return pages


# Processing the uploaded PDF file if available
if uploaded_file is not None:
    try:
        pages = pdf_to_document(uploaded_file)

        # Configuring the text splitter to segment the PDF into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function=len, is_separator_regex=False)

        # Splitting the PDF into segments for processing
        texts = text_splitter.split_documents(pages)

        # Initializing OpenAI embeddings with the user's API key
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)

        # Creating a Chroma vector store with the processed text segments
        db = Chroma.from_documents(texts, embedding_model)

        # Custom handler class for streaming output
        class StreamHandler(BaseCallbackHandler):
            def __init__(self, container, initial_text=""):
                self.container = container
                self.text = initial_text

            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.text += token
                self.container.markdown(self.text)

        # Interface for users to input their questions
        st.header("Ask the PDF!")
        question = st.text_input("Type your question")

        # Submit button for processing the question
        if st.button("Ask"):
            with st.spinner('Wait for it...'):
                chat_box = st.empty()
                stream_handler = StreamHandler(chat_box)

                # Initializing the ChatOpenAI model with user-selected configurations
                llm = ChatOpenAI(model_name=selected_model,
                                 temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_handler])

                # Creating a question-answering chain with the ChatOpenAI model and Chroma retriever
                qa_chain = RetrievalQA.from_chain_type(
                    llm, retriever=db.as_retriever())
                qa_chain({"query": question})

    except Exception as e:
        # Error handling for any exceptions during the process
        st.error(f"An error occurred: {e}")
