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
import sys

# Workaround for a known SQLite issue in specific deployment environments.
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Adding a Buy Me a Coffee button for support/donations.
button(username="develku", floating=True, width=221)

# Setting up the title and layout for the Streamlit application.
st.title("ChatPDF")
st.write("---")

# Creating a field for users to enter their OpenAI API key.
openai_key = st.text_input("Enter your OpenAI Key", type="password")

# Creating an uploader for users to upload PDF files.
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
st.write("---")

# Function to process the uploaded PDF file.


def pdf_to_document(uploaded_file):
    # Temporarily storing and handling the uploaded file.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        # Loading and splitting the PDF into individual pages.
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
        return pages


# Processing the uploaded PDF file if available.
if uploaded_file is not None:
    try:
        pages = pdf_to_document(uploaded_file)

        # Splitting the PDF text into smaller chunks.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function=len, is_separator_regex=False)
        texts = text_splitter.split_documents(pages)

        # Embedding the text for numerical representation.
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
        db = Chroma.from_documents(texts, embedding_model)

        # Custom handler for streaming output.
        class StreamHandler(BaseCallbackHandler):
            def __init__(self, container, initial_text=""):
                self.container = container
                self.text = initial_text

            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.text += token
                self.container.markdown(self.text)

        st.header("Ask the PDF!")
        question = st.text_input("Type your question")

        if st.button("Ask"):
            with st.spinner('Wait for it...'):
                chat_box = st.empty()
                stream_handler = StreamHandler(chat_box)

                # Initializing ChatOpenAI model with streaming capability.
                llm = ChatOpenAI(model_name="gpt-4",
                                 temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_handler])

                # Creating a QA chain for answering questions from the PDF.
                qa_chain = RetrievalQA.from_chain_type(
                    llm, retriever=db.as_retriever())
                qa_chain({"query": question})

    except Exception as e:
        # Displaying any exceptions that occur during processing.
        st.error(f"An error occurred: {e}")
