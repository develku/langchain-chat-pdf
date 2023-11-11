# Import modules for PDF processing, text splitting, embedding generation, environment management,
# vector storage, chat modeling, and retrieval-based question answering.
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables for configurations such as API keys.
load_dotenv()

# Initialize a loader for the PDF file "romeo_juliet.pdf".
loader = PyPDFLoader("romeo_juliet.pdf")

# Load the PDF and split it into individual pages.
pages = loader.load_and_split()

# Configure a text splitter for creating smaller text chunks from the PDF pages.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,          # Defines the size of each chunk.
    # Determines the overlap between consecutive chunks.
    chunk_overlap=20,
    length_function=len,     # Function to measure text length.
    # Indicates whether a regex pattern is used as a separator.
    is_separator_regex=False
)

# Split the PDF into smaller text segments.
texts = text_splitter.split_documents(pages)

# Create an instance of OpenAI Embeddings for numerical text representation.
embedding_model = OpenAIEmbeddings()

# Set up a Chroma vector store with the text segments and embedding model.
# This store will hold vector representations of the text for retrieval.
db = Chroma.from_documents(texts, embedding_model)

# Define a specific question for information retrieval.
question = "What is the name of the main character in Romeo and Juliet?"

# Initialize a ChatOpenAI model with GPT-4 and a specified temperature.
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Create a question-answering chain using the ChatOpenAI model and Chroma retriever.
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

# Retrieve an answer to the question using the QA chain.
result = qa_chain({"query": question})

# Print the retrieved answer.
print(result)
