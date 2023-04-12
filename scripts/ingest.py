from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

import dotenv
ENV = dotenv.dotenv_values(".env")
PINECONE_INDEX_NAME = ENV["PINECONE_INDEX_NAME"]
# creating a pdf file object
pdfFileObj = PyPDFLoader("docs/Concepts.pdf")
  
documents = pdfFileObj.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
)
docs = text_splitter.split_documents(documents)

model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
pinecone.init(api_key=ENV["PINECONE_API_KEY"], environment=ENV["PINECONE_ENVIRONMENT"])

index = pinecone.Index(PINECONE_INDEX_NAME)
print(PINECONE_INDEX_NAME)
embedding = HuggingFaceEmbeddings(model_name=model_name)
vectorstore = Pinecone.from_documents(
    documents=docs,
    embedding=embedding,
    text_key="text",
    index_name=ENV["PINECONE_INDEX_NAME"],
)
    