from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
import chromadb

# embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# llm
llm = OllamaLLM(model="llama3.2")
critic_llm = OllamaLLM(model="llama3.2") 

# database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="movies3")

vStore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embedding_model
)

retriever = vStore.as_retriever()