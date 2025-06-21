
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


with open("sdlc.txt", "r", encoding="utf-8") as file:
    data = file.read()


with open("cloud.txt", "r", encoding="utf-8") as file:
    data2 = file.read()



# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
texts = text_splitter.split_text(data)

# Set up HuggingFace embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Chroma DB and persist it 
db = Chroma.from_texts(
    texts,                     
    embedding,                 
    persist_directory="chroma_db_sdlc_store"
)

# Split into chunks
texts2 = text_splitter.split_text(data2)

# Create Chroma vector store with local persistence
db2 = Chroma.from_texts(texts2, embedding, persist_directory='chroma_db_cloud_store')



