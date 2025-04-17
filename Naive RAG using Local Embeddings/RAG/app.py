import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

# Local Embedding function definition
sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Chroma Client Initialization with Persistence
chromaClient = chromadb.PersistentClient(path="chroma_persistent_storage")

collection_name = "document_qa_collection"
collection = chromaClient.get_or_create_collection(
    name=collection_name,
    embedding_function=sentence_transformer_ef
)

def load_documents_from_directory(dir_path):
    print("---Loading documents from directory---")
    documents=[]
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(dir_path, filename), "r",encoding="utf-8"
            ) as file:
                documents.append({"id":filename, "text": file.read()})
    return documents

#Splitting text to chunk
def split_text(text,chunk_size=1000, chunk_overlap=20):
    chunks=[]
    start=0
    while start < len(text):
        end= start+ chunk_size
        chunks.append(text[start:end])
        start=end- chunk_overlap
    return chunks

dir_path = "news_articles"
documents = load_documents_from_directory(dir_path)
print(f"Loaded {len(documents)} documents")

chunked_documents=[]
for doc in documents:
    chunks= split_text(doc["text"])
    print("--- Splitting documents into chunks---")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text":chunk})

#print(f"Split documents into {len(chunked_documents)} chunks")

for doc in chunked_documents:
    print("--- Inserting Chunks into DB ---")
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])

def query_documents(question, n_results=2):
    results= collection.query(query_texts=[question], n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("--- Returning relevant chunks ---")
    return relevant_chunks

def generate_response(question, relevant_chunks):
    context= "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    print("---- Generated Answer ---")
    return prompt

question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)