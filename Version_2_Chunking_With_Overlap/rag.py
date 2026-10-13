from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# Path to shared data_source.txt at repository root
file_path = str(Path(__file__).resolve().parent.parent / "data_source.txt")
model_name="all-MiniLM-L6-v2"
model=SentenceTransformer(model_name)
vector_index: faiss.IndexFlatL2 | None = None
chunks: list[str] = []

def load_data(file_path: str) -> str:
    '''
    Load the data from a text file and return it as a string. 
    '''
    if file_path.endswith('.txt'):
        #UTF‑8 makes sure your text file is read properly into Python strings. The model then handles converting those strings into machine‑readable format.
        with open(file_path,'r',encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Provide .txt extension files only")

def clean_text(text: str):
    """Clean common text-extraction and Markdown noise."""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove Markdown heading markers at the start of lines
    text = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+", "", text)

    # Remove bold and italic formatting markers while preserving text
    text = re.sub(r"\*\*\*(.*?)\*\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)

    # Remove excessive spaces and tabs, preserving paragraph boundaries
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100):
    '''
    Split text into overlapping chunks using a sliding window.
    '''
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks = []
    text_length = len(text)
    
    # Calculate the step size
    step = chunk_size - chunk_overlap
    for start in range(0, text_length, step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        
    return chunks

def indexing():
    '''
    Index the text data by creating embeddings for each chunk and storing them in a FAISS index.
    This allows for efficient similarity search later on.
    '''
    global chunks, vector_index
    data=load_data(file_path)
    data=clean_text(data)
    chunks=split_text(data)
    embed=model.encode(chunks)
    #store the emdeb chunks in vector index
    #converting the chunks into numpy each element is in float type
    embedding_array=np.array(embed).astype('float32')
    #shape[0]-row and shape[1]-column
    #Example:
    #embedding_array = [[0.12, -0.45, 0.87]]
    #embedding_array.shape[0]=1-> chunk
    #embedding_array.shape[1]=3->embedding has 3 numbers
    dim=embedding_array.shape[1]
    #Creates a FAISS index that uses L2 distance (Euclidean distance) to compare vectors.
    vector_index=faiss.IndexFlatL2(dim)
    #Adds all your chunk embeddings into the FAISS index.
    vector_index.add(embedding_array)
    print("Indexing Completed")

def search(query,top=5):
    '''
    Search for the most similar chunks to the query using the FAISS index.
    Returns the top N most similar chunks.
    '''
    global vector_index
    if vector_index is None:
        indexing()
    if vector_index is None:
        raise RuntimeError("Vector index has not been initialized.")

    embed_query=model.encode([query]).astype('float32')
    distances, indices=vector_index.search(embed_query,top)
    vector_chunks=[chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
    return vector_chunks
indexing()