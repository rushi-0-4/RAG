from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

file_path="data_source.txt"
model_name="all-MiniLM-L6-v2"
model=SentenceTransformer(model_name)
vector_index=None
chunks=[]
def load_data(file_path):
    '''
    Load the data from a text file and return it as a string. 
    '''
    if file_path.endswith('.txt'):
        #UTF‑8 makes sure your text file is read properly into Python strings. The model then handles converting those strings into machine‑readable format.
        with open(file_path,'r',encoding='utf-8') as f:
            return f.read()
    else:
        print("Provide .txt extension files only")

def split_text(text):
    '''
    Split the text into chunks based on a delimiter.
    '''
    text=text.split("\n##")
    return text

def indexing():
    '''
    Index the text data by creating embeddings for each chunk and storing them in a FAISS index.
    This allows for efficient similarity search later on.
    '''
    global chunks, vector_index
    data=load_data(file_path)
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
    embed_query=model.encode([query]).astype('float32')
    distances, indices=vector_index.search(embed_query,top)
    vector_chunks=[chunks[i] for i in indices[0]]
    return vector_chunks
indexing()
