from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from data_preprocessor import load_and_clean
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_index(df):
    model_path = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")  # Local model path
    model = SentenceTransformer(model_path)  # Load from local dir
    embeddings = model.encode(df['Text for Embedding'].tolist(), show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, model

def save_index(index, df, embeddings):
    faiss.write_index(index, os.path.join(BASE_DIR, 'shl_index.faiss'))
    df.to_csv(os.path.join(BASE_DIR, "shl_indexed_data.csv"), index=False)
    np.save(os.path.join(BASE_DIR, "embeddings.npy"), embeddings)

if __name__ == "__main__":
    df = load_and_clean()
    index, embeddings, model = build_index(df)
    save_index(index, df, embeddings)
