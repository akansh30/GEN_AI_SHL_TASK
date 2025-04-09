import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Loading CSV file
df = pd.read_csv("shl_assessments_full.csv")

# Initializing embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Defining the embedding column
texts = df["Description"].fillna("").tolist()

# Creating embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Converting to FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Saving index
faiss.write_index(index, "shl_index.faiss")

df.to_csv("shl_assessments_with_ids.csv", index=False)
