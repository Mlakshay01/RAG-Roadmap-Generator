import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # This will take some time first time (model download)

print("Reading JSON...")
with open("C:/Users/laksh/Desktop/RAG PROJECT/dataset/ai roadmap.json", "r") as file:

    data = json.load(file)

texts = []
metadatas = []

print("Preparing data...")
for phase in data["phases"]:
    for node in phase["nodes"]:
        full_text = f"{node['title']}: {node['description']}"
        texts.append(full_text)
        metadatas.append({
            "id": node["id"],
            "title": node["title"],
            "description": node["description"],
            "level": node["level"],
            "resources": node.get("resources", []),
            "phase": phase["phase"]
        })

print(f"Total texts: {len(texts)}")

print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

print("Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Saving FAISS index to file...")
faiss.write_index(index, "roadmap_index.faiss")

print("Saving metadata to file...")
with open("roadmap_metadata.pkl", "wb") as f:
    pickle.dump(metadatas, f)

print("✅ All done! FAISS vector database and metadata saved successfully!")
