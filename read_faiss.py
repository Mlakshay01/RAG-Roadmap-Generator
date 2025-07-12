import faiss
index = faiss.read_index("roadmap_index.faiss")
print("Number of vectors in FAISS index:", index.ntotal)
