from flask import Flask, request, jsonify
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load FAISS index and metadata
index = faiss.read_index("roadmap_index.faiss")
with open("roadmap_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ollama model
OLLAMA_MODEL = "llama3.2"

@app.route('/generate', methods=['POST'])
def generate_roadmap():
    data = request.get_json()
    query = data.get('query', '')
    
    # Get timeline from request (default to 8 if not provided)
    timeline = data.get('timeline', 8)
    
    # Get timeline_unit from request (default to "weeks" if not provided)
    timeline_unit = data.get('timeline_unit', 'weeks')

    # Step 1: Embed query
    query_vector = model.encode([query]).astype('float32')

    # Step 2: Get top 5 matches from FAISS
    distances, indices = index.search(query_vector, 5)
    top_chunks = [
        f"{metadata[i]['title']}: {metadata[i]['description']}"
        for i in indices[0] if i < len(metadata)
    ]
    context = "\n".join(top_chunks)

    # Step 3: Prompt for roadmap with user-defined timeline
#     prompt = f"""You are an expert AI learning assistant.

# Based on the following concepts:

# {context}

# Generate a clear, personalized AI learning roadmap with {timeline} {timeline_unit} for a user interested in: "{query}".

# Each {timeline_unit.rstrip('s')} should have a **title** and a clear purpose. Avoid using just "Week 1", "Week 2", etc. or "Month 1", "Month 2", etc.

# At the end, include a Mermaid.js diagram. Node names must use **underscores** only (no spaces, no dots, no special characters). Example:
# ```mermaid
# graph TD
# Start --> Basics_AI
# Basics_AI --> Perception_Systems
# ...
# ```"""
    # Modified prompt 
    prompt = f"""You are an expert AI learning assistant.

Based on the following concepts:

{context}

The user is interested in: "{query}"

IMPORTANT INSTRUCTIONS:
1. If the user wants to LEARN ABOUT a topic, create a learning roadmap.
2. If the user wants to BUILD something, create a development roadmap.
3. Focus SPECIFICALLY on what the user wants, not general AI concepts unless directly relevant.

Generate a clear, personalized roadmap with {timeline} {timeline_unit}.
Each {timeline_unit.rstrip('s')} should have a **title** and a clear purpose.

At the end, include a Mermaid.js diagram. Node names must use **underscores** only (no spaces, no dots, no special characters)."""

    # Step 4: Send to Ollama
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            output = response.json()["response"]
        else:
            output = f"❌ Ollama error ({response.status_code}): {response.text}"
    except Exception as e:
        output = f"❌ Failed to connect to Ollama: {e}"

    return jsonify({"roadmap": output})

if __name__ == '__main__':
    app.run(debug=True, port=5000)