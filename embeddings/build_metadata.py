import os, json, pickle
import numpy as np
from dotenv import load_dotenv
import openai

# Load your OpenAI API key from .env or environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Config
INPUT_FILE = "embeddings/clean_chunks.json"
OUTPUT_FILE = "embeddings/metadata.pkl"
EMBED_MODEL = "text-embedding-3-small"

# Load clean chunks
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

# Batch embed
def get_embeddings(texts, model):
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai.embeddings.create(input=batch, model=model)
        all_embeddings.extend([e.embedding for e in response.data])
    return all_embeddings

# Generate embeddings
print(f"🔍 Embedding {len(texts)} chunks...")
embeddings = get_embeddings(texts, EMBED_MODEL)

# Save metadata file
print("💾 Saving metadata.pkl...")
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({
        "embeddings": np.array(embeddings),
        "metadata": chunks
    }, f)

print("✅ Done! metadata.pkl is ready.")
