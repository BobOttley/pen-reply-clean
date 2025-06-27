import os
import json
import pickle
import re
import numpy as np

INPUT_PATH = "output/clean_chunks.json"
OUTPUT_METADATA = "output/metadata.pkl"
OUTPUT_TEXT = "output/text_chunks.txt"

CHUNK_SIZE = 800
OVERLAP = 100

def fallback_chunker(text, max_len=CHUNK_SIZE, overlap=OVERLAP):
    # Splits by sentence but falls back to slicing if no punctuation
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) <= max_len:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent if len(sent) < max_len else sent[-overlap:]
    if current:
        chunks.append(current.strip())
    return chunks

# Load raw scraped data
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError("clean_chunks.json not found. Run the scraper first.")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    raw_chunks = json.load(f)

prepared_chunks = []
embedding_texts = []

for page in raw_chunks:
    text = page.get("text", "").strip()
    url = page.get("url", "")
    if not text or len(text) < 50:
        continue
    chunks = fallback_chunker(text)
    for i, chunk in enumerate(chunks):
        prepared_chunks.append({
            "text": chunk,
            "url": url,
            "type": page.get("type", "html"),
            "chunk": i
        })
        embedding_texts.append(chunk)

# Dummy embeddings (real ones generated in app.py later)
dummy_embeddings = np.random.rand(len(prepared_chunks), 1536)

metadata_obj = {
    "embeddings": dummy_embeddings,
    "metadata": prepared_chunks
}

# Save metadata.pkl
with open(OUTPUT_METADATA, "wb") as f:
    pickle.dump(metadata_obj, f)

# Save readable chunks
with open(OUTPUT_TEXT, "w", encoding="utf-8") as f:
    for entry in prepared_chunks:
        f.write(f"{entry['url']}\n{entry['text']}\n\n{'-'*60}\n\n")

print(f"✅ Prepared {len(prepared_chunks)} clean content chunks.")
