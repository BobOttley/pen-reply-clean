import os
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────────────
# CONFIG & PATHS
# ─────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent
SCRAPED_DIR = BASE_DIR.parent / "scraped"
CONTENT_FILE = SCRAPED_DIR / "enhanced_content.jsonl"
EMBEDDINGS_FILE = SCRAPED_DIR / "embeddings.pkl"

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("📄 Loading documents...")
with open(CONTENT_FILE, encoding="utf-8") as f:
    documents = [json.loads(line) for line in f if '"error"' not in line]

model = SentenceTransformer("all-MiniLM-L6-v2")

if EMBEDDINGS_FILE.exists():
    print("📦 Loading cached embeddings...")
    doc_embeddings = pickle.loads(EMBEDDINGS_FILE.read_bytes())
else:
    print("🧠 Generating new embeddings...")
    text_blocks = [doc["title"] + " " + doc["summary"] for doc in documents]
    doc_embeddings = model.encode(text_blocks, convert_to_tensor=True)
    EMBEDDINGS_FILE.write_bytes(pickle.dumps(doc_embeddings))
    print("✅ Embeddings saved to cache.")

# ─────────────────────────────────────────────
# FLASK SETUP
# ─────────────────────────────────────────────
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder="templates", static_folder="static")

# ─────────────────────────────────────────────
# Serve homepage
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def serve_index():
    return render_template("index.html")

# ─────────────────────────────────────────────
# Serve static files (e.g. CSS)
# ─────────────────────────────────────────────
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# ─────────────────────────────────────────────
# API endpoint: Generate reply
# ─────────────────────────────────────────────
@app.route("/reply", methods=["POST"])
def reply():
    data = request.get_json()
    query = data.get("message", "").strip()

    if not query:
        return jsonify({"error": "No message provided."}), 400

    results = find_relevant_docs(query)
    response = generate_reply(query, results)
    return jsonify({"reply": response})

# ─────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────
def find_relevant_docs(query, top_k=3):
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, doc_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    return [documents[i] for i in top_indices]

def generate_reply(query, docs):
    links = "\n".join(
        f"- **{d['title']}**: {d['summary']}" +
        (f" [Read here]({d['url']})" if 'url' in d else "")
        for d in docs
    )

    prompt = f"""
You are Jess Ottley-Woodd, Director of Admissions at Basset House School.
Write a warm, helpful reply to the parent enquiry below. Use British spelling.
Embed references from the supporting documents where helpful.

Parent enquiry:
\"\"\"{query}\"\"\"

Supporting documents:
{links}

Sign off as:
Jess Ottley-Woodd
Director of Admissions
Basset House School

Now write the email reply:
"""

    try:
        chat = client.chat.completions.create(
            model="gpt-4",
            temperature=0.4,
            messages=[{"role": "user", "content": prompt}]
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error generating reply: {e}"

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False)
