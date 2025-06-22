import os, json, pickle, re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from scipy.spatial.distance import cosine

load_dotenv()
app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("🚀 PEN Reply Flask server starting (torch-free)")

# Config
EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.3
RESPONSE_LIMIT = 3

# Load knowledge base
with open("embeddings/metadata.pkl", "rb") as f:
    data = pickle.load(f)
    doc_embeddings = data["embeddings"]
    metadata = data["metadata"]

print(f"✅ Loaded {len(metadata)} knowledge chunks.")


# Helper: Embed query
def embed_text(text):
    result = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return result.data[0].embedding


# Helper: Markdown to HTML
def convert_markdown_to_html(text):
    return re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)


@app.route("/reply", methods=["POST"])
def reply():
    try:
        data = request.get_json()
        question = data.get("message", "").strip()

        if not question:
            return jsonify({"reply": "<p>⚠️ No message received.</p>"}), 400

        print(f"📩 Parent enquiry: {question}")

        query_vec = embed_text(question)
        # Score similarity against each document
        scored = []
        for vec, meta in zip(doc_embeddings, metadata):
            try:
                similarity = 1 - cosine(query_vec, vec)
                if similarity >= SIMILARITY_THRESHOLD:
                    scored.append((similarity, meta))
            except Exception:
                continue

        top_matches = sorted(scored, key=lambda x: x[0], reverse=True)[:RESPONSE_LIMIT]

        if not top_matches:
            return jsonify({
                "reply": "<p>Thank you for your enquiry. A member of our admissions team will follow up with you shortly.</p>"
            })

        # Build prompt context
        top_context = ""
        for _, meta in top_matches:
            url = meta.get("url", "")
            top_context += f"{meta['text']}\n[Source]({url})\n---\n"

        # GPT prompt
        prompt = f"""
You are Jess Ottley-Woodd, Director of Admissions at Bassett House School, a UK prep school.

A parent has sent the following enquiry. Write a warm, caring, professional, informative reply using only the school information provided.
If helpful, include source URLs using Markdown-style links where the **anchor text is meaningful**, like [personalised prospectus](https://...).
Do not list links at the bottom. Do not use raw URLs. Never say 'click here'.

Weave the links naturally into the sentence like a professional email.

Parent Email:
\"\"\"
{question}
\"\"\"

School Info:
\"\"\"
{top_context}
\"\"\"

Your Reply:

Please sign off the email as:
Jess Ottley-Woodd  
Director of Admissions  
Bassett House School
"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        reply_markdown = response.choices[0].message.content.strip()
        reply_html = convert_markdown_to_html(reply_markdown)
        return jsonify({"reply": reply_html})

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return jsonify({
            "reply": "<p>⚠️ Internal error generating the reply.</p>"
        }), 500


@app.route("/", methods=["GET"])
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        return f"500 INTERNAL ERROR: {e}", 500


if __name__ == "__main__":
    app.run(debug=True)
