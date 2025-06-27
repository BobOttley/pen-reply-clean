import os
import json
import pickle
import re
import numpy as np
from datetime import datetime
from markdown import markdown
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from markdownify import markdownify as html_to_markdown


def parse_url_box(url_text):
    """
    Parses both newline-separated and semicolon-separated anchor=URL entries.
    """
    url_map = {}
    parts = re.split(r'[;\n]+', url_text.strip())
    for part in parts:
        if '=' in part:
            anchor, url = part.split('=', 1)
            url_map[anchor.strip()] = url.strip()
    return url_map




def insert_links(text, url_map):
    """
    Finds any words/phrases in the text that match the anchors and replaces
    them with Markdown links (e.g. Head → [Head](...)).
    """
    def safe_replace(match):
        word = match.group(0)
        for anchor, url in url_map.items():
            if word.lower() == anchor.lower():
                return f"[{word}]({url})"
        return word

    sorted_anchors = sorted(url_map.keys(), key=len, reverse=True)
    pattern = r'\b(' + '|'.join(re.escape(a) for a in sorted_anchors) + r')\b'
    return re.sub(pattern, safe_replace, text, flags=re.IGNORECASE)

# ──────────────────────────────
# ✅  SET-UP
# ──────────────────────────────
load_dotenv()
app = Flask(__name__, template_folder="templates")
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("🚀 PEN Reply Flask server starting…")

EMBED_MODEL               = "text-embedding-3-small"
SIMILARITY_THRESHOLD      = 0.30
RESPONSE_LIMIT            = 3
STANDARD_MATCH_THRESHOLD  = 0.85

# ──────────────────────────────
# 🔒  PII REDACTION
# ──────────────────────────────
PII_PATTERNS = [
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",       # emails
    r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b",                      # US-style phone
    r"\+?\d[\d\s\-]{7,}\d",                                      # int’l phone
    r"\b[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}\b",                    # UK postcode
]
def remove_personal_info(text:str)->str:
    for pat in PII_PATTERNS:
        text = re.sub(pat, "[redacted]", text, flags=re.I)
    # crude “My name is …” removal
    text = re.sub(r"\bmy name is ([A-Z][a-z]+)\b", "my name is [redacted]", text, flags=re.I)
    return text

# ──────────────────────────────
# 📦  HELPERS
# ──────────────────────────────
def embed_text(text: str) -> np.ndarray:
    text = text.replace("\n", " ")
    res  = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(res.data[0].embedding)

def markdown_to_html(text: str) -> str:
    """Convert markdown links to clickable HTML (keeps anchor text)."""
    return re.sub(
        r'\[([^\]]+)\]\((https?://[^\)]+)\)',
        lambda m: f'<a href="{m.group(2)}" target="_blank">{m.group(1)}</a>',
        text
    ) 

def clean_gpt_email_output(md: str) -> str:
    """Clean up GPT output to remove markdown/code block labels and subject lines."""
    md = md.strip()
    # Remove any triple backticks (and possible markdown label)
    md = re.sub(r"^```(?:markdown)?", "", md, flags=re.I).strip()
    md = re.sub(r"```$", "", md, flags=re.I).strip()
    # Remove 'markdown:' or 'Subject:' at the very start
    md = re.sub(r"^(markdown:|subject:)[\s]*", "", md, flags=re.I).strip()
    # Remove accidental 'Subject:' anywhere at the very start of a line
    md = re.sub(r"^Subject:.*\n?", "", md, flags=re.I)
    return md.strip()

def cosine_similarity(a, b):
    """Return cosine similarity between two numpy arrays."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


   

# ──────────────────────────────
# 📚  LOAD SCHOOL KB
# ──────────────────────────────
with open("embeddings/metadata.pkl", "rb") as f:
    kb = pickle.load(f)
    doc_embeddings = np.array(kb["embeddings"])
    metadata = kb["metadata"]

# ──────────────────────────────
# 📁  LOAD SAVED STANDARD RESPONSES
# ──────────────────────────────
standard_messages, standard_embeddings, standard_replies = [], [], []

def _load_standard_library():
    path="standard_responses.json"
    if not os.path.exists(path):
        print("⚠️ No standard_responses.json found.")
        return
    try:
        with open(path,"r") as f:
            saved=json.load(f)
        for entry in saved:
            msg = remove_personal_info(entry["message"])
            rep = entry["reply"]                   # reply already HTML-ised
            standard_messages.append(msg)
            standard_embeddings.append(embed_text(msg))
            standard_replies.append(rep)
        print(f"✅ Loaded {len(standard_messages)} template replies.")
    except Exception as e:
        print(f"❌ Failed loading templates: {e}")

_load_standard_library()

def check_standard_match(q_vec: np.ndarray) -> str:
    if not standard_embeddings: return ""
    sims = [cosine_similarity(q_vec, emb) for emb in standard_embeddings]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= STANDARD_MATCH_THRESHOLD:
        print(f"🔁 Using template (similarity {sims[best_idx]:.2f})")
        return standard_replies[best_idx]
    return ""

# ──────────────────────────────
# 📨  POST /reply
# ──────────────────────────────
@app.route("/reply", methods=["POST"])
def generate_reply():
    try:
        body = request.get_json(force=True)
        question_raw = (body.get("message") or "").strip()
        url_box_text = (body.get("url_box") or "").strip()
        url_map = parse_url_box(url_box_text)
        instruction_raw = (body.get("instruction") or "").strip()

        # 🔒 sanitise
        question    = remove_personal_info(question_raw)
        instruction = remove_personal_info(instruction_raw)

        if not question:
            return jsonify({"error":"No message received."}), 400

        q_vec = embed_text(question)

        # 1) pre-approved template?
        matched = check_standard_match(q_vec)
        if matched:
            # If matched is a string, no URL is available
            if isinstance(matched, str):
                return jsonify({
                    "reply": matched,
                    "sentiment_score": 10,
                    "strategy_explanation": "Used approved template.",
                    "url": "",
                    "link_label": ""
                })
            # If matched is a dict with 'reply', 'url', and 'link_label'
            else:
                return jsonify({
                    "reply": matched.get("reply", ""),
                    "sentiment_score": 10,
                    "strategy_explanation": "Used approved template.",
                    "url": matched.get("url", ""),
                    "link_label": matched.get("link_label", "")
                })


        # 2) sentiment (mini model, cheap)
        sent_prompt = f"""
Rate the sentiment (1–10) of the enquiry then give a response strategy in JSON:
{{"score":7,"strategy":"Begin warmly …"}}

Enquiry:
\"\"\"{question}\"\"\"""".strip()

        sent_json = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":sent_prompt}],
            temperature=0.3
        ).choices[0].message.content.strip()

        try:
            sent = json.loads(sent_json)
            score = int(sent.get("score",5))
            strat = sent.get("strategy","")
        except Exception:
            score, strat = 5, ""
            print("⚠️ Sentiment parse failed.")

        # 3) KB retrieval
        sims = [(cosine_similarity(q_vec, vec), meta) for vec, meta in zip(doc_embeddings, metadata)]
        top = [m for m in sims if m[0] >= SIMILARITY_THRESHOLD]
        top = sorted(top, key=lambda x:x[0], reverse=True)[:RESPONSE_LIMIT]

        if not top:
            return jsonify({
                "reply":"<p>Thank you for your enquiry. A member of our admissions team will contact you shortly.</p>",
                "sentiment_score":score,"strategy_explanation":strat
            })

        context_blocks = [f"{m['text']}\n[Info source]({m.get('url','')})" if m.get('url') else m['text']
                          for _,m in top]
        top_context = "\n---\n".join(context_blocks)

        # 4) main reply prompt

        from datetime import datetime

        today_date = datetime.now().strftime('%d %B %Y')

        prompt = f"""

TODAY'S DATE IS {today_date}.

You are Jess Ottley-Woodd, Director of Admissions at Bassett House School, a UK prep school.

Write a warm, professional email reply to the parent below, using only the approved school information provided.

Follow these essential rules:
- Always use British spelling (e.g. organise, programme, enrolment)
- Do NOT fabricate or guess any information. If something is unknown, say so honestly.
- DO include relevant links using Markdown format: [Anchor Text](https://...). Embed links naturally in the body of the reply.
- DO use approved anchor phrases like “Open Events page”, “Admissions page”, or “registration form”
- NEVER use vague anchors like “click here”, “more info”, “register here”, “visit page”, etc.
- NEVER show raw URLs, list links at the bottom, or use markdown formatting like bold, italics, or bullet points
- NEVER mention an Open Day – Bassett House does not offer them. Instead, refer to the Stay & Play sessions or Visit Us page as appropriate
- NEVER include expired dates. If unsure, direct the parent to the relevant web page instead

Reply only with the full email body in Markdown format, ready to send. Do not include 'Subject:', triple backticks, or code blocks.

Parent Email:
\"\"\"{question}\"\"\"

School Info:
\"\"\"{top_context}\"\"\"

Sign off:
Jess Ottley-Woodd  
Director of Admissions  
Bassett House School
""".strip()

        reply_md = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.4
        ).choices[0].message.content.strip()
        reply_md = clean_gpt_email_output(reply_md)


        # Format the reply
        reply_md = insert_links(reply_md, url_map)
        reply_html = markdown(reply_md)


        # ✅ Extract URLs from HTML
        import re

        def extract_links_from_html(html):
            matches = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html)
            return [(text.strip(), url.strip()) for url, text in matches]

        links = extract_links_from_html(reply_html)
        matched_url = links[0][1] if links else ""
        matched_source = links[0][0] if links else ""

        # ✅ Return enriched result
        return jsonify({
            "reply": reply_html,
            "sentiment_score": score,
            "strategy_explanation": strat,
            "url": matched_url,
            "link_label": matched_source
        })


    except Exception as e:
        print(f"❌ REPLY ERROR: {e}")
        return jsonify({"error":"Internal server error."}), 500

# ──────────────────────────────
# ✏️  POST /revise
# ──────────────────────────────

@app.route("/revise", methods=["POST"])
def revise():
    try:
        body = request.get_json(force=True)
        message_raw   = (body.get("message") or "").strip()
        prev_reply    = (body.get("previous_reply") or "").strip()
        instruction_raw = (body.get("instruction") or "").strip()
        url_box_text  = (body.get("url_box") or "").strip()

        if not (message_raw and prev_reply):
            return jsonify({"error":"Missing fields."}), 400

        message     = remove_personal_info(message_raw)
        instruction = remove_personal_info(instruction_raw)
        url_map     = parse_url_box(url_box_text)

        prompt = f"""
Revise the admissions reply below according to the instruction.

Instruction: {instruction}

Parent enquiry:
\"\"\"{message}\"\"\"

Current reply (Markdown):
\"\"\"{prev_reply}\"\"\"

Return only the revised reply in Markdown.
""".strip()

        new_md = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.4
        ).choices[0].message.content.strip()
        new_md = clean_gpt_email_output(new_md)

        # 🔗 Insert anchor links (if provided)
        new_md_linked = insert_links(new_md, url_map)

        return jsonify({"reply": markdown_to_html(new_md_linked)})

    except Exception as e:
        print(f"❌ REVISION ERROR: {e}")
        return jsonify({"error":"Revision failed."}), 500

# ──────────────────────────────
# 💾  POST /save-standard
# ──────────────────────────────
@app.route("/save-standard", methods=["POST"])
def save_standard():
    try:
        body = request.get_json(force=True)
        msg_raw = (body.get("message") or "").strip()
        reply   = (body.get("reply")   or "").strip()

        if not (msg_raw and reply):
            return jsonify({"status":"error","message":"Missing fields"}), 400

        msg_redacted = remove_personal_info(msg_raw)

        # append & persist
        record = {"timestamp":datetime.now().isoformat(),"message":msg_redacted,"reply":reply}
        path="standard_responses.json"
        data=[]
        if os.path.exists(path):
            with open(path,"r") as f: data=json.load(f)
        data.append(record)
        with open(path,"w") as f: json.dump(data,f,indent=2)

        # in-memory
        standard_messages.append(msg_redacted)
        standard_embeddings.append(embed_text(msg_redacted))
        standard_replies.append(reply)

        return jsonify({"status":"ok"})
    except Exception as e:
        print(f"❌ SAVE ERROR: {e}")
        return jsonify({"status":"error","message":"Save failed"}),500

# ──────────────────────────────
# 🌐  SERVE FRONT END
# ──────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

# ──────────────────────────────
# ▶️  MAIN
# ──────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
