import json, pickle, re, numpy as np
import datetime
import threading
import os
import difflib

STANDARD_RESPONSES_FILE = "standard_responses.json"
LOCK = threading.Lock()


from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from url_mapping import URL_MAPPING, URL_ALIASES, BAD_ANCHORS

from static_qa import STATIC_QA


def replace_link_keys(answer):
    def replacer(match):
        anchor_text = match.group(1).strip()

        # 🔁 Replace bad anchor text with clean alias
        if anchor_text.lower() in BAD_ANCHORS:
            anchor_text = BAD_ANCHORS[anchor_text.lower()]

        # ✅ Direct match
        if anchor_text in URL_MAPPING:
            url = URL_MAPPING[anchor_text]
        else:
            # 🔁 Try exact alias
            alias_key = URL_ALIASES.get(anchor_text.lower())
            if alias_key and alias_key in URL_MAPPING:
                url = URL_MAPPING[alias_key]
            else:
                # 🔍 NEW: Try fuzzy match as last resort
                from difflib import get_close_matches

                # Try fuzzy against URL_MAPPING keys
                candidates = list(URL_MAPPING.keys())
                close = get_close_matches(anchor_text, candidates, n=1, cutoff=0.75)
                if close:
                    url = URL_MAPPING[close[0]]
                else:
                    url = "#"

        print(f"🔍 Anchor: '{anchor_text}' → {url}")
        return f"[{anchor_text}]({url})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: replacer(m), answer)



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

import datetime
import re

def filter_past_dates(reply, today=None):
    """Remove lines with event dates before today (in formats like '12 June' or '12 June 2025')."""
    if not today:
        today = datetime.datetime.now()
    date_pattern = re.compile(r'(\d{1,2} [A-Z][a-z]+(?: \d{4})?)')
    lines = reply.split('\n')
    filtered = []
    for line in lines:
        match = date_pattern.search(line)
        if match:
            try:
                date_str = match.group(1)
                # Try parsing '12 June 2025' and '12 June'
                for fmt in ('%d %B %Y', '%d %B'):
                    try:
                        dt = datetime.datetime.strptime(date_str, fmt)
                        if fmt == '%d %B':
                            dt = dt.replace(year=today.year)
                        if dt.date() >= today.date():
                            filtered.append(line)
                        # else: skip line
                        break
                    except Exception:
                        continue
            except Exception:
                filtered.append(line)
        else:
            filtered.append(line)
    return '\n'.join(filtered)

def load_standards():
    if not os.path.exists(STANDARD_RESPONSES_FILE):
        return []
    with open(STANDARD_RESPONSES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_standards(data):
    with LOCK:
        with open(STANDARD_RESPONSES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def find_best_static_match(question):
        question_lower = question.lower().strip()

    # Match against main questions
    for q_key, q_data in STATIC_QA.items():
        if difflib.get_close_matches(question_lower, [q_key], n=1, cutoff=0.85):
            return STATIC_QA[q_key]

    # Match against tag list if no direct match
    for q_key, q_data in STATIC_QA.items():
        for tag in q_data.get("tags", []):
            if difflib.get_close_matches(question_lower, [tag], n=1, cutoff=0.85):
                return STATIC_QA[q_key]

    return None

# ──────────────────────────────
# 📚  LOAD SCHOOL KB
# ──────────────────────────────
with open("embeddings/metadata.pkl", "rb") as f:
    kb = pickle.load(f)
    doc_embeddings = np.array(kb["embeddings"])
    metadata       = kb["metadata"]
print(f"✅ Loaded {len(metadata)} knowledge chunks.")

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
def reply():
    try:
        body = request.get_json(force=True)
        question_raw = (body.get("message") or "").strip()
        instruction_raw = (body.get("instruction") or "").strip()

        # 🔒 sanitise
        question    = remove_personal_info(question_raw)
        instruction = remove_personal_info(instruction_raw)

        if not question:
            return jsonify({"error":"No message received."}), 400

        # ──────────────
        # 0) STATIC QA
        # ──────────────
            match = find_best_static_match(question)
            if match:
                safe_answer = replace_link_keys(match["answer"])
                reply_html = markdown_to_html(safe_answer)
                return jsonify({
                    "reply": reply_html,
                    "sentiment_score": 10,
                    "strategy_explanation": "Matched static QA using fuzzy logic."
                })


        # Continue as before with vector match, etc.
        q_vec = embed_text(question)

        # 1) pre-approved template?
        matched = check_standard_match(q_vec)
        if matched:
            return jsonify({
                "reply": matched,
                "sentiment_score": 10,
                "strategy_explanation": "Used approved template."
            })

        # ...rest of your code unchanged...

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
        reply_md = filter_past_dates(reply_md)
        reply_md = replace_link_keys(reply_md)  # 🔹 Inject correct URLs from mapping
        print("🔗 Link-mapped Markdown:\n", reply_md)
        reply_html = markdown_to_html(reply_md)


        return jsonify({
            "reply": reply_html,
            "sentiment_score": score,
            "strategy_explanation": strat
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
        message_raw  = (body.get("message") or "").strip()
        prev_reply   = (body.get("previous_reply") or "").strip()  # already HTML
        instruction_raw = (body.get("instruction") or "").strip()

        if not (message_raw and prev_reply and instruction_raw):
            return jsonify({"error":"Missing fields."}), 400

        message    = remove_personal_info(message_raw)
        instruction= remove_personal_info(instruction_raw)

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
        new_md = replace_link_keys(new_md)  # 🔧 Map anchors to correct URLs
        # Optional (recommended): strip hallucinated external links
        new_md = re.sub(
            r"\[([^\]]+)\]\((https?://[^)]+)\)",
            lambda m: m.group(0) if m.group(2).startswith("https://www.bassetths.org.uk") else f"[{m.group(1)}](#)",
            new_md
        )

        return jsonify({"reply": markdown_to_html(new_md)})


    except Exception as e:
        print(f"❌ REVISION ERROR: {e}")
        return jsonify({"error":"Revision failed."}), 500

# ──────────────────────────────
# 💾  POST /save-standard
# ──────────────────────────────
@app.route("/save-standard", methods=["POST"])
def save_standard():
    data = request.json
    enquiry = data.get("message", "").strip()
    reply = data.get("reply", "").strip()
    urls = data.get("urls", [])

    if not enquiry or not reply:
        return jsonify({"error": "Missing enquiry or reply"}), 400

    standards = load_standards()
    updated = False
    for item in standards:
        if item["enquiry"].lower() == enquiry.lower():
            item["reply"] = reply
            item["urls"] = urls
            updated = True
            break
    if not updated:
        standards.append({"enquiry": enquiry, "reply": reply, "urls": urls})

    save_standards(standards)
    return jsonify({"status": "saved"})

# ──────────────────────────────
# 🌐  SERVE FRONT END
# ──────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

# ──────────────────────────────
# ▶️  MAIN
# ──────────────────────────────

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT not set
    app.run(host="0.0.0.0", port=port)
