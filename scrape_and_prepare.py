import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json

BASE_URL = "https://www.bassetths.org.uk"
VISITED = set()
CRAWLED_DATA = []

EXCLUDE_PATTERNS = ["wp-login", "login", "contact-form", "admin", "cookie", "feed", "rss", "mailto:", ".ics"]

def is_valid_url(href):
    if not href:
        return False
    parsed = urlparse(href)
    if parsed.netloc and parsed.netloc != urlparse(BASE_URL).netloc:
        return False
    if href.startswith("javascript:") or any(x in href for x in EXCLUDE_PATTERNS):
        return False
    return True

def clean_text(soup):
    for tag in soup(["script", "style", "header", "footer", "nav", "noscript", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=' ', strip=True)
    return re.sub(r"\s+", " ", text)

def crawl(url, depth=0, max_depth=2):
    if url in VISITED or depth > max_depth:
        return
    try:
        response = requests.get(url, timeout=10)
        if not response.ok or "text/html" not in response.headers.get("Content-Type", ""):
            return
        soup = BeautifulSoup(response.text, "html.parser")
        text = clean_text(soup)
        if len(text) < 150:
            return
        CRAWLED_DATA.append({
            "url": url,
            "text": text,
            "type": "html"
        })
        VISITED.add(url)
        for link in soup.find_all("a", href=True):
            abs_url = urljoin(url, link['href'].split('#')[0])
            if is_valid_url(abs_url):
                crawl(abs_url, depth + 1, max_depth)
    except Exception as e:
        print(f"⚠️ Failed to crawl {url}: {e}")

# Crawl and save
crawl(BASE_URL)
os.makedirs("output", exist_ok=True)
with open("output/clean_chunks.json", "w", encoding="utf-8") as f:
    json.dump(CRAWLED_DATA, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(CRAWLED_DATA)} pages.")
