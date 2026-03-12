#!/usr/bin/env python3
"""Web search helper — agent can call this from bash to look up ML techniques."""
import html
import re
import sys
import urllib.parse
import urllib.request


def search(query: str, num_results: int = 5) -> list[dict]:
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=10)
    text = resp.read().decode("utf-8", errors="replace")

    results = []
    # DuckDuckGo lite HTML has result__a (title link) and result__snippet
    blocks = re.split(r'class="result\s', text)
    for block in blocks[1:]:
        title_m = re.search(r'class="result__a"[^>]*>(.*?)</a>', block, re.DOTALL)
        snippet_m = re.search(r'class="result__snippet"[^>]*>(.*?)</(?:span|td)', block, re.DOTALL)
        url_m = re.search(r'class="result__url"[^>]*>(.*?)</a>', block, re.DOTALL)
        if title_m:
            title = re.sub(r"<.*?>", "", html.unescape(title_m.group(1))).strip()
            snippet = re.sub(r"<.*?>", "", html.unescape(snippet_m.group(1))).strip() if snippet_m else ""
            link = re.sub(r"<.*?>", "", html.unescape(url_m.group(1))).strip() if url_m else ""
            if title:
                results.append({"title": title, "url": link, "snippet": snippet})
        if len(results) >= num_results:
            break
    return results


def fetch_text(url: str, max_chars: int = 3000) -> str:
    """Fetch a URL and extract readable text (strips HTML tags)."""
    if not url.startswith("http"):
        url = "https://" + url
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=10)
    raw = resp.read().decode("utf-8", errors="replace")
    # Strip scripts, styles, then tags
    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 search.py <query>          # search the web")
        print("  python3 search.py --fetch <url>     # fetch and read a page")
        sys.exit(1)

    if sys.argv[1] == "--fetch":
        url = sys.argv[2] if len(sys.argv) > 2 else ""
        if not url:
            print("Error: provide a URL")
            sys.exit(1)
        print(fetch_text(url))
    else:
        query = " ".join(sys.argv[1:])
        results = search(query)
        if not results:
            print("No results found.")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']}")
            if r["url"]:
                print(f"   {r['url']}")
            if r["snippet"]:
                print(f"   {r['snippet']}")
            print()
