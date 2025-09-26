#!/usr/bin/env python3
# wikipedia_scraper_async_seriallike.py
# Async implementation with the SAME semantics as the serial script:
# - langlinks: no llprop; read legacy ll["*"]; keep ALL languages
# - global in-memory dedup (like the serial "visited" sets)
# - nested folder layout
# - robust HTTP + skip non-JSON responses (e.g., incubator redirects)

import argparse
import asyncio
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import aiohttp
import pycountry

API_URL_TMPL = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT   = "WikipediaScraper/5.0 (contact: you@example.org)"
TIMEOUT_S    = 30
SLEEP_BETWEEN_CALLS = 0.05
MAX_DEPTH_DEFAULT   = 4

# Root categories (like your later runs)
EN_ROOT_CATEGORIES = [
    "Historical objects",
    "History of sports",
    "History of ideologies",
]

# ------------- filename helpers -------------
try:
    from unidecode import unidecode
except Exception:
    unidecode = None

ASCII_FILENAMES = False  # set by CLI

def translit_ascii(s: str) -> str:
    if unidecode:
        return unidecode(s)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def sanitize(name: str) -> str:
    raw = unicodedata.normalize("NFC", name or "")
    s = translit_ascii(raw) if ASCII_FILENAMES else raw
    s = "".join(c if (c.isalnum() or c in " _-()") else "_" for c in s)
    s = " ".join(s.split()).strip("._ ")
    return s or "untitled"

def get_language_name(code: str) -> str:
    try:
        lang = pycountry.languages.get(alpha_2=code) or pycountry.languages.get(alpha_3=code)
        return lang.name if lang and getattr(lang, "name", None) else code
    except Exception:
        return code

# ------------- global in-memory dedup (serial-like) -------------
visited_pages: Set[Tuple[str, str]] = set()       # (lang, page_title)
visited_categories: Set[Tuple[str, str]] = set()  # (lang, "Category:Title")
visit_lock = asyncio.Lock()

# ------------- HTTP client -------------
class WikiClient:
    def __init__(self, global_concurrency: int, per_host: int):
        self.global_sema = asyncio.Semaphore(global_concurrency)
        self.host_semas: Dict[str, asyncio.Semaphore] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.per_host = per_host

    def _host_sema(self, host: str) -> asyncio.Semaphore:
        sema = self.host_semas.get(host)
        if sema is None:
            sema = self.host_semas[host] = asyncio.Semaphore(self.per_host)
        return sema

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=TIMEOUT_S, sock_read=TIMEOUT_S)
        self.session = aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": USER_AGENT})
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def query(self, lang: str, params: Dict, retries: int = 3) -> Optional[Dict]:
        """Return JSON dict or None; never raise to the top (keeps job running)."""
        assert self.session is not None
        url = API_URL_TMPL.format(lang=lang)
        host = f"{lang}.wikipedia.org"
        backoff = 1.0
        for attempt in range(retries):
            async with self.global_sema, self._host_sema(host):
                try:
                    async with self.session.get(url, params=params, allow_redirects=True) as resp:
                        # If a non-JSON HTML redirect (e.g., incubator), skip gracefully
                        ctype = (resp.headers.get("Content-Type") or "").lower()
                        text = await resp.text()
                        if "application/json" not in ctype:
                            # Try JSON anyway; if fails, warn and return None
                            try:
                                data = json.loads(text)
                            except Exception:
                                print(f"? non-JSON from {resp.url} (Content-Type={ctype}) ? skipping")
                                await asyncio.sleep(SLEEP_BETWEEN_CALLS)
                                return None
                        else:
                            data = json.loads(text)
                        await asyncio.sleep(SLEEP_BETWEEN_CALLS)
                        return data
                except Exception as e:
                    if attempt == retries - 1:
                        print(f"? query failed for {lang} {params.get('action')}:{params.get('prop') or params.get('list')} ? {e}")
                        return None
                    await asyncio.sleep(backoff)
                    backoff *= 2

# ------------- API wrappers (serial semantics) -------------
async def get_category_langlinks(client: WikiClient, lang: str, category_without_ns: str) -> List[Tuple[str, str]]:
    """
    SERIAL semantics: do NOT use llprop; read legacy ll["*"]
    Returns [(lang_code, localized_category_title_without_ns)]
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "langlinks",
        "titles": f"Category:{category_without_ns}",
        "lllimit": "max",
    }
    data = await client.query(lang, params)
    if not data:
        return []
    pages = data.get("query", {}).get("pages", {}) or {}
    page = next(iter(pages.values()), {}) if pages else {}
    out = []
    for ll in page.get("langlinks", []) or []:
        lg = ll.get("lang")
        ttl = ll.get("*")  # legacy title field (serial uses this)
        if lg and ttl:
            out.append((lg, ttl))
    return out

async def get_category_members(client: WikiClient, lang: str, category_without_ns: str, want_subcats: bool) -> List[str]:
    """
    SERIAL semantics:
      - if want_subcats=True: return subcat titles WITHOUT "Category:" prefix
      - if want_subcats=False: return page (article) titles
    """
    members, cont = [], None
    cmtype = "subcat" if want_subcats else "page"
    ns = "14" if want_subcats else "0"
    while True:
        params = {
            "action": "query", "format": "json", "list": "categorymembers",
            "cmtitle": f"Category:{category_without_ns}",
            "cmtype": cmtype, "cmlimit": "500", "cmnamespace": ns
        }
        if cont:
            params["cmcontinue"] = cont
        data = await client.query(lang, params)
        if not data:
            break
        items = data.get("query", {}).get("categorymembers", []) or []
        if want_subcats:
            members += [it["title"].split("Category:", 1)[-1] for it in items if "title" in it]
        else:
            members += [it["title"] for it in items if "title" in it]
        cont = (data.get("continue", {}) or {}).get("cmcontinue")
        if not cont:
            break
    return members

async def get_page_langlinks(client: WikiClient, lang: str, title: str) -> List[Tuple[str, str]]:
    """SERIAL semantics: no llprop; read ll['*']"""
    params = {
        "action": "query", "format": "json",
        "prop": "langlinks",
        "titles": title, "lllimit": "max"
    }
    data = await client.query(lang, params)
    if not data:
        return []
    pages = data.get("query", {}).get("pages", {}) or {}
    page = next(iter(pages.values()), {}) if pages else {}
    out = []
    for ll in page.get("langlinks", []) or []:
        lg = ll.get("lang")
        ttl = ll.get("*")
        if lg and ttl:
            out.append((lg, ttl))
    return out

async def fetch_extract(client: WikiClient, lang: str, title: str) -> str:
    """SERIAL semantics: extracts, no redirects flag."""
    params = {
        "action": "query", "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title
    }
    data = await client.query(lang, params)
    if not data:
        return ""
    pages = data.get("query", {}).get("pages", {}) or {}
    page = next(iter(pages.values()), {}) if pages else {}
    return page.get("extract", "") or ""

# ------------- saving -------------
async def save_article_and_variants(
    client: WikiClient,
    base_lang: str,
    title: str,
    out_dir: Path,
):
    """SERIAL semantics: article dir = sanitized title; files named by LanguageName.txt."""
    # Build variants exactly like serial: base + all page langlinks
    try:
        variants = [(base_lang, title)] + await get_page_langlinks(client, base_lang, title)
    except Exception as e:
        print(f"? langlinks failed for {title} [{base_lang}]: {e}")
        variants = [(base_lang, title)]

    entry_dir = out_dir / sanitize(title)
    entry_dir.mkdir(parents=True, exist_ok=True)

    for lg, localized_title in variants:
        async with visit_lock:
            key = (lg, localized_title)
            if key in visited_pages:
                continue
            visited_pages.add(key)

        try:
            text = await fetch_extract(client, lg, localized_title)
            if not text.strip():
                continue
            fname = sanitize(get_language_name(lg)) + ".txt"  # SERIAL-style filenames
            (entry_dir / fname).write_text(text, encoding="utf-8")
            print(f"? Saved: {localized_title} [{lg}]")
        except Exception as e:
            print(f"? fetch/save failed for {localized_title} [{lg}]: {e}")

# ------------- crawl -------------
async def scrape_category(
    client: WikiClient,
    lang: str,
    category_without_ns: str,
    out_dir: Path,
    depth: int,
    max_depth: int
):
    print(f"{'  '*(depth-1)}? depth={depth}, scraping Category:{category_without_ns} [{lang}]")

    # Dedup category (serial style)
    async with visit_lock:
        ckey = (lang, f"Category:{category_without_ns}")
        if ckey in visited_categories:
            return
        visited_categories.add(ckey)

    # Pages (articles)
    titles = await get_category_members(client, lang, category_without_ns, want_subcats=False)
    # Run saves concurrently for this level
    page_tasks = [save_article_and_variants(client, lang, t, out_dir) for t in titles]
    await asyncio.gather(*page_tasks)

    # Recurse into subcategories
    if depth < max_depth:
        subcats = await get_category_members(client, lang, category_without_ns, want_subcats=True)
        sub_tasks = []
        for subcat in subcats:
            sub_dir = out_dir / sanitize(subcat)
            sub_tasks.append(scrape_category(client, lang, subcat, sub_dir, depth + 1, max_depth))
        await asyncio.gather(*sub_tasks)

async def amain(
    output_dir: Path,
    concurrency: int,
    per_host: int,
    max_depth: int
):
    output_dir.mkdir(parents=True, exist_ok=True)

    async with WikiClient(concurrency, per_host) as client:
        # Build root category list like the serial script: English + its langlinks
        roots: List[Tuple[str, str]] = []
        for root in EN_ROOT_CATEGORIES:
            root_norm = root
            roots.append(("en", root_norm))
            try:
                links = await get_category_langlinks(client, "en", root_norm)
            except Exception as e:
                print(f"? langlinks failed for root '{root_norm}': {e}")
                links = []
            roots.extend(links)  # [(lg, localized_category_title_without_ns), ...]

        print("Will scrape these category entries:")
        for lg, ttl in roots:
            print(f" - [{lg}] Category:{ttl}")

        # Kick off one crawl per (lang, localized_category) like serial
        tasks = []
        for lg, ttl in roots:
            out_dir = output_dir / lg / sanitize(ttl)
            tasks.append(scrape_category(client, lg, ttl, out_dir, depth=1, max_depth=max_depth))
        await asyncio.gather(*tasks)

def main():
    global ASCII_FILENAMES

    p = argparse.ArgumentParser(description="Async Wikipedia category scraper with serial behavior")
    p.add_argument("--output", type=str, default=str(Path.home() / "Desktop" / "wikipedia_articles"),
                   help="Output directory (default: ~/Desktop/Wiki_Scrape)")
    p.add_argument("--concurrency", type=int, default=12,
                   help="Global concurrent HTTP requests (default: 12)")
    p.add_argument("--per-host", type=int, default=4,
                   help="Concurrent requests per host (default: 4)")
    p.add_argument("--max-depth", type=int, default=MAX_DEPTH_DEFAULT,
                   help=f"Max subcategory depth (default: {MAX_DEPTH_DEFAULT})")
    p.add_argument("--ascii-filenames", action="store_true",
                   help="Transliterate folder and file names to ASCII")
    args = p.parse_args()

    ASCII_FILENAMES = bool(args.ascii_filenames)
    out = Path(args.output)

    asyncio.run(amain(
        output_dir=out,
        concurrency=args.concurrency,
        per_host=args.per_host,
        max_depth=args.max_depth
    ))

if __name__ == "__main__":
    main()


