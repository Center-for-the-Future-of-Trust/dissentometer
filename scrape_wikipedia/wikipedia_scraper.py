#!/usr/bin/env python3
# wiki_scraper_hardened.py

import argparse
import asyncio
import os
import sqlite3
import unicodedata
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Iterable

import aiohttp
import pycountry

API_URL_TMPL = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT   = "WikipediaScraper/4.0 (contact: steph.buon@gmail.com)"
TIMEOUT_S    = 30
SLEEP_BETWEEN_CALLS = 0.05

EN_ROOT_CATEGORIES = [
    "Historical objects",
    "History of sport",
    "History of ideologies",
]

# -------------------- utils --------------------
def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in " _-()" else "_" for c in name)

def norm_title(t: str) -> str:
    # normalize unicode and trim whitespace
    return unicodedata.normalize("NFC", (t or "")).strip()

def short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def lang_name(code: str) -> str:
    try:
        lang = pycountry.languages.get(alpha_2=code) or pycountry.languages.get(alpha_3=code)
        return lang.name if lang and hasattr(lang, "name") else code
    except Exception:
        return code

def chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

# -------------------- SQLite ledger (dedup + resume) --------------------
class Ledger:
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout=5000;")  # avoid 'database is locked'
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                lang TEXT NOT NULL,
                title TEXT NOT NULL,
                PRIMARY KEY (lang, title)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                lang TEXT NOT NULL,
                title TEXT NOT NULL,
                PRIMARY KEY (lang, title)
            )
        """)
        self.conn.commit()

    def claim_category(self, lang: str, title: str) -> bool:
        title = norm_title(title)
        cur = self.conn.cursor()
        try:
            cur.execute("INSERT OR IGNORE INTO categories(lang,title) VALUES(?,?)", (lang, title))
            self.conn.commit()
            return cur.rowcount == 1
        finally:
            cur.close()

    def claim_pages_many(self, rows: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Insert many page claims; return only the newly-claimed ones."""
        rows = [(lg, norm_title(ttl)) for lg, ttl in rows]
        cur = self.conn.cursor()
        try:
            cur.execute("BEGIN")
            cur.executemany("INSERT OR IGNORE INTO pages(lang,title) VALUES(?,?)", rows)
            self.conn.commit()
            # We can't directly know which inserted; re-check is cheap for small batches:
            claimed = []
            for lg, ttl in rows:
                r = cur.execute("SELECT 1 FROM pages WHERE lang=? AND title=? LIMIT 1", (lg, ttl)).fetchone()
                if r:
                    claimed.append((lg, ttl))
            return claimed
        finally:
            cur.close()

    def claim_page(self, lang: str, title: str) -> bool:
        title = norm_title(title)
        cur = self.conn.cursor()
        try:
            cur.execute("INSERT OR IGNORE INTO pages(lang,title) VALUES(?,?)", (lang, title))
            self.conn.commit()
            return cur.rowcount == 1
        finally:
            cur.close()

    def close(self):
        self.conn.close()

# -------------------- HTTP client with per-host semaphores --------------------
class WikiClient:
    def __init__(self, global_concurrency: int, per_host: int):
        self.global_sema = asyncio.Semaphore(global_concurrency)
        self.host_semas = defaultdict(lambda: asyncio.Semaphore(per_host))
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=TIMEOUT_S, sock_read=TIMEOUT_S)
        self.session = aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": USER_AGENT})
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def query(self, lang: str, params: Dict, retries: int = 3) -> Dict:
        assert self.session is not None
        url = API_URL_TMPL.format(lang=lang)
        host = f"{lang}.wikipedia.org"
        backoff = 1.0
        for attempt in range(retries):
            async with self.global_sema, self.host_semas[host]:
                try:
                    async with self.session.get(url, params=params) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        await asyncio.sleep(SLEEP_BETWEEN_CALLS)
                        return data
                except Exception:
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(backoff)
                    backoff *= 2

# -------------------- API wrappers --------------------
async def get_category_langlinks(client: WikiClient, lang: str, category_without_ns: str) -> List[Tuple[str, str]]:
    params = {"action": "query", "format": "json", "prop": "langlinks",
              "titles": f"Category:{category_without_ns}", "lllimit": "max"}
    data = await client.query(lang, params)
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {}) if pages else {}
    out = []
    for ll in page.get("langlinks", []) or []:
        lg, ttl = ll.get("lang"), norm_title(ll.get("*"))
        if lg and ttl:
            out.append((lg, ttl))
    return out

async def get_category_members(client: WikiClient, lang: str, category_full_title: str, want_subcats: bool) -> List[Dict]:
    members, cont = [], None
    cmtype = "subcat" if want_subcats else "page"
    ns = "14" if want_subcats else "0"
    while True:
        params = {"action": "query", "format": "json", "list": "categorymembers",
                  "cmtitle": norm_title(category_full_title), "cmtype": cmtype, "cmlimit": "500", "cmnamespace": ns}
        if cont:
            params["cmcontinue"] = cont
        data = await client.query(lang, params)
        items = data.get("query", {}).get("categorymembers", []) or []
        # normalize titles
        for it in items:
            if "title" in it:
                it["title"] = norm_title(it["title"])
        members.extend(items)
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return members

async def batch_pages_info(client: WikiClient, lang: str, titles: List[str], batch_size: int) -> Dict[str, Dict]:
    """
    Batch fetch `extracts|langlinks` for many titles in a single request.
    Returns: {title: {"extract": str, "langlinks": [(lg, localized_title), ...]}}
    """
    result: Dict[str, Dict] = {}
    for group in chunked(titles, batch_size):
        params = {
            "action": "query", "format": "json",
            "prop": "extracts|langlinks",
            "redirects": 1,          # ← follow redirects so we get canonical target content
            "explaintext": True,
            "lllimit": "max",
            "titles": "|".join(group)
        }
        data = await client.query(lang, params)
        pages = data.get("query", {}).get("pages", {}) or {}
        for pg in pages.values():
            title = norm_title(pg.get("title"))
            if not title:
                continue
            extract = pg.get("extract", "") or ""
            links = pg.get("langlinks", []) or []
            ll = []
            for x in links:
                lg, ttl = x.get("lang"), norm_title(x.get("*"))
                if lg and ttl:
                    ll.append((lg, ttl))
            result[title] = {"extract": extract, "langlinks": ll}
    return result

async def batch_extracts(client: WikiClient, lang: str, titles: List[str], batch_size: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for group in chunked(titles, batch_size):
        params = {
            "action": "query", "format": "json",
            "prop": "extracts",
            "redirects": 1,          # ← follow redirects here too
            "explaintext": True,
            "titles": "|".join(group)
        }
        data = await client.query(lang, params)
        pages = data.get("query", {}).get("pages", {}) or {}
        for pg in pages.values():
            title = norm_title(pg.get("title"))
            if title is None:
                continue
            out[title] = pg.get("extract", "") or ""
    return out

# -------------------- File I/O (write-once atomic) --------------------
def atomic_write_text_once(path: Path, text: str):
    """Write only if file does not exist; atomic replace with Windows retry."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.replace(tmp, path)
    except PermissionError:
        # Windows AV/OneDrive lock — retry a few times
        import time
        for _ in range(5):
            time.sleep(0.2)
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                continue
        else:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            raise

# -------------------- Crawl logic --------------------
async def process_pages_batch(
    client: WikiClient,
    ledger: Ledger,
    base_lang: str,
    titles: List[str],
    out_dir: Path,
    depth: int,
    allowed_langs: Set[str],         # empty -> all
    base_batch: int,
    variants_batch: int
):
    # Claim base pages up-front in a single transaction; only process newly-claimed
    claimed = []
    for t in titles:
        if ledger.claim_page(base_lang, t):
            claimed.append(t)
    if not claimed:
        return

    info = await batch_pages_info(client, base_lang, claimed, base_batch)

    # Save base extracts and gather variant claims
    variants_by_lang: Dict[str, List[Tuple[str, Path]]] = {}  # lang -> [(localized_title, entry_dir)]
    for title in claimed:
        data = info.get(title, {})
        extract = data.get("extract", "")
        # Disambiguate article directory with short hash to avoid sanitize collisions
        entry_dir = out_dir / f"depth_{depth}" / f"{sanitize(title)}__{short_hash(title)}"

        if extract.strip():
            fpath = entry_dir / f"{sanitize(lang_name(base_lang))}__{base_lang}.txt"
            atomic_write_text_once(fpath, extract)
            print(f"✔ Saved: {title} [{base_lang}] → depth {depth}")

        for lg, localized in data.get("langlinks", []):
            if allowed_langs and lg not in allowed_langs:
                continue
            if ledger.claim_page(lg, localized):
                variants_by_lang.setdefault(lg, []).append((localized, entry_dir))

    # Batch fetch & save variants per language
    tasks = []
    for lg, items in variants_by_lang.items():
        titles_lg = [t for (t, _dir) in items]
        tasks.append(_save_variants_for_language(client, lg, items, titles_lg, variants_batch, depth))
    await asyncio.gather(*tasks)

async def _save_variants_for_language(
    client: WikiClient,
    lg: str,
    items: List[Tuple[str, Path]],
    titles: List[str],
    batch_size: int,
    depth: int
):
    extracts = await batch_extracts(client, lg, titles, batch_size)
    for localized, entry_dir in items:
        text = extracts.get(localized, "")
        if not text.strip():
            continue
        fname = f"{sanitize(lang_name(lg))}__{lg}.txt"
        atomic_write_text_once(entry_dir / fname, text)
        print(f"✔ Saved: {localized} [{lg}] → depth {depth}")

async def crawl_category(
    client: WikiClient,
    ledger: Ledger,
    lang: str,
    category_full_title: str,
    out_dir: Path,
    depth: int,
    max_depth: int,
    allowed_langs: Set[str],
    base_batch: int,
    variants_batch: int
):
    # Dedup categories across runs/workers
    if not ledger.claim_category(lang, category_full_title):
        return

    indent = "  " * (depth - 1)
    print(f"{indent}→ depth={depth}, scraping {category_full_title} [{lang}]")

    # Pages at this level
    pages = await get_category_members(client, lang, category_full_title, want_subcats=False)
    page_titles = [norm_title(it["title"]) for it in pages if it.get("ns") == 0]

    # Process in batches (batched base extracts+langlinks)
    for group in chunked(page_titles, base_batch):
        await process_pages_batch(client, ledger, lang, list(group), out_dir, depth,
                                  allowed_langs, base_batch, variants_batch)

    # Recurse into subcategories
    if depth < max_depth:
        subcats = await get_category_members(client, lang, category_full_title, want_subcats=True)
        tasks = []
        for it in subcats:
            if it.get("ns") != 14:
                continue
            sub_full = norm_title(it["title"])  # keep "Category:..."
            sub_dir  = out_dir / sanitize(sub_full.split(":", 1)[-1])
            tasks.append(crawl_category(
                client, ledger, lang, sub_full, sub_dir, depth + 1, max_depth,
                allowed_langs, base_batch, variants_batch
            ))
        await asyncio.gather(*tasks)

async def build_roots(client: WikiClient) -> List[Tuple[str, str, str]]:
    roots: List[Tuple[str, str, str]] = []
    for root in EN_ROOT_CATEGORIES:
        root_norm = norm_title(root)
        en_full = f"Category:{root_norm}"
        roots.append(("en", en_full, root_norm))
        try:
            links = await get_category_langlinks(client, "en", root_norm)
        except Exception as e:
            print(f"⚠ langlinks failed for root '{root_norm}': {e}")
            links = []
        for lg, localized in links:
            roots.append((lg, f"Category:{localized}", root_norm))
    return roots

# -------------------- main --------------------
async def amain(
    output_dir: Path,
    db_path: Path,
    global_concurrency: int,
    per_host: int,
    max_depth: int,
    langs: str,
    base_batch: int,
    variants_batch: int
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(db_path)

    allowed_langs: Set[str] = set()
    if langs.strip().lower() != "all":
        allowed_langs = {s.strip().lower() for s in langs.split(",") if s.strip()}

    try:
        async with WikiClient(global_concurrency, per_host) as client:
            roots = await build_roots(client)

            print("Will scrape these category roots:")
            for lg, cat_full, root_label in roots:
                print(f" - [{lg}] {cat_full} (root: {root_label})")

            tasks = []
            for lg, cat_full, root_label in roots:
                out_dir = output_dir / sanitize(root_label) / lg / sanitize(cat_full.split(":", 1)[-1])
                tasks.append(crawl_category(
                    client, ledger, lg, cat_full, out_dir, depth=1, max_depth=max_depth,
                    allowed_langs=allowed_langs, base_batch=base_batch, variants_batch=variants_batch
                ))
            await asyncio.gather(*tasks)
    finally:
        ledger.close()

    print(f"\n✅ All results saved under: {output_dir}")
    print(f"🧾 Resume database: {db_path}")

def main():
    parser = argparse.ArgumentParser(description="Parallel, batched, hardened Wikipedia category scraper")
    parser.add_argument("--output", type=str, default=str(Path.home() / "Desktop" / "Wiki_Scrape"),
                        help="Output directory (default: ~/Desktop/Wiki_Scrape)")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to resume/dedup SQLite DB (default: <output>/crawler_state.sqlite)")
    parser.add_argument("--concurrency", type=int, default=12,
                        help="Global max concurrent HTTP requests (default: 12)")
    parser.add_argument("--per-host", type=int, default=4,
                        help="Per-host concurrent requests (default: 4)")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Max subcategory depth (default: 4)")
    parser.add_argument("--langs", type=str, default="all",
                        help='Variant languages to save, e.g. "en,fr,es" or "all" (default: all)')
    parser.add_argument("--batch-size", type=int, default=40,
                        help="Base pages batch size for extracts|langlinks (default: 40)")
    parser.add_argument("--variants-batch-size", type=int, default=50,
                        help="Variant extracts batch size per language (default: 50)")
    args = parser.parse_args()

    out = Path(args.output)
    db_path = Path(args.db) if args.db else out / "crawler_state.sqlite"

    asyncio.run(amain(
        output_dir=out,
        db_path=db_path,
        global_concurrency=args.concurrency,
        per_host=args.per_host,
        max_depth=args.max_depth,
        langs=args.langs,
        base_batch=args.batch_size,
        variants_batch=args.variants_batch_size
    ))

if __name__ == "__main__":
    main()


# Default (All langauge variants), depth=4
#python wiki_scraper_hardened.py --output /data/wiki_scrape

# Increase parallelism, keep hosts fair; slightly larger batches
#python wiki_scraper_hardened.py --output /data/=wiki_scrape --concurrency 16 --per-host 5 --batch-size 60

# Only English + German variants; shallow crawl
#python wiki_scraper_hardened.py --langs en,de --max-depth 2

# Resume later (uses the same SQLite file inside the output folder)
#python wiki_scraper_hardened.py --output /data/wiki_scrape



## Mine 
#python wikipedia_scraper.py --output /wikipedia_articles --concurrency 16 --per-host 5 --batch-size 60