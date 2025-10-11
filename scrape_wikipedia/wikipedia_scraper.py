#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# wikipedia_scraper_async_seriallike.py
#
# Features:
# - Robust HTTP:
#   * Global min-interval throttle (GLOBAL_MIN_INTERVAL)
#   * Honors 429 Retry-After with jitter and MediaWiki maxlag
#   * Occasional throttle notices to stderr and structured entries in _skipped.tsv
# - Batching for base-language extracts (--batch, safe 50 per request)
# - Hash-based sharding across jobs (--shards N --shard-id K)
# - Global in-memory dedup of pages and categories
# - Keeps non-JSON responses as .raw.* sidecars
# - Logs skips to _skipped.tsv and writes a final _summary.txt
#
# Safe starter flags:
#   --concurrency 4 --per-host 1 --batch 50 --max-depth 3
#
# Split across 8 jobs with SLURM array:
#   --shards 8 --shard-id $SLURM_ARRAY_TASK_ID
#
# Notes:
# - Tune GLOBAL_MIN_INTERVAL (0.20-0.30) only after checking _skipped.tsv for 429/maxlag.

import argparse
import asyncio
import json
import random
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

import aiohttp
import pycountry
import zlib

API_URL_TMPL       = "https://{lang}.wikipedia.org/w/api.php"
INCUBATOR_API_URL  = "https://incubator.wikimedia.org/w/api.php"
USER_AGENT         = "WikipediaScraper/5.0 (contact: you@example.org)"
TIMEOUT_S          = 30
MAX_DEPTH_DEFAULT  = 4

# Global minimum interval between ANY two outgoing requests (seconds).
GLOBAL_MIN_INTERVAL = 0.05
_last_request_ts = 0.0
_rate_lock = asyncio.Lock()

# Throttle logging (occasional notices to stderr)
THROTTLE_LOG_EVERY = 20
HTTP_429_COUNT = 0
MAXLAG_COUNT = 0

# Root categories (English seeds; WITHOUT namespace)
EN_ROOT_CATEGORIES = [
    "Historical objects",
    "History of sports",
    "History of ideologies",
]

# -------- filename helpers --------
try:
    from unidecode import unidecode
except Exception:
    unidecode = None

ASCII_FILENAMES = False

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

def incubator_title_for(lang_code: str, page_title: str) -> str:
    return f"Wp/{lang_code}/{page_title}"

# -------- sharding --------
def belongs_to_shard(lang: str, title: str, shard_id: int, shards: int) -> bool:
    if shards <= 1:
        return True
    h = zlib.crc32(f"{lang}\t{title}".encode("utf-8")) & 0xffffffff
    return (h % shards) == shard_id

# -------- dedup --------
visited_pages: Set[Tuple[str, str]] = set()
visited_categories: Set[Tuple[str, str]] = set()
visit_lock = asyncio.Lock()

# -------- skip logging --------
skip_log_lock = asyncio.Lock()
skip_log_path: Optional[Path] = None

async def log_skip(lang: str, title: str, reason: str, url: str = "", extra: str = ""):
    global skip_log_path
    if not skip_log_path:
        return
    line = f"{datetime.utcnow().isoformat()}Z\t{lang}\t{title}\t{reason}\t{url}\t{extra}\n"
    async with skip_log_lock:
        skip_log_path.parent.mkdir(parents=True, exist_ok=True)
        with skip_log_path.open("a", encoding="utf-8") as f:
            f.write(line)

def _raw_ext_from_ctype(ctype: str) -> str:
    c = (ctype or "").lower()
    if "html" in c: return ".html"
    if "xml" in c: return ".xml"
    if "plain" in c: return ".txt"
    return ".raw"

def _write_raw_variant(entry_dir: Path, language_name: str, raw_text: str, content_type: str):
    ext = _raw_ext_from_ctype(content_type)
    fname = sanitize(language_name) + ".raw" + ext
    (entry_dir / fname).write_text(raw_text or "", encoding="utf-8")

# -------- counters --------
raw_count_lock = asyncio.Lock()
RAW_ANY_COUNT = 0
RAW_HTML_COUNT = 0

async def incr_raw_counts(content_type: str):
    global RAW_ANY_COUNT, RAW_HTML_COUNT
    c = (content_type or "").lower()
    async with raw_count_lock:
        RAW_ANY_COUNT += 1
        if "html" in c:
            RAW_HTML_COUNT += 1

# -------- global rate limiter --------
async def global_rate_wait():
    global _last_request_ts
    async with _rate_lock:
        now = time.monotonic()
        delta = now - _last_request_ts
        if delta < GLOBAL_MIN_INTERVAL:
            await asyncio.sleep(GLOBAL_MIN_INTERVAL - delta)
        _last_request_ts = time.monotonic()

def _retry_after_seconds(resp) -> Optional[float]:
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return float(ra)
    except Exception:
        return None

def _param_title_hint(params: Dict[str, Any]) -> str:
    return str(params.get("titles") or params.get("cmtitle") or params.get("gcmtitle") or "")

# -------- HTTP client --------
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

    async def query(self, lang: str, params: Dict, retries: int = 5) -> Optional[Dict[str, Any]]:
        global HTTP_429_COUNT, MAXLAG_COUNT
        assert self.session is not None
        if lang == "incubator":
            url = INCUBATOR_API_URL
            host = "incubator.wikimedia.org"
        else:
            url = API_URL_TMPL.format(lang=lang)
            host = f"{lang}.wikipedia.org"
        params = dict(params)
        params.setdefault("maxlag", "5")

        backoff = 1.0
        for attempt in range(retries):
            await global_rate_wait()
            async with self.global_sema, self._host_sema(host):
                try:
                    async with self.session.get(url, params=params, allow_redirects=True) as resp:
                        ctype = (resp.headers.get("Content-Type") or "").lower()

                        # 429 handling with structured logging and occasional stderr notice
                        if resp.status == 429:
                            HTTP_429_COUNT += 1
                            ra = _retry_after_seconds(resp)
                            sleep_for = ra if ra is not None else min(backoff, 30.0)
                            sleep_for += random.uniform(0, 0.5)

                            await log_skip(
                                lang,
                                _param_title_hint(params),
                                "http_429_retry",
                                url=str(resp.url),
                                extra=f"Retry-After={ra}"
                            )
                            if HTTP_429_COUNT % THROTTLE_LOG_EVERY == 1:
                                print(f"[throttle] 429 from {host}; sleeping {sleep_for:.2f}s (count={HTTP_429_COUNT})", file=sys.stderr)

                            await asyncio.sleep(sleep_for)
                            backoff = min(backoff * 2, 30.0)
                            continue

                        text = await resp.text()

                        if "application/json" in ctype:
                            try:
                                data = json.loads(text)
                            except Exception:
                                return {"__non_json__": True, "raw": text, "content_type": ctype, "url": str(resp.url)}

                            # MediaWiki maxlag error handling
                            err = (data.get("error") or {})
                            if isinstance(err, dict) and err.get("code") == "maxlag":
                                MAXLAG_COUNT += 1
                                await log_skip(
                                    lang,
                                    _param_title_hint(params),
                                    "maxlag_retry",
                                    url=str(resp.url),
                                    extra=f"backoff={backoff:.2f}"
                                )
                                if MAXLAG_COUNT % THROTTLE_LOG_EVERY == 1:
                                    print(f"[throttle] maxlag from {host}; backoff {backoff:.2f}s (count={MAXLAG_COUNT})", file=sys.stderr)
                                await asyncio.sleep(min(backoff, 30.0) + random.uniform(0, 0.5))
                                backoff = min(backoff * 2, 30.0)
                                continue

                            return data

                        # Non-JSON response (e.g., redirects to HTML)
                        return {"__non_json__": True, "raw": text, "content_type": ctype, "url": str(resp.url)}

                except Exception as e:
                    # Network or parsing error; final attempt returns error marker
                    if attempt == retries - 1:
                        return {"__error__": f"{type(e).__name__}: {e}"}
                    await asyncio.sleep(min(backoff, 30.0) + random.uniform(0, 0.5))
                    backoff = min(backoff * 2, 30.0)

# -------- API wrappers --------
async def get_category_langlinks(client: WikiClient, lang: str, category_without_ns: str) -> List[Tuple[str, str]]:
    """
    From an English category name WITHOUT namespace (e.g., 'History of sports'),
    return list of (lang, full_localized_category_title) pairs.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "langlinks",
        "titles": f"Category:{category_without_ns}",
        "lllimit": "max",
        "redirects": 1,
    }
    data = await client.query(lang, params)
    if not data or data.get("__error__") or data.get("__non_json__"):
        reason = "langlinks_none" if not data else ("langlinks_error" if data.get("__error__") else "langlinks_non_json")
        await log_skip(lang, f"Category:{category_without_ns}", reason, url=(data or {}).get("url",""), extra=(data or {}).get("content_type",""))
        return []
    pages = data.get("query", {}).get("pages", {}) or {}
    page = next(iter(pages.values()), {}) if pages else {}
    out = []
    for ll in page.get("langlinks", []) or []:
        lg = ll.get("lang")
        ttl = ll.get("*")  # full localized title like "Thể loại:Lịch sử thể thao"
        if lg and ttl:
            out.append((lg, ttl))
    return out

async def get_category_members(
    client: WikiClient,
    lang: str,
    category_title_full: str,   # full, localized title, e.g., "Thể loại:Lịch sử thể thao"
    want_subcats: bool
) -> List[str]:
    members, cont = [], None
    cmtype = "subcat" if want_subcats else "page"
    ns = "14" if want_subcats else "0"
    while True:
        params = {
            "action": "query", "format": "json", "list": "categorymembers",
            "cmtitle": category_title_full,          # use as-is (no manual 'Category:' added)
            "cmtype": cmtype, "cmlimit": "500", "cmnamespace": ns,
        }
        if cont:
            params["cmcontinue"] = cont
        data = await client.query(lang, params)
        if not data or data.get("__error__") or data.get("__non_json__"):
            reason = "members_none" if not data else ("members_error" if data.get("__error__") else "members_non_json")
            extra = f"{cmtype} | {(data or {}).get('__error__','') or (data or {}).get('content_type','')}"
            await log_skip(lang, category_title_full, reason, url=(data or {}).get("url",""), extra=extra)
            break
        items = data.get("query", {}).get("categorymembers", []) or []
        # Keep full titles exactly as returned
        members += [it["title"] for it in items if "title" in it]
        cont = (data.get("continue", {}) or {}).get("cmcontinue")
        if not cont:
            break
    return members

async def get_page_langlinks(client: WikiClient, lang: str, title: str) -> List[Tuple[str, str]]:
    params = {
        "action": "query", "format": "json",
        "prop": "langlinks",
        "titles": title, "lllimit": "max",
        "redirects": 1,
    }
    data = await client.query(lang, params)
    if not data or data.get("__error__") or data.get("__non_json__"):
        reason = "page_langlinks_none" if not data else ("page_langlinks_error" if data.get("__error__") else "page_langlinks_non_json")
        await log_skip(lang, title, reason, url=(data or {}).get("url",""), extra=(data or {}).get("content_type",""))
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

async def fetch_extract_or_raw(client: WikiClient, lang: str, title: str) -> Tuple[str, Optional[Dict[str, str]]]:
    params = {"action": "query", "format": "json", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title}
    data = await client.query(lang, params)
    if not data:
        return "", None
    if data.get("__non_json__"):
        url = (data.get("url") or "").lower()
        if "incubator.wikimedia.org" in url:
            inc_params = {
                "action": "query", "format": "json", "prop": "extracts",
                "explaintext": 1, "redirects": 1, "titles": incubator_title_for(lang, title)
            }
            inc = await client.query("incubator", inc_params)
            if inc and not inc.get("__non_json__") and not inc.get("__error__"):
                pages = (inc.get("query", {}) or {}).get("pages", {}) or {}
                page = next(iter(pages.values()), {}) if pages else {}
                extract = (page.get("extract", "") or "")
                if extract.strip():
                    return extract, None
            return "", {"raw": data.get("raw", ""), "content_type": data.get("content_type", ""), "url": data.get("url", "")}
        return "", {"raw": data.get("raw", ""), "content_type": data.get("content_type", ""), "url": data.get("url", "")}
    if data.get("__error__"):
        return "", {"raw": "", "content_type": "", "url": ""}

    pages = (data.get("query", {}) or {}).get("pages", {}) or {}
    page = next(iter(pages.values()), {}) if pages else {}
    extract = (page.get("extract", "") or "")
    if extract.strip():
        return extract, None

    # Fallback: fetch HTML via parse API
    parse_params = {"action": "parse", "format": "json", "page": title, "prop": "text", "redirects": 1}
    parsed = await client.query(lang, parse_params)
    if parsed and not parsed.get("__non_json__") and not parsed.get("__error__"):
        html = ""
        parse_obj = parsed.get("parse", {})
        if "text" in parse_obj:
            t = parse_obj["text"]
            if isinstance(t, dict):
                html = t.get("*", "")
            elif isinstance(t, str):
                html = t
        if html.strip():
            return "", {"raw": html, "content_type": "text/html; charset=utf-8", "url": ""}
    return "", None

# -------- batching helpers --------
def chunked(seq: List[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

async def fetch_extracts_batch(client: WikiClient, lang: str, titles: List[str]) -> Dict[str, str]:
    if not titles:
        return {}
    params = {
        "action": "query", "format": "json",
        "prop": "extracts", "explaintext": 1, "redirects": 1,
        "titles": "|".join(titles)
    }
    data = await client.query(lang, params)
    if not data or data.get("__error__") or data.get("__non_json__"):
        return {}
    out: Dict[str, str] = {}
    pages = (data.get("query", {}) or {}).get("pages", {}) or {}
    for p in pages.values():
        t = p.get("title", "")
        ex = (p.get("extract", "") or "")
        if t and ex.strip():
            out[t] = ex
    return out

# -------- saving --------
async def save_article_and_variants(
    client: WikiClient,
    base_lang: str,
    title: str,
    out_dir: Path,
    base_text_override: str = ""
):
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

        language_name = get_language_name(lg)
        try:
            if lg == base_lang and base_text_override:
                text, raw_info = base_text_override, None
            else:
                text, raw_info = await fetch_extract_or_raw(client, lg, localized_title)

            if raw_info is not None:
                _write_raw_variant(entry_dir, language_name, raw_info.get("raw",""), raw_info.get("content_type",""))
                await incr_raw_counts(raw_info.get("content_type",""))
                await log_skip(lg, localized_title, reason="non_json_or_error_extract", url=raw_info.get("url",""),
                               extra=raw_info.get("content_type",""))
                print(f"? Saved RAW (weird response): {localized_title} [{lg}]")
                continue

            if not text.strip():
                await log_skip(lg, localized_title, reason="empty_extract")
                print(f"? Empty extract, logged skip: {localized_title} [{lg}]")
                continue

            fname = sanitize(language_name) + ".txt"
            (entry_dir / fname).write_text(text, encoding="utf-8")
            print(f"? Saved: {localized_title} [{lg}]")
        except Exception as e:
            await log_skip(lg, localized_title, reason="fetch_or_save_exception", extra=str(e))
            print(f"? fetch/save failed for {localized_title} [{lg}]: {e}")

# -------- crawl --------
async def scrape_category(
    client: WikiClient,
    lang: str,
    category_title_full: str,  # full, localized title (e.g., "Thể loại:Lịch sử thể thao")
    out_dir: Path,
    depth: int,
    max_depth: int,
    batch_size: int,
    shard_id: int,
    shards: int
):
    print(f"{'  '*(depth-1)}? depth={depth}, scraping {category_title_full} [{lang}]")

    # Dedup category visits per process
    async with visit_lock:
        ckey = (lang, category_title_full)
        if ckey in visited_categories:
            return
        visited_categories.add(ckey)

    # Pages (articles) in this category
    titles = await get_category_members(client, lang, category_title_full, want_subcats=False)
    print(f"[dbg] {category_title_full} [{lang}] pages={len(titles)} depth={depth}")

    # Shard titles: each job saves only its share
    if shards > 1:
        titles = [t for t in titles if belongs_to_shard(lang, t, shard_id, shards)]

    # Batch fetch base-language extracts
    base_texts: Dict[str, str] = {}
    if batch_size and batch_size > 1:
        for batch in chunked(titles, batch_size):
            batched = await fetch_extracts_batch(client, lang, batch[:50])  # safe limit = 50 per call
            base_texts.update(batched)

    # Save concurrently
    page_tasks = [
        save_article_and_variants(client, lang, t, out_dir, base_text_override=base_texts.get(t, ""))
        for t in titles
    ]
    await asyncio.gather(*page_tasks)

    # Recurse into subcategories (must traverse all to find this shard's pages deeper)
    if depth < max_depth:
        subcats = await get_category_members(client, lang, category_title_full, want_subcats=True)
        sub_tasks = []
        for subcat_full in subcats:  # already full, localized titles
            sub_dir = out_dir / sanitize(subcat_full)
            sub_tasks.append(scrape_category(
                client, lang, subcat_full, sub_dir, depth + 1, max_depth, batch_size, shard_id, shards
            ))
        await asyncio.gather(*sub_tasks)

# -------- main driver --------
async def amain(
    output_dir: Path,
    concurrency: int,
    per_host: int,
    max_depth: int,
    batch_size: int,
    shard_id: int,
    shards: int
):
    global skip_log_path, RAW_ANY_COUNT, RAW_HTML_COUNT, HTTP_429_COUNT, MAXLAG_COUNT, ASCII_FILENAMES
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_log_path = output_dir / "_skipped.tsv"
    with skip_log_path.open("w", encoding="utf-8") as f:
        f.write("timestamp_utc\tlang\ttitle\treason\turl\textra\n")

    async with WikiClient(concurrency, per_host) as client:
        # Build roots: English seeds (with namespace) + their langlinks (already full, localized titles)
        roots: List[Tuple[str, str]] = []
        for root in EN_ROOT_CATEGORIES:
            roots.append(("en", f"Category:{root}"))  # English canonical namespace once
            try:
                links = await get_category_langlinks(client, "en", root)
            except Exception as e:
                print(f"? langlinks failed for root '{root}': {e}")
                links = []
            roots.extend(links)  # [(lg, full_localized_category_title), ...]

        # DO NOT SHARD ROOTS — every shard traverses all roots; only pages are sharded.
        print(f"[info] total roots to traverse (unsharded): {len(roots)}")
        for lg, ttl in roots[:20]:
            print(f"  - [{lg}] {ttl}")
        if len(roots) > 20:
            print("  - ...")

        tasks = []
        for lg, ttl in roots:
            out_dir = output_dir / lg / sanitize(ttl)
            tasks.append(scrape_category(
                client, lg, ttl, out_dir, depth=1, max_depth=max_depth,
                batch_size=batch_size, shard_id=shard_id, shards=shards
            ))
        await asyncio.gather(*tasks)

    summary = (
        f"RAW responses saved (any content-type): {RAW_ANY_COUNT}\n"
        f"RAW HTML files saved: {RAW_HTML_COUNT}\n"
        f"HTTP 429 retries: {HTTP_429_COUNT}\n"
        f"maxlag retries: {MAXLAG_COUNT}\n"
    )
    print("\n=== Summary ===")
    print(summary)
    try:
        with (output_dir / "_summary.txt").open("w", encoding="utf-8") as f:
            f.write(summary)
    except Exception:
        pass

def main():
    global ASCII_FILENAMES

    p = argparse.ArgumentParser(description="Async Wikipedia category scraper (robust + batching + sharding + 429 logging)")
    p.add_argument("--output", type=str, default=str(Path.home() / "Desktop" / "wikipedia_articles"),
                   help="Output directory (default: ~/Desktop/wikipedia_articles)")
    p.add_argument("--concurrency", type=int, default=12,
                   help="Global concurrent HTTP requests (default: 12)")
    p.add_argument("--per-host", type=int, default=4,
                   help="Concurrent requests per host (default: 4)")
    p.add_argument("--max-depth", type=int, default=MAX_DEPTH_DEFAULT,
                   help=f"Max subcategory depth (default: {MAX_DEPTH_DEFAULT})")
    p.add_argument("--ascii-filenames", action="store_true",
                   help="Transliterate folder and file names to ASCII")
    p.add_argument("--batch", type=int, default=1,
                   help="Batch size for base-language extracts (1 = no batching; 50 is a safe max)")
    p.add_argument("--shards", type=int, default=1,
                   help="Total number of shards (jobs). Use >1 to split work across jobs.")
    p.add_argument("--shard-id", type=int, default=0,
                   help="Shard id for this run (0-based). Must be < --shards.")
    args = p.parse_args()

    if args.shards < 1:
        raise SystemExit("--shards must be >= 1")
    if not (0 <= args.shard_id < args.shards):
        raise SystemExit("--shard-id must be in [0, --shards-1]")

    ASCII_FILENAMES = bool(args.ascii_filenames)
    out = Path(args.output)

    asyncio.run(amain(
        output_dir=out,
        concurrency=args.concurrency,
        per_host=args.per_host,
        max_depth=args.max_depth,
        batch_size=args.batch,
        shard_id=args.shard_id,
        shards=args.shards
    ))

if __name__ == "__main__":
    main()




