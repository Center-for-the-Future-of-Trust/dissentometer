#!/usr/bin/env python3
"""
Sentence-wise date tagging over CSVs with optional Groq rate limiting.
Async version: up to N concurrent HTTP requests (default 100).
Requires: aiohttp
"""

from __future__ import annotations
import os, re, sys, csv, io, json, time, math, hashlib, argparse, asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# fcntl is POSIX-only (ok for HPC/Linux)
import fcntl

try:
    import aiohttp  # type: ignore
except Exception as e:
    print("[BOOT] Missing dependency 'aiohttp'. Install it in your venv: pip install aiohttp", file=sys.stderr)

# ---------- Helpers ----------

def _parse_bool(s: str) -> bool:
    return (s or "").strip().lower() in {"1", "true", "yes", "y"}

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    try:
        return int(float(raw)) if raw.strip() else default
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name, "")
    return raw if raw.strip() else default

DATEY_PATTERN = re.compile(
    r"""(?ix)
    \b\d{3,4}\b
    |
    \b(?:AD|CE|BC|BCE)\b
    |
    \b(?:century|centuries)\b
    |
    \b(?:dynasty|period|era)\b
    |
    \b(?:\d{3,4}s)\b
    """, re.UNICODE,
)

def approx_token_count(text: str, char_per_token: int = 4) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / max(1, char_per_token)))

def split_sentences_keep_delims(text: str) -> List[Tuple[str, str]]:
    if not text:
        return []
    out: List[Tuple[str, str]] = []
    start = 0
    for m in re.finditer(r'[.!?]+[\)\]\}\"\']*', text):
        end = m.end()
        sentence = text[start:end]
        md = re.search(r'([.!?]+[\)\]\}\"\']*)\s*$', sentence)
        if md:
            tail = md.group(1)
            core = sentence[:-len(tail)]
            out.append((core, tail))
        else:
            out.append((sentence, ""))
        start = end
    if start < len(text):
        out.append((text[start:], ""))
    return out

SYSTEM_PROMPT = (
    "You are a helpful assistant for tagging dates. "
    "Wrap exactly one date mention per sentence with <DATE>...</DATE>. "
    "Return the sentence with only that markup added; do not rephrase."
)

def build_user_prompt(sentence: str) -> str:
    return (
        "Tag the primary date in this sentence with <DATE> and </DATE>.\n\n"
        f"Sentence: {sentence}"
    )

@dataclass
class Limits:
    rpm: int
    tpm: int
    rpd: int
    tpd: int

class TokenBudget:
    def __init__(self, state_path: Path, limits: Limits):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        if not self.state_path.exists():
            init = {
                "minute": {"toks": 0, "reqs": 0, "ts": int(time.time())},
                "day": {"toks": 0, "reqs": 0, "ts": int(time.time())},
            }
            self._atomic_write(init)

    def _atomic_write(self, obj: dict) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(json.dumps(obj).encode("utf-8"))
        os.replace(tmp, self.state_path)

    def _read_locked(self) -> tuple[dict, any]:
        with open(self.state_path, "rb+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            raw = f.read().decode("utf-8") or "{}"
            state = json.loads(raw) if raw.strip() else {}
            if "minute" not in state or "day" not in state:
                state = {
                    "minute": {"toks": 0, "reqs": 0, "ts": int(time.time())},
                    "day": {"toks": 0, "reqs": 0, "ts": int(time.time())},
                }
            return state, f

    def _write_locked(self, f, state: dict) -> None:
        f.seek(0); f.truncate(0)
        f.write(json.dumps(state).encode("utf-8"))
        f.flush(); os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)

    def _maybe_reset_windows(self, state: dict, now: int) -> None:
        if now - state["minute"]["ts"] >= 60:
            state["minute"] = {"toks": 0, "reqs": 0, "ts": now}
        if now - state["day"]["ts"] >= 86400:
            state["day"] = {"toks": 0, "reqs": 0, "ts": now}

    def _would_exceed(self, state: dict, add_toks: int, add_reqs: int) -> tuple[bool, float]:
        now = int(time.time())
        self._maybe_reset_windows(state, now)
        if state["minute"]["reqs"] + add_reqs > self.limits.rpm:
            return True, max(0.0, 60 - (now - state["minute"]["ts"]))
        if state["minute"]["toks"] + add_toks > self.limits.tpm:
            return True, max(0.0, 60 - (now - state["minute"]["ts"]))
        if state["day"]["reqs"] + add_reqs > self.limits.rpd:
            return True, max(0.0, 86400 - (now - state["day"]["ts"]))
        if state["day"]["toks"] + add_toks > self.limits.tpd:
            return True, max(0.0, 86400 - (now - state["day"]["ts"]))
        return False, 0.0

    def reserve(self, add_toks: int, add_reqs: int) -> bool:
        # Blocking; we will call it via asyncio.to_thread inside async tasks.
        while True:
            state, f = self._read_locked()
            exceed, sleep_for = self._would_exceed(state, add_toks, add_reqs)
            if not exceed:
                state["minute"]["toks"] += add_toks
                state["minute"]["reqs"] += add_reqs
                state["day"]["toks"] += add_toks
                state["day"]["reqs"] += add_reqs
                self._write_locked(f, state)
                return True
            else:
                fcntl.flock(f, fcntl.LOCK_UN)
                time.sleep(min(sleep_for, 2.0))

    def commit(self, add_toks: int, add_reqs: int) -> None:
        # No-op (we already accounted at reserve time)
        pass

class NoopBudget:
    def __init__(self, *_, **__): pass
    def reserve(self, *_, **__): return True
    def commit(self, *_, **__):  pass

def should_tag(sentence: str) -> bool:
    if not sentence or not sentence.strip():
        return False
    if "<DATE>" in sentence and "</DATE>" in sentence:
        return False
    return bool(DATEY_PATTERN.search(sentence))

async def groq_chat_once_async(query: str, model: str, system_prompt: Optional[str], session: "aiohttp.ClientSession", timeout: float = 90.0):
    is_groq = _parse_bool(os.getenv("IS_GROQ", ""))
    api_key = os.getenv("GROQ_API_KEY")
    if is_groq:
        if not api_key:
            print("[API ] GROQ_API_KEY is not set", file=sys.stderr)
            return None, None
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        mode = "Groq cloud"
    else:
        url = _env_str("API_BASE_URL", "http://localhost:58112/v1/chat/completions")
        headers = {"Content-Type": "application/json"}
        mode = "Local/dev"
    print(f"[API ] Async -> {mode} endpoint: {url}", file=sys.stderr)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False,
    }
    try:
        start_time = time.time()
        async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
            status = resp.status
            text = await resp.text()
            elapsed = time.time() - start_time
            print(f"[API ] Request completed in {elapsed:.2f} seconds (status={status})", file=sys.stderr)
            if status != 200:
                print(f"[API ] ERROR status={status} body={text[:500]}", file=sys.stderr)
                return None, None
            try:
                raw = json.loads(text)
            except Exception as je:
                print(f"[API ] ERROR: JSON parse failure: {je}; body={text[:500]}", file=sys.stderr)
                return None, None
    except Exception as e:
        print(f"[API ] exception calling {mode} endpoint: {e}", file=sys.stderr)
        return None, None

    try:
        content = raw["choices"][0]["message"]["content"]
        usage = raw.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
    except Exception:
        content = None
        total_tokens = 0
    return content, total_tokens

async def tag_sentence_once_async(sentence: str, model: str, budget, session: "aiohttp.ClientSession", char_per_token: int = 4) -> str:
    # Reserve tokens in a thread to avoid blocking the event loop
    approx_in = approx_token_count(sentence, char_per_token=char_per_token) + 64
    await asyncio.to_thread(budget.reserve, approx_in, 1)
    query = build_user_prompt(sentence)
    out, used = await groq_chat_once_async(query, model=model, system_prompt=SYSTEM_PROMPT, session=session)

    # --- Robust failure logging ---
    if not out or not isinstance(out, str):
        preview = sentence[:200].replace("\n", " ")
        print(f"[FAIL] No content returned for sentence: {preview!r}", file=sys.stderr)
        return sentence  # fall back to original sentence (no drop)

    if "<DATE>" not in out or "</DATE>" not in out:
        preview = sentence[:200].replace("\n", " ")
        print(f"[WARN] Model returned no <DATE> tags for a date-y sentence: {preview!r}", file=sys.stderr)

    return out

async def tag_text_async(text: str, model: str, budget, char_per_token: int = 4, max_concurrency: int = 100) -> str:
    pieces = split_sentences_keep_delims(text)
    if not pieces:
        return ""

    # Build list preserving order
    cores = [core for core, _ in pieces]
    tails = [tail for _, tail in pieces]

    sem = asyncio.Semaphore(max_concurrency)

    async def task_for(i: int, s: str, session: "aiohttp.ClientSession"):
        if should_tag(s):
            async with sem:
                return await tag_sentence_once_async(s, model=model, budget=budget, session=session, char_per_token=char_per_token)
        else:
            return s

    # Single shared session for efficiency
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(task_for(i, s, session)) for i, s in enumerate(cores)]
        results = await asyncio.gather(*tasks)

    out_parts: List[str] = []
    for i, core_out in enumerate(results):
        out_parts.append(core_out + tails[i])
    return "".join(out_parts)

# ---------- Main ----------

def _validate_complete(out_csv: Path, content_col: str, tag_col: str, encoding: str = "utf-8") -> bool:
    """
    Return True if out_csv exists, has tag_col, and tag_col is non-empty wherever content_col is non-empty.
    """
    if not out_csv.exists():
        return False
    try:
        with open(out_csv, "r", encoding=encoding, newline="") as r:
            rdr = csv.DictReader(r)
            rows = list(rdr)
    except Exception:
        return False
    if not rows:
        return False
    if tag_col not in rows[0]:
        return False
    for r in rows:
        content = (r.get(content_col) or "").strip()
        tagv = (r.get(tag_col) or "").strip()
        if content and not tagv:
            return False
    return True

def _read_rows(fp: Path, encoding: str):
    with open(fp, "r", encoding=encoding, newline="") as r:
        rdr = csv.DictReader(r)
        rows = list(rdr)
        fields = rdr.fieldnames or []
    return rows, fields

def _atomic_write_csv(path: Path, fieldnames, rows, encoding: str = "utf-8"):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding=encoding) as w:
        wr = csv.DictWriter(w, fieldnames=fieldnames, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    os.replace(tmp, path)

def main() -> None:
    p = argparse.ArgumentParser(description="Sentence-wise date tagging (async, optional Groq rate limiting).")
    p.add_argument("--data-dir", required=True, help="Directory containing CSV files.")
    p.add_argument("--content-col", default="English translation", help="Column to read text from.")
    p.add_argument("--tag-col", default="date_tagged", help="New column to write tagged text into.")
    default_model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    p.add_argument("--model", default=default_model, help="Model name (Groq or local).")
    p.add_argument("--encoding", default="utf-8", help="CSV encoding.")
    p.add_argument("--char-per-token", type=int, default=4, help="Token estimate (~chars per token).")
    p.add_argument("--file-glob", default="*.csv", help="Glob for input files (non-recursive).")
    p.add_argument("--out-root", default=None, help="If set, write outputs under this root; else <data-dir>/date_tagged/.")
    p.add_argument("--max-concurrency", type=int, default=int(os.getenv("MAX_CONCURRENCY", "100")),
                   help="Max number of concurrent HTTP requests (default 100).")
    # New resume/skip/overwrite flags
    p.add_argument("--resume", action="store_true", help="Resume incomplete outputs (skip complete ones).")
    p.add_argument("--skip-existing", action="store_true", help="Skip files with complete outputs.")
    p.add_argument("--overwrite", action="store_true", help="Force overwrite even if output exists.")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERR ] data-dir does not exist: {data_dir}", file=sys.stderr)
        sys.exit(2)

    files = sorted(data_dir.glob(args.file_glob))
    if not files:
        print(f"[WARN] No files matched {args.file_glob} under {data_dir}", file=sys.stderr)

    out_root = (Path(args.out_root) / data_dir.name) if args.out_root else (data_dir / "date_tagged")
    out_root.mkdir(parents=True, exist_ok=True)

    is_groq = _parse_bool(os.getenv("IS_GROQ", ""))

    if is_groq:
        MODEL_LIMIT_DEFAULTS = {
            "llama-3.3-70b-versatile": dict(rpm=1000, tpm=300000, rpd=500000, tpd=10**12),
            "llama-3.1-8b-instant":    dict(rpm=1000, tpm=250000, rpd=500000, tpd=10**12),
        }
        md = MODEL_LIMIT_DEFAULTS.get(args.model, None)
        rpm_default = (md or {}).get("rpm", 30)
        tpm_default = (md or {}).get("tpm", 6000)
        rpd_default = (md or {}).get("rpd", 14400)
        tpd_default = (md or {}).get("tpd", 500000)
        limits = Limits(
            rpm=_env_int("GROQ_RPM", rpm_default),
            tpm=_env_int("GROQ_TPM", tpm_default),
            rpd=_env_int("GROQ_RPD", rpd_default),
            tpd=_env_int("GROQ_TPD", tpd_default),
        )
        token_state_dir = Path(os.getenv("TOKEN_STATE_DIR", ".rate_limit_state"))
        api_key = os.getenv("GROQ_API_KEY", "")
        key_id = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16] if api_key else "nokey"
        model_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.model)
        state_path = token_state_dir / key_id / f"budget_{model_id}.json"
        budget = TokenBudget(state_path, limits)
        rl = True
    else:
        budget = NoopBudget()
        rl = False

    for fp in files:
        rel = fp.relative_to(data_dir)
        out_csv = out_root / rel
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        start_csv = time.time()
        try:
            # Skip/Resume guards
            if not args.overwrite and (args.resume or args.skip_existing):
                if _validate_complete(out_csv, args.content_col, args.tag_col, args.encoding):
                    print(f"[SKIP] {rel} (already complete)")
                    continue

            # Read input
            rows, in_fields = _read_rows(fp, args.encoding)

            if not rows:
                # propagate empty
                _atomic_write_csv(out_csv, in_fields or [args.content_col, args.tag_col], [])
                elapsed_csv = time.time() - start_csv
                print(f"[PASS] {rel} (empty) in {elapsed_csv:.2f}s")
                continue

            if args.content_col not in rows[0]:
                # copy-through if content column missing
                txt = fp.read_text(encoding=args.encoding, errors="ignore")
                with open(out_csv, "w", newline="", encoding="utf-8") as w:
                    w.write(txt)
                elapsed_csv = time.time() - start_csv
                print(f"[PASS] {rel} (copied: missing '{args.content_col}') in {elapsed_csv:.2f}s")
                continue

            # Prepare fieldnames
            fieldnames = list(rows[0].keys())
            if args.tag_col not in fieldnames:
                fieldnames.append(args.tag_col)

            # If resume and prior out exists, load it
            existing_rows = []
            if args.resume and out_csv.exists() and not args.overwrite:
                try:
                    existing_rows, _ = _read_rows(out_csv, args.encoding)
                except Exception:
                    existing_rows = []

            # Build subset of rows to tag (or all if default/overwrite mode)
            to_tag_indices: List[int] = []
            contents_to_tag: List[str] = []
            if args.resume:
                for i, r in enumerate(rows):
                    # Prefer any existing tag (either from current rows or prior output)
                    cur_tag = (r.get(args.tag_col) or "").strip()
                    prior_tag = ""
                    if existing_rows and i < len(existing_rows):
                        prior_tag = (existing_rows[i].get(args.tag_col) or "").strip()

                    if cur_tag:
                        # keep it
                        continue
                    if prior_tag:
                        r[args.tag_col] = prior_tag
                        continue

                    # Needs tagging
                    to_tag_indices.append(i)
                    contents_to_tag.append(r.get(args.content_col, "") or "")
            else:
                # Default/overwrite path: tag all rows
                to_tag_indices = list(range(len(rows)))
                contents_to_tag = [r.get(args.content_col, "") or "" for r in rows]

            # Tag needed subset
            if contents_to_tag:
                tagged_subset: List[str] = asyncio.run(
                    tag_texts_over_rows(
                        texts=contents_to_tag,
                        model=args.model,
                        budget=budget,
                        char_per_token=args.char_per_token,
                        max_concurrency=args.max_concurrency,
                    )
                )
                for idx, tagged in zip(to_tag_indices, tagged_subset):
                    rows[idx][args.tag_col] = tagged

            # Atomic write
            _atomic_write_csv(out_csv, fieldnames, rows, encoding=args.encoding)
            elapsed_csv = time.time() - start_csv

            # Post-write validation/logging: count rows with content but missing tags
            missing = 0
            for r in rows:
                content = (r.get(args.content_col) or "").strip()
                tagv = (r.get(args.tag_col) or "").strip()
                if content and not tagv:
                    missing += 1
            if missing > 0:
                print(f"[WARN] {rel}: {missing} row(s) had content but no tags", file=sys.stderr)

            mode = "resumed" if args.resume else ("overwrote" if args.overwrite else "processed")
            print(f"[OK  ] {rel} ({mode}) -> {out_csv.relative_to(out_root)} in {elapsed_csv:.2f}s")
        except Exception as e:
            elapsed_csv = time.time() - start_csv
            print(f"[ERR ] {rel}: {e} after {elapsed_csv:.2f}s", file=sys.stderr)

    print("Done; files processed.")

async def tag_texts_over_rows(texts: List[str], model: str, budget, char_per_token: int, max_concurrency: int) -> List[str]:
    # We process each row's text independently; within each, sentence-level tasks are limited by a shared semaphore.
    # To balance throughput and memory, we reuse a single aiohttp session across all rows.
    results: List[str] = []
    async with aiohttp.ClientSession() as session:
        for idx, text in enumerate(texts):
            result = await tag_text_async(text, model=model, budget=budget, char_per_token=char_per_token, max_concurrency=max_concurrency)
            results.append(result)
    return results

if __name__ == "__main__":
    main()
