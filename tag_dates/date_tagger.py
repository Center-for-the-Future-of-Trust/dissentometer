#!/usr/bin/env python3
"""
Sentence-wise date tagging over CSVs, using Groq with a shared token budget.
- No retry logic. One HTTP attempt per sentence.
- Coordinated per-minute and per-day limits across processes via a JSON state file.

Env knobs (model-aware defaults; override via env):
  GROQ_RPM, GROQ_TPM, GROQ_RPD, GROQ_TPD
  TOKEN_STATE_DIR=.rate_limit_state

If GROQ returns usage tokens, we record actuals; otherwise we fall back
to the estimate used for reservation.
"""

import argparse, csv, hashlib, json, os, re, sys, time
import sys
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import pandas as pd
import requests
import fcntl  # POSIX file locking

SYSTEM_PROMPT = """Your job is to tag dates and time periods in text.
Insert <DATE>…</DATE> around every date or time-period expression. Return the original text
with only the tags inserted. Do not add any commentary.

Examples:
- Specific days (e.g., 2 March 2012, March 2012)
- Years and eras (e.g., 1992, 300 BC/BCE, AD/CE 79, AH 622)
- Decades with qualifiers (e.g., 1920s, late 1990s, early 2000s)
- Centuries numeric or spelled (e.g., 20th century, eighteenth century, mid-18th century)
- Ranges (e.g., 1880–1889, 1000 BCE to 200 AD, between 1914 and 1918, 1945/46)
- Seasons/quarters with a year (e.g., winter 1942, Q4 2020)
- Reign/activity markers (e.g., r. 1558–1603, fl. 1510s, c./circa 1200)
- Named periods/dynasties/eras (e.g., Qing Dynasty, Tokugawa period, Bronze Age, Middle Ages, Heisei era, Jurassic period).
"""

DATEY_PATTERN = re.compile(
    r"""(?ix)
    \b\d{3,4}s?\b                     # 3- or 4-digit numbers, with optional trailing 's' (e.g., 1920, 1920s)
    |
    \b(?:AD|CE|BC|BCE)\b              # era markers
    |
    \b(?:century|centuries)\b         # century/centuries
    |
    \b(?:dynasty|period|era)\b        # named historical periods
    """, re.UNICODE,
)

SENT_SPLIT = re.compile(r"([.!?…؛।。！？]+)(\s+|$)")

def split_sentences_keep_delims(text: str) -> List[Tuple[str, str]]:
    parts: List[Tuple[str, str]] = []
    start = 0
    for m in SENT_SPLIT.finditer(text):
        end = m.start()
        core = text[start:end]
        delim = m.group(1) + (m.group(2) or "")
        parts.append((core, delim))
        start = m.end()
    if start < len(text):
        parts.append((text[start:], ""))
    return parts

def estimate_tokens(s: str, char_per_token: int = 4) -> int:
    s = s or ""
    return max(1, len(s) // max(1, char_per_token))

def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```") and s.endswith("```"):
        inner = s.strip("`").strip()
        if "\n" in inner:
            first, rest = inner.split("\n", 1)
            if first.strip().lower() in {"text","plain","txt","markdown","md","html","xml"}:
                return rest.strip()
        return inner
    return s

def should_tag(sentence: str) -> bool:
    if not sentence or not sentence.strip():
        return False
    if "<DATE>" in sentence and "</DATE>" in sentence:
        return False
    return bool(DATEY_PATTERN.search(sentence))

@dataclass
class Limits:
    rpm: int
    tpm: int
    rpd: int
    tpd: int

class TokenBudget:
    def __init__(self, state_path: Path, limits: Limits):
        self.state_path = state_path
        self.limits = limits
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked_state(self):
        with open(self.state_path, "a+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            raw = f.read()
            if not raw:
                state = {"minute": {"events": []}, "day": {"date": datetime.now(timezone.utc).date().isoformat(), "tokens": 0, "requests": 0}}
            else:
                try:
                    state = json.loads(raw.decode("utf-8"))
                except Exception:
                    state = {"minute": {"events": []}, "day": {"date": datetime.now(timezone.utc).date().isoformat(), "tokens": 0, "requests": 0}}
            yield state, f
            f.seek(0); f.truncate(0)
            f.write(json.dumps(state).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

    def _purge_and_roll_day(self, state):
        now = time.time()
        one_min_ago = now - 60.0
        events = [(ts, tok, req) for (ts, tok, req) in state["minute"]["events"] if ts >= one_min_ago]
        state["minute"]["events"] = events
        today = datetime.now(timezone.utc).date().isoformat()
        if state["day"].get("date") != today:
            state["day"] = {"date": today, "tokens": 0, "requests": 0}

    def _minute_totals(self, state):
        toks = sum(e[1] for e in state["minute"]["events"])
        reqs = sum(e[2] for e in state["minute"]["events"])
        return toks, reqs

    def reserve(self, est_tokens: int, est_requests: int = 1, mode: str = "sleep", poll_s: float = 0.25) -> bool:
        est_tokens = max(1, int(est_tokens))
        est_requests = max(1, int(est_requests))
        while True:
            with self._locked_state() as (state, _f):
                self._purge_and_roll_day(state)
                min_tokens, min_requests = self._minute_totals(state)
                day_tokens = state["day"]["tokens"]
                day_requests = state["day"]["requests"]

                can_minute = (min_tokens + est_tokens) <= self.limits.tpm and (min_requests + est_requests) <= self.limits.rpm
                can_day = (day_tokens + est_tokens) <= self.limits.tpd and (day_requests + est_requests) <= self.limits.rpd

                if can_minute and can_day:
                    state["minute"]["events"].append([time.time(), est_tokens, est_requests])
                    state["day"]["tokens"] += est_tokens
                    state["day"]["requests"] += est_requests
                    return True

                if mode == "skip":
                    return False

                now = time.time()
                soonest = None
                for ts, tok, req in state["minute"]["events"]:
                    free_in = max(0.0, (ts + 60.0) - now)
                    if soonest is None or free_in < soonest:
                        soonest = free_in
                time.sleep(max(soonest or 0.25, poll_s))

    def commit(self, actual_tokens: Optional[int] = None, actual_requests: Optional[int] = None):
        with self._locked_state() as (state, _f):
            self._purge_and_roll_day(state)
            if not state["minute"]["events"]:
                return
            ts, est_tok, est_req = state["minute"]["events"][-1]
            atok = max(1, int(actual_tokens)) if actual_tokens is not None else est_tok
            areq = max(1, int(actual_requests)) if actual_requests is not None else est_req
            delta_tok = atok - est_tok
            delta_req = areq - est_req
            state["minute"]["events"][-1] = [ts, atok, areq]
            state["day"]["tokens"] += delta_tok
            state["day"]["requests"] += delta_req

_SESSION = requests.Session()

def groq_chat_once(query: str, model: str, system_prompt: Optional[str], timeout: float = 60.0):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[API ] GROQ_API_KEY is not set", file=sys.stderr)
        return None, None

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    payload = {"model": model, "messages": messages, "temperature": 0.0}

    try:
        resp = _SESSION.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            print(f"[API ] status={resp.status_code} body={resp.text[:500]}", file=sys.stderr)
            return None, None
        data = resp.json()
        content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
        content = _strip_code_fences(content or "")
        usage = data.get("usage", {})
        total_tokens = usage.get("total_tokens")
        return (content or None), total_tokens
    except Exception as e:
        print(f"[API ] exception={e}", file=sys.stderr)
        return None, None

def tag_sentence_once(s: str, *, model: str, budget: TokenBudget, char_per_token: int) -> str:
    est_user = estimate_tokens(s, char_per_token)
    est_sys  = estimate_tokens(SYSTEM_PROMPT, char_per_token)
    est_assistant = max(64, est_user // 2)
    est_total = est_user + est_sys + est_assistant

    ok = budget.reserve(est_total, est_requests=1, mode="sleep")
    if not ok:
        return s

    content, actual_total = groq_chat_once(s, model, SYSTEM_PROMPT)
    budget.commit(actual_tokens=actual_total, actual_requests=1)

    return content if content else s

def process_cell(cell: Optional[str], *, model: str, budget: TokenBudget, char_per_token: int) -> str:
    if cell is None:
        return ""
    text = str(cell)
    if not text.strip():
        return text

    pieces = split_sentences_keep_delims(text)
    out: List[str] = []
    for core, tail in pieces:
        s = core
        if should_tag(s):
            s = tag_sentence_once(s, model=model, budget=budget, char_per_token=char_per_token)
        out.append(s + tail)
    return "".join(out)

def main() -> None:
    p = argparse.ArgumentParser(description="Sentence-wise date tagging (no retries) with shared token budget.")
    p.add_argument("--data-dir", required=True, help="Directory containing CSV files.")
    p.add_argument("--content-col", default="English translation", help="Column to read text from.")
    p.add_argument("--tag-col", default="date_tagged", help="New column to write tagged text into.")
    #p.add_argument("--model", default="llama-3.3-70b-versatile", help="Groq model name.")
    p.add_argument("--model", default="llama-3.1-8b-instant", help="Groq model name.")
    p.add_argument("--encoding", default="utf-8", help="CSV encoding.")
    p.add_argument("--char-per-token", type=int, default=4, help="Token estimate (~chars per token).")
    p.add_argument("--file-glob", default="*.csv", help="Glob for input files (non-recursive).")
    p.add_argument("--out-root", default=None, help="If set, write outputs under this root; else <data-dir>/date_tagged/.")
    args = p.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set.", file=sys.stderr); sys.exit(2)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: Not a directory: {data_dir}", file=sys.stderr); sys.exit(1)

    out_root = (Path(args.out_root) / data_dir.name) if args.out_root else (data_dir / "date_tagged")
    out_root.mkdir(parents=True, exist_ok=True)

    MODEL_LIMIT_DEFAULTS = {
        "llama-3.3-70b-versatile": dict(rpm=1000, tpm=300000, rpd=500000, tpd=10**12),
        "llama-3.1-8b-instant":    dict(rpm=1000, tpm=250000, rpd=500000, tpd=10**12),
    }
    md = MODEL_LIMIT_DEFAULTS.get(args.model, None)

    rpm_default = str((md or {}).get("rpm", 30))
    tpm_default = str((md or {}).get("tpm", 6000))
    rpd_default = str((md or {}).get("rpd", 14400))
    tpd_default = str((md or {}).get("tpd", 500000))

    limits = Limits(
        rpm=int(float(os.getenv("GROQ_RPM", rpm_default))),
        tpm=int(float(os.getenv("GROQ_TPM", tpm_default))),
        rpd=int(float(os.getenv("GROQ_RPD", rpd_default))),
        tpd=int(float(os.getenv("GROQ_TPD", tpd_default))),
    )

    token_state_dir = Path(os.getenv("TOKEN_STATE_DIR", ".rate_limit_state"))
    key_id = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    model_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.model)
    state_path = token_state_dir / key_id / f"budget_{model_id}.json"
    budget = TokenBudget(state_path, limits)

    csvs = sorted(data_dir.glob(args.file_glob))
    if not csvs:
        print(f"[WARN] No files matching {args.file_glob} in {data_dir}")
        return

    print(f"[INFO] DATA_DIR={data_dir}")
    print(f"[INFO] Model={args.model}")
    print(f"[INFO] Limits: RPM={limits.rpm} TPM={limits.tpm} RPD={limits.rpd} TPD={limits.tpd}")
    print(f"[INFO] Token state: {state_path}")
    print(f"[INFO] Content col='{args.content_col}' → Tag col='{args.tag_col}'")
    print(f"[INFO] Writing outputs under: {out_root}")
    print(f"[INFO] Found {len(csvs)} CSV file(s).")

    for in_path in csvs:
        base = in_path.stem
        out_path = out_root / f"{base}_date_tagged.csv"
        try:
            df = pd.read_csv(in_path, dtype=str, encoding=args.encoding)
        except Exception as e:
            print(f"[SKIP] Could not read {in_path}: {e}", file=sys.stderr); continue

        if args.content_col not in df.columns:
            print(f"[SKIP] Missing column '{args.content_col}' in {in_path}", file=sys.stderr); continue

        print(f"[RUN ] {in_path.name} → {out_path.name}")
        df[args.tag_col] = df[args.content_col].apply(
            lambda cell: process_cell(cell, model=args.model, budget=budget, char_per_token=args.char_per_token)
        )

        try:
            df.to_csv(out_path, index=False, encoding=args.encoding, quoting=csv.QUOTE_MINIMAL)
            print(f"[OK  ] Wrote: {out_path}")
        except Exception as e:
            print(f"[FAIL] Could not write {out_path}: {e}")

    print("[DONE] All files processed.")

if __name__ == "__main__":
    main()
