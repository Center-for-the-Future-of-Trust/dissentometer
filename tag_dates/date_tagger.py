#!/usr/bin/env python3
"""
Sentence-wise date tagging over CSVs with optional Groq rate limiting.
Now logs per-CSV timing.
"""

from __future__ import annotations
import os, re, sys, csv, io, json, time, math, hashlib, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import fcntl

# ---------- Helpers ----------

def _parse_bool(s: str) -> bool:
    return (s or "").strip().lower() in {"1", "true", "yes", "y"}

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    try:
        return int(float(raw)) if raw.strip() else default
    except Exception:
        return default

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

def groq_chat_once(query: str, model: str, system_prompt: Optional[str], timeout: float = 90.0):
    import urllib.request
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
        url = os.getenv("LOCAL_CHAT_URL", "http://localhost:58112/v1/chat/completions")
        headers = {"Content-Type": "application/json"}
        mode = "Local/dev"
    print(f"[API ] Sending request to {mode} endpoint: {url}", file=sys.stderr)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
    }
    try:
        start_time = time.time()
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - start_time
        print(f"[API ] Request completed in {elapsed:.2f} seconds", file=sys.stderr)
    except Exception as e:
        print(f"[API ] exception while calling {mode} endpoint: {e}", file=sys.stderr)
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

def tag_sentence_once(sentence: str, model: str, budget: TokenBudget, char_per_token: int = 4) -> str:
    approx_in = approx_token_count(sentence, char_per_token=char_per_token) + 64
    budget.reserve(approx_in, 1)
    query = build_user_prompt(sentence)
    out, used = groq_chat_once(query, model=model, system_prompt=SYSTEM_PROMPT)
    return sentence if not out else out

def tag_text(text: str, model: str, budget: TokenBudget, char_per_token: int = 4) -> str:
    pieces = split_sentences_keep_delims(text)
    out: List[str] = []
    for core, tail in pieces:
        s = core
        if should_tag(s):
            s = tag_sentence_once(s, model=model, budget=budget, char_per_token=char_per_token)
        out.append(s + tail)
    return "".join(out)

# ---------- Main ----------

def main() -> None:
    p = argparse.ArgumentParser(description="Sentence-wise date tagging (no retries) with optional Groq rate limiting.")
    p.add_argument("--data-dir", required=True, help="Directory containing CSV files.")
    p.add_argument("--content-col", default="English translation", help="Column to read text from.")
    p.add_argument("--tag-col", default="date_tagged", help="New column to write tagged text into.")
    default_model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    p.add_argument("--model", default=default_model, help="Groq model name.")
    p.add_argument("--encoding", default="utf-8", help="CSV encoding.")
    p.add_argument("--char-per-token", type=int, default=4, help="Token estimate (~chars per token).")
    p.add_argument("--file-glob", default="*.csv", help="Glob for input files (non-recursive).")
    p.add_argument("--out-root", default=None, help="If set, write outputs under this root; else <data-dir>/date_tagged/.")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: Not a directory: {data_dir}", file=sys.stderr); sys.exit(1)

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

    print(f"[INFO] Model={args.model}")
    print(f"[INFO] IS_GROQ={is_groq}")
    if rl:
        print(f"[INFO] Limits: RPM={limits.rpm} TPM={limits.tpm} RPD={limits.rpd} TPD={limits.tpd}")
        print(f"[INFO] Token state: {state_path}")
    else:
        print(f"[INFO] Rate limiting: disabled (NoopBudget)")

    files = sorted(data_dir.glob(args.file_glob))
    if not files:
        print(f"[WARN] No files matching {args.file_glob} in {data_dir}", file=sys.stderr)

    for fp in files:
        rel = fp.relative_to(data_dir)
        out_csv = out_root / rel
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        start_csv = time.time()
        try:
            with open(fp, "rb") as f:
                raw = f.read()
            try:
                txt = raw.decode(args.encoding)
            except UnicodeDecodeError:
                txt = raw.decode("utf-8", errors="replace")

            buf = io.StringIO(txt)
            rdr = csv.DictReader(buf)
            rows = list(rdr)
            if not rows:
                with open(out_csv, "w", newline="", encoding="utf-8") as w:
                    w.write(txt)
                elapsed_csv = time.time() - start_csv
                print(f"[PASS] {rel} (empty) in {elapsed_csv:.2f}s")
                continue

            if args.content_col not in rows[0]:
                print(f"[WARN] Missing column '{args.content_col}' in {rel}; copying through.", file=sys.stderr)
                with open(out_csv, "w", newline="", encoding="utf-8") as w:
                    w.write(txt)
                elapsed_csv = time.time() - start_csv
                print(f"[PASS] {rel} (copied) in {elapsed_csv:.2f}s")
                continue

            fieldnames = list(rows[0].keys())
            if args.tag_col not in fieldnames:
                fieldnames.append(args.tag_col)

            for r in rows:
                content = r.get(args.content_col, "")
                tagged = tag_text(content, model=args.model, budget=budget, char_per_token=args.char_per_token)
                r[args.tag_col] = tagged

            with open(out_csv, "w", newline="", encoding="utf-8") as w:
                wr = csv.DictWriter(w, fieldnames=fieldnames)
                wr.writeheader()
                for r in rows:
                    wr.writerow(r)

            elapsed_csv = time.time() - start_csv
            print(f"[OK  ] {rel} -> {out_csv.relative_to(out_root)} in {elapsed_csv:.2f}s")
        except Exception as e:
            elapsed_csv = time.time() - start_csv
            print(f"[ERR ] {rel}: {e} after {elapsed_csv:.2f}s", file=sys.stderr)

    print("Done; files processed.")

if __name__ == "__main__":
    main()
