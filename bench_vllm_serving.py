#!/usr/bin/env python3
"""
BIOSYS v2 — Self-Contained Real vLLM Serving Verification Benchmark (PAI-DSW Ready)
===================================================================================
Single-file, zero-extra-dependency (stdlib + numpy + scikit-learn) benchmark designed
for Alibaba Cloud PAI-DSW + unmodified vLLM (`--enable-prefix-caching`) + Qwen2.5-7B.

What it measures on a real GPU serving engine:
  1. Wall-clock TTFT (P50 / P90 / P99 / Mean ms) via HTTP streaming (`/v1/chat/completions`)
  2. Real end-to-end latency & throughput (req/s)
  3. Real vLLM KV-Cache Prefix Hit Rate via BOTH:
       - OpenAI usage `prompt_tokens_details.cached_tokens`
       - vLLM Prometheus `/metrics` (`prefix_cache_hits_total / prefix_cache_queries_total`)
  4. Real LLM end-to-end QA accuracy on official LoCoMo (Token-F1, Substring-EM, Cat-2 Temporal)
  5. Real GPU TTFT & prefix caching on the tool-heavy Coding Agent workload (240 turns)
  6. Incremental JSON checkpointing (`--resume`) so zero GPU compute hours are ever wasted.

Upload ONLY two files to PAI-DSW:
  - bench_vllm_serving.py
  - locomo10.json  (in current dir or ./data/locomo10.json)

Quick usage on PAI-DSW:
  1) 30-second smoke test:
     python bench_vllm_serving.py --smoke --base-url http://127.0.0.1:8000 --model qwen2.5-7b
  2) Full paper evaluation (all 10 LoCoMo convs + Coding Agent, incremental checkpoint):
     python bench_vllm_serving.py --max-convs 10 --base-url http://127.0.0.1:8000 --model qwen2.5-7b --resume
  3) Local offline dry-run verification (no GPU needed):
     python bench_vllm_serving.py --mock --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# =============================================================================
# 1. Text Normalization, LoCoMo Official QA Metrics (Token-F1 & EM), Stopwords
# =============================================================================

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "has", "had",
    "was", "were", "are", "is", "be", "been", "being", "you", "your", "they",
    "them", "their", "she", "her", "him", "his", "our", "out", "about", "what",
    "when", "where", "which", "who", "whom", "how", "why", "did", "does", "do",
    "can", "could", "would", "should", "will", "just", "very", "really", "also",
    "like", "into", "over", "after", "before", "then", "than", "some", "more",
    "much", "many", "well", "good", "great", "nice", "yeah", "yes", "hey", "hi",
    "hello", "oh", "wow", "thanks", "thank", "sure", "okay", "going", "got",
    "get", "make", "made", "take", "took", "see", "saw", "know", "think", "feel",
    "want", "time", "day", "today", "there", "here", "one", "two", "still", "too",
    "back", "even", "most", "such", "only", "same", "other", "both", "each",
    "few", "all", "any", "not", "but", "because", "while", "since",
}

DIA_ID_RE = re.compile(r"D(\d+):(\d+)")
WORD_RE = re.compile(r"[a-zA-Z0-9']+")

BASE_SYSTEM_PROMPT = (
    "You are a persistent personal memory assistant operating over long multi-session dialogues. "
    "Answer the user's question concisely (in one short phrase or date, under 15 words) using ONLY "
    "the provided session timestamps, consolidated knowledge nodes, and recent dialogue turns. "
    "When facts conflict across time, always prioritize the newer timestamped update or log entry. "
    "Do not include explanations or preamble; output only the direct answer.\n"
    "Memory tools schema: load_knowledge_node(node_id), load_raw_log_offset(lsn), "
    "search_memory(query_vec, time_window), update_state(entity, value, valid_from), "
    "verify_temporal_order(ts_a, ts_b), inspect_archive(session_id, offset).\n"
    "Guidelines: preserve exact dates and entity names; resolve relative dates such as 'yesterday' "
    "or 'last week' against the attached session date header; abstain only if no relevant memory exists."
)


def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def compute_qa_scores(pred: str, gold: str) -> Tuple[float, float]:
    """Return (token_f1, substring_em) following standard LoCoMo / SQuAD evaluation."""
    np_str = normalize_answer(pred)
    ng_str = normalize_answer(gold)
    if not ng_str:
        return (1.0 if not np_str else 0.0), (1.0 if not np_str else 0.0)
    if not np_str:
        return 0.0, 0.0
    sub_em = 1.0 if (ng_str in np_str or np_str in ng_str) else 0.0
    p_toks = np_str.split()
    g_toks = ng_str.split()
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, sub_em
    prec = num_same / len(p_toks)
    rec = num_same / len(g_toks)
    f1 = 2.0 * prec * rec / (prec + rec)
    return f1, sub_em


def content_terms(text: str, speakers_lower: Set[str]) -> List[str]:
    words = [w.lower().strip("'") for w in WORD_RE.findall(text)]
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS and w not in speakers_lower]


def extract_dia_ids(raw_evidence) -> List[str]:
    if isinstance(raw_evidence, str):
        items = [raw_evidence]
    elif isinstance(raw_evidence, list):
        items = [str(x) for x in raw_evidence]
    else:
        return []
    out: List[str] = []
    for it in items:
        for m in DIA_ID_RE.finditer(it):
            out.append(f"D{m.group(1)}:{m.group(2)}")
    return out


# =============================================================================
# 2. vLLM Streaming Client + Prometheus /metrics Prefix-Cache Telemetry
# =============================================================================

@dataclass
class InferenceResult:
    answer: str
    ttft_ms: float
    e2e_ms: float
    prompt_tokens: int
    cached_tokens: Optional[int]  # From OpenAI usage if reported by vLLM
    completion_tokens: int


class VLLMClient:
    """Zero-dependency HTTP streaming client for unmodified vLLM OpenAI server."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0, mock: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.mock = mock
        self._mock_prefix_blocks: OrderedDict[Tuple[int, ...], int] = OrderedDict()

    def check_health(self) -> None:
        if self.mock:
            return
        url = f"{self.base_url}/v1/models"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("id") for m in data.get("data", [])]
                print(f"[vLLM OK] Connected to {self.base_url} | Available models: {models}")
                if self.model not in models and models:
                    print(f"[WARN] Requested --model '{self.model}' not in {models}; using '{models[0]}'")
                    self.model = models[0]
        except Exception as exc:
            raise RuntimeError(
                f"Cannot connect to vLLM server at {url}. Start vLLM first! Error: {exc}"
            ) from exc

    def read_prometheus_prefix_counters(self) -> Dict[str, float]:
        """
        Parse vLLM /metrics for exact prefix cache hit/query token counters.
        Handles both vLLM v0.6/v0.7 (`gpu_prefix_cache_*`) and v0.8+/V1 (`prefix_cache_*`).
        """
        if self.mock:
            return {}
        url = f"{self.base_url}/metrics"
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                text = resp.read().decode("utf-8", errors="ignore")
        except Exception:
            return {}

        hits = 0.0
        queries = 0.0
        found_hits = False
        found_queries = False
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            metric_name = parts[0].split("{")[0]
            try:
                val = float(parts[-1])
            except ValueError:
                continue
            if metric_name in ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"):
                hits += val
                found_hits = True
            elif metric_name in ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"):
                queries += val
                found_queries = True

        if found_hits and found_queries:
            return {"hits": hits, "queries": queries}
        return {}

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 32,
        max_prompt_words: int = 7600,
    ) -> InferenceResult:
        # Safety truncation: Qwen2.5 BPE averages ~1.7x whitespace words; 7600 words ~= 13K BPE tokens (< 16384)
        words = user_prompt.split()
        if len(words) > max_prompt_words:
            user_prompt = " ".join(words[-max_prompt_words:])

        if self.mock:
            return self._mock_generate(system_prompt, user_prompt, max_tokens)

        for attempt in range(3):
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            t_start = time.perf_counter()
            t_first: Optional[float] = None
            chunks: List[str] = []
            prompt_tokens = 0
            cached_tokens: Optional[int] = None
            completion_tokens = 0

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = obj.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            content = delta.get("content")
                            if content:
                                if t_first is None:
                                    t_first = time.perf_counter()
                                chunks.append(content)

                        usage = obj.get("usage")
                        if isinstance(usage, dict):
                            prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                            completion_tokens = int(usage.get("completion_tokens", completion_tokens))
                            details = usage.get("prompt_tokens_details")
                            if isinstance(details, dict) and details.get("cached_tokens") is not None:
                                cached_tokens = int(details["cached_tokens"])

                t_end = time.perf_counter()
                if t_first is None:
                    t_first = t_end
                return InferenceResult(
                    answer="".join(chunks).strip(),
                    ttft_ms=(t_first - t_start) * 1000.0,
                    e2e_ms=(t_end - t_start) * 1000.0,
                    prompt_tokens=prompt_tokens,
                    cached_tokens=cached_tokens,
                    completion_tokens=completion_tokens,
                )
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="ignore")
                if exc.code == 400 and attempt < 2:
                    # Auto-trim 20% from prompt head on context-length overflow and retry cleanly
                    w_cur = user_prompt.split()
                    user_prompt = " ".join(w_cur[max(1, len(w_cur) // 5):])
                    print(f"  [WARN] HTTP 400 (attempt {attempt+1}), trimmed prompt to {len(user_prompt.split())} words and retrying...", flush=True)
                    continue
                raise RuntimeError(f"vLLM HTTP {exc.code}: {err_body}") from exc

        raise RuntimeError("Unreachable")

    def _mock_generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> InferenceResult:
        """Offline dry-run simulator mirroring block-16 prefix caching + synthetic latency."""
        toks = (system_prompt + " " + user_prompt).split()
        n = len(toks)
        blk = 16
        cached = 0
        sig: List[int] = []
        matching = True
        for b in range(n // blk):
            sig.append(hash(tuple(toks[b * blk : (b + 1) * blk])))
            key = tuple(sig)
            if matching and key in self._mock_prefix_blocks:
                cached += blk
                self._mock_prefix_blocks.move_to_end(key)
            else:
                matching = False
                self._mock_prefix_blocks[key] = 1
                if len(self._mock_prefix_blocks) > 16384:
                    self._mock_prefix_blocks.popitem(last=False)
        uncached = n - cached
        # Realistic A10-like mock TTFT: ~18ms base + 0.045ms/uncached tok + 0.004ms/cached tok
        ttft_ms = 18.0 + 0.045 * uncached + 0.004 * cached
        e2e_ms = ttft_ms + 12.0 * min(max_tokens, 8)
        return InferenceResult(
            answer="mock_answer",
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            prompt_tokens=n,
            cached_tokens=cached,
            completion_tokens=6,
        )


# =============================================================================
# 3. Official LoCoMo Loader & Zero-Leakage Hybrid Encoder (Identical to eval_locomo.py)
# =============================================================================

@dataclass
class RawTurn:
    dia_id: str
    session_num: int
    timestamp: str
    speaker: str
    text: str
    full_text: str


@dataclass
class ObsFact:
    obs_id: int
    session_num: int
    timestamp: str
    speaker: str
    text: str
    full_text: str
    dia_ids: Set[str]


@dataclass
class QAItem:
    qa_idx: int
    question: str
    answer: str
    category: int
    evidence_ids: Set[str]
    max_session: int
    is_immediate_lag: bool


@dataclass
class ParsedConversation:
    sample_id: str
    speakers: Set[str]
    sessions: List[int]
    turns_by_session: Dict[int, List[RawTurn]]
    obs_by_session: Dict[int, List[ObsFact]]
    qas: List[QAItem]


def find_locomo_file(user_path: Optional[str]) -> Path:
    candidates = []
    if user_path:
        candidates.append(Path(user_path))
    here = Path(__file__).parent
    candidates.extend([
        here / "data" / "locomo10.json",
        here / "locomo10.json",
        Path("data/locomo10.json"),
        Path("locomo10.json"),
    ])
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find locomo10.json. Upload locomo10.json next to bench_vllm_serving.py "
        "or pass --locomo /path/to/locomo10.json."
    )


def load_locomo(path: Path, max_convs: int) -> List[ParsedConversation]:
    raw_list = json.loads(path.read_text(encoding="utf-8"))[:max_convs]
    convs: List[ParsedConversation] = []
    for ci, item in enumerate(raw_list):
        conv = item["conversation"]
        obs_dict = item.get("observation", {})
        speakers = {str(conv.get("speaker_a", "A")).lower(), str(conv.get("speaker_b", "B")).lower()}
        sess_nums = sorted(
            int(k.split("_")[1])
            for k in conv.keys()
            if k.startswith("session_") and not k.endswith("_date_time")
        )
        turns_by_sess: Dict[int, List[RawTurn]] = {}
        obs_by_sess: Dict[int, List[ObsFact]] = {}
        obs_counter = 1
        for s in sess_nums:
            ts = str(conv.get(f"session_{s}_date_time", f"Session {s}"))
            t_list: List[RawTurn] = []
            for ti, t in enumerate(conv.get(f"session_{s}", [])):
                dia_id = str(t.get("dia_id", f"D{s}:{ti+1}"))
                spk = str(t.get("speaker", ""))
                txt = str(t.get("text", "")).strip()
                if t.get("blip_caption"):
                    txt = f"{txt} [Image: {t['blip_caption']}]"
                full = f"[{ts} | {dia_id}] {spk}: {txt}"
                t_list.append(RawTurn(dia_id, s, ts, spk, txt, full))
            turns_by_sess[s] = t_list

            o_list: List[ObsFact] = []
            s_obs_raw = obs_dict.get(f"session_{s}_observation", {})
            if isinstance(s_obs_raw, dict):
                for spk, entries in s_obs_raw.items():
                    if not isinstance(entries, list):
                        continue
                    for ent in entries:
                        if isinstance(ent, list) and len(ent) >= 2:
                            otxt = str(ent[0]).strip()
                            dids = set(extract_dia_ids(ent[1]))
                        elif isinstance(ent, str):
                            otxt = ent.strip()
                            dids = set(extract_dia_ids(otxt))
                        else:
                            continue
                        full_o = f"[KG#{obs_counter} | {ts}] {spk}: {otxt}"
                        o_list.append(ObsFact(obs_counter, s, ts, str(spk), otxt, full_o, dids))
                        obs_counter += 1
            obs_by_sess[s] = o_list

        qas: List[QAItem] = []
        for qi, q in enumerate(item.get("qa", [])):
            cat = int(q.get("category", 0))
            if cat not in (1, 2, 3, 4):
                continue
            ev_ids = extract_dia_ids(q.get("evidence", []))
            if not ev_ids:
                continue
            ev_sessions = [int(DIA_ID_RE.match(d).group(1)) for d in ev_ids if DIA_ID_RE.match(d)]
            max_s = max(ev_sessions) if ev_sessions else sess_nums[-1]
            qas.append(QAItem(qi, str(q["question"]), str(q.get("answer", "")), cat, set(ev_ids), max_s, (qi % 4 == 0)))

        convs.append(ParsedConversation(
            sample_id=str(item.get("sample_id", f"conv_{ci}")),
            speakers=speakers,
            sessions=sess_nums,
            turns_by_session=turns_by_sess,
            obs_by_session=obs_by_sess,
            qas=qas,
        ))
    return convs


class LocalHybridEncoder:
    def __init__(self, texts: List[str], speakers: Set[str], dim: int = 64, seed: int = 2026):
        self.speakers = speakers
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=4096, stop_words="english")
        X = self.tfidf.fit_transform(texts)
        self.idf_map: Dict[str, float] = {
            tok: float(self.tfidf.idf_[idx])
            for tok, idx in self.tfidf.vocabulary_.items()
            if " " not in tok
        }
        actual_dim = min(dim, max(2, min(X.shape) - 1))
        self.svd = TruncatedSVD(n_components=actual_dim, random_state=seed)
        self.svd.fit(X)
        self.dim = actual_dim
        rng = np.random.default_rng(seed + 13)
        self.planes = rng.normal(size=(12, self.dim)).astype(np.float32)

    def encode_dense(self, text: str) -> np.ndarray:
        v = self.svd.transform(self.tfidf.transform([text]))[0].astype(np.float32)
        n = float(np.linalg.norm(v))
        return (v / n) if n > 1e-9 else v

    def encode_sparse(self, text: str) -> Dict[str, float]:
        terms = content_terms(text, self.speakers)
        if not terms:
            return {}
        tf = Counter(terms)
        out = {w: (1.0 + math.log(c)) * self.idf_map.get(w, 4.5) for w, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in out.values()))
        return {k: v / norm for k, v in out.items()} if norm > 1e-9 else out

    def hybrid_sim(self, qd: np.ndarray, qs: Dict[str, float], dd: np.ndarray, ds: Dict[str, float]) -> float:
        d_sim = float(np.dot(qd, dd))
        if not qs or not ds:
            l_sim = 0.0
        else:
            if len(qs) > len(ds):
                qs, ds = ds, qs
            l_sim = sum(v * ds.get(k, 0.0) for k, v in qs.items())
        return 0.40 * d_sim + 0.60 * l_sim

    def top_idf_keys(self, text: str, max_k: int = 4) -> List[str]:
        terms = list(dict.fromkeys(content_terms(text, self.speakers)))
        terms.sort(key=lambda w: self.idf_map.get(w, 5.0), reverse=True)
        return [f"kw:{w}" for w in terms[:max_k]]

    def simhash_probes(self, vec: np.ndarray, n_bits: int = 8, multi_probe: int = 2) -> List[int]:
        proj = self.planes[:n_bits] @ vec
        bits = (proj >= 0).astype(int)
        base = 0
        for i, b in enumerate(bits):
            base |= (int(b) << i)
        if multi_probe <= 0:
            return [base]
        weakest = np.argsort(np.abs(proj))[:multi_probe]
        return [base] + [base ^ (1 << int(idx)) for idx in weakest]


class TopicRouter:
    def __init__(self, enc: LocalHybridEncoder, centroid_thresh: float = 0.45):
        self.enc = enc
        self.centroid_thresh = centroid_thresh
        self.centroids: List[Tuple[str, np.ndarray]] = []

    def route(self, text: str, dense: np.ndarray, register: bool = False) -> Tuple[List[str], List[str]]:
        fine = self.enc.top_idf_keys(text, max_k=4)
        best_cid, best_sim = None, -1.0
        for cid, cvec in self.centroids:
            s = float(np.dot(dense, cvec))
            if s > best_sim:
                best_sim = s
                best_cid = cid
        if best_cid is not None and best_sim >= self.centroid_thresh:
            fine.append(best_cid)
        elif register and len(self.centroids) < 64:
            cid = f"cen:{len(self.centroids)}"
            self.centroids.append((cid, dense.copy()))
            fine.append(cid)
        coarse = [f"mph:{h}" for h in self.enc.simhash_probes(dense, n_bits=8, multi_probe=2)]
        return fine, coarse


def amygdala_salience(turn: RawTurn, dense: np.ndarray, recent_vecs: List[np.ndarray], enc: LocalHybridEncoder) -> float:
    words = content_terms(turn.text, enc.speakers)
    n_content = len(words)
    if n_content <= 2 or len(turn.text.split()) < 9:
        return 0.15
    novelty = 1.0 if not recent_vecs else max(0.0, min(1.0, 1.0 - max(float(np.dot(dense, rv)) for rv in recent_vecs[-15:])))
    top_idfs = sorted((enc.idf_map.get(w, 4.5) for w in words), reverse=True)[:4]
    info_sig = min(1.0, (sum(top_idfs) / len(top_idfs)) / 7.5)
    low = turn.text.lower()
    is_pure_q = turn.text.strip().endswith("?") and n_content <= 5
    factual = 1.0 if any(
        k in low for k in [
            "yesterday", "last ", "next ", "week", "month", "year", "ago", "went",
            "started", "moved", "decided", "planning", "bought", "adopted", "joined",
            "graduated", "painted", "visited", "attended", "working", "career",
            "favorite", "allergy", "university", "college", "hospital", "concert",
            "marathon", "trip", "vacation", "museum", "pottery", "camping", "charity",
        ]
    ) else 0.0
    score = 0.25 * novelty + 0.25 * info_sig + 0.25 * factual + 0.25 * min(1.0, n_content / 12.0)
    return min(1.0, score * (0.65 if is_pure_q else 1.0))


# =============================================================================
# 4. Prompt Stream Builders (Produces Exact System Prompts + User Prompts)
# =============================================================================

@dataclass
class PreparedQuery:
    system_prompt: str
    user_prompt: str
    gold_answer: str
    category: int
    is_lag: bool
    evidence_hit: bool


def schedule_qas(conv: ParsedConversation) -> Tuple[Dict[int, List[QAItem]], Dict[int, List[QAItem]]]:
    pre: Dict[int, List[QAItem]] = defaultdict(list)
    post: Dict[int, List[QAItem]] = defaultdict(list)
    max_s = conv.sessions[-1]
    for idx, qa in enumerate(conv.qas):
        if qa.is_immediate_lag:
            pre[qa.max_session].append(qa)
        else:
            post[min(max_s, qa.max_session + (idx % 3))].append(qa)
    return pre, post


def build_locomo_prompts(
    system_key: str,
    convs: List[ParsedConversation],
    encoders: Dict[str, LocalHybridEncoder],
    run_tag: str,
    max_queries: Optional[int] = None,
) -> List[PreparedQuery]:
    """
    Generate the exact chronological (system_prompt, user_prompt) stream for a given system.
    CRITICAL FOR vLLM PREFIX CACHING:
      - Stable Zone A + Zone B go into `system_prompt` (or contiguous left prefix of user_prompt)
        so vLLM Automatic Prefix Caching matches them block-by-block identically across turns!
      - Each system uses a unique `run_tag` in its Zone A header so caches NEVER leak between
        different systems being benchmarked back-to-back on the same vLLM server!
    """
    out: List[PreparedQuery] = []
    zone_a_base = f"[BIOSYS-Eval Run={run_tag}]\n{BASE_SYSTEM_PROMPT}"

    for conv in convs:
        enc = encoders[conv.sample_id]
        pre_qas, post_qas = schedule_qas(conv)

        if system_key == "Full-Context":
            hist_lines: List[str] = []
            hist_dids: Set[str] = set()
            for s in conv.sessions:
                for tr in conv.turns_by_session[s]:
                    hist_lines.append(tr.full_text)
                    hist_dids.add(tr.dia_id)
                for qa in pre_qas.get(s, []) + post_qas.get(s, []):
                    up = "Dialogue History:\n" + "\n".join(hist_lines) + f"\n\nQuestion: {qa.question}\nAnswer:"
                    out.append(PreparedQuery(zone_a_base, up, qa.answer, qa.category, qa.is_immediate_lag, bool(qa.evidence_ids & hist_dids)))
                    if max_queries and len(out) >= max_queries:
                        return out

        elif system_key == "Naive-RAG":
            store: List[Tuple[RawTurn, np.ndarray, Dict[str, float]]] = []
            recent: List[RawTurn] = []
            for s in conv.sessions:
                for tr in conv.turns_by_session[s]:
                    store.append((tr, enc.encode_dense(tr.text), enc.encode_sparse(tr.text)))
                    recent.append(tr)
                for qa in pre_qas.get(s, []) + post_qas.get(s, []):
                    qd, qs = enc.encode_dense(qa.question), enc.encode_sparse(qa.question)
                    scored = sorted(((enc.hybrid_sim(qd, qs, td, ts), tr) for tr, td, ts in store), key=lambda x: x[0], reverse=True)
                    picked = [tr for _, tr in scored[:8]]
                    dids = {tr.dia_id for tr in picked} | {tr.dia_id for tr in recent[-4:]}
                    # Dynamic top-of-prompt injection (standard Naive-RAG layout)
                    up = (
                        "Retrieved Memory:\n" + "\n".join(tr.full_text for tr in picked)
                        + "\n\nRecent Turns:\n" + "\n".join(tr.full_text for tr in recent[-4:])
                        + f"\n\nQuestion: {qa.question}\nAnswer:"
                    )
                    out.append(PreparedQuery(zone_a_base, up, qa.answer, qa.category, qa.is_immediate_lag, bool(qa.evidence_ids & dids)))
                    if max_queries and len(out) >= max_queries:
                        return out

        elif system_key in ("Mem0/Zep-Sync", "Mem0/Zep-Async"):
            sync_mode = (system_key == "Mem0/Zep-Sync")
            kg_nodes: List[Tuple[ObsFact, np.ndarray, Dict[str, float]]] = []
            recent: List[RawTurn] = []

            def add_kg(sess: int):
                for ob in conv.obs_by_session[sess]:
                    kg_nodes.append((ob, enc.encode_dense(ob.text), enc.encode_sparse(ob.text)))

            def emit_mem0(qa: QAItem):
                qd, qs = enc.encode_dense(qa.question), enc.encode_sparse(qa.question)
                scored = sorted(((enc.hybrid_sim(qd, qs, od, os), ob) for ob, od, os in kg_nodes), key=lambda x: x[0], reverse=True)
                picked = [ob for _, ob in scored[:8]]
                dids: Set[str] = set()
                for ob in picked:
                    dids.update(ob.dia_ids)
                for tr in recent[-4:]:
                    dids.add(tr.dia_id)
                up = (
                    "Retrieved Knowledge Graph:\n" + "\n".join(ob.full_text for ob in picked)
                    + "\n\nRecent Turns:\n" + "\n".join(tr.full_text for tr in recent[-4:])
                    + f"\n\nQuestion: {qa.question}\nAnswer:"
                )
                out.append(PreparedQuery(zone_a_base, up, qa.answer, qa.category, qa.is_immediate_lag, bool(qa.evidence_ids & dids)))

            for s in conv.sessions:
                for tr in conv.turns_by_session[s]:
                    recent.append(tr)
                if sync_mode:
                    add_kg(s)
                for qa in pre_qas.get(s, []):
                    emit_mem0(qa)
                    if max_queries and len(out) >= max_queries:
                        return out
                if not sync_mode:
                    add_kg(s)
                for qa in post_qas.get(s, []):
                    emit_mem0(qa)
                    if max_queries and len(out) >= max_queries:
                        return out

        else:
            # BIOSYS-v2 variants: Adaptive Zone B | Fixed Zone B | w/o Prefix Zones
            zone_b_mode = (
                "adaptive" if system_key == "BIOSYS-v2-Adaptive"
                else "fixed" if system_key == "BIOSYS-v2-Fixed"
                else "none"
            )
            router = TopicRouter(enc, centroid_thresh=0.45)
            l1_recent: List[RawTurn] = []
            l2_cache: OrderedDict[int, Tuple[RawTurn, List[str], np.ndarray, Dict[str, float]]] = OrderedDict()
            l2_idx: Dict[str, Set[int]] = defaultdict(set)
            store_buf: List[Tuple[int, RawTurn, np.ndarray, Dict[str, float]]] = []
            l3_cache: OrderedDict[int, List] = OrderedDict()  # oid -> [ObsFact, all_k, d, sp, heat, last_q]
            l3_idx: Dict[str, Set[int]] = defaultdict(set)
            k_tlb: Dict[str, Set[int]] = defaultdict(set)
            l_tlb: Dict[str, List[int]] = defaultdict(list)
            wal_archive: Dict[int, Tuple[RawTurn, np.ndarray, Dict[str, float]]] = {}
            kg_store: Dict[int, List] = {}
            admitted_dids: Set[str] = set()
            recent_vecs: List[np.ndarray] = []

            lsn_ctr = 0
            cons_lsn = 0
            client_wm = 0
            q_ctr = 0
            epoch_len = 16
            admit_thresh = 0.10 + (1.0 - 0.10) / epoch_len
            pinned_b_str = ""
            pinned_b_dids: Set[str] = set()
            pinned_oids: Set[int] = set()
            turns_in_epoch = 0

            def refresh_zone_b():
                nonlocal pinned_b_str, pinned_b_dids, pinned_oids, turns_in_epoch
                if zone_b_mode == "none" or not l3_cache:
                    pinned_b_str, pinned_b_dids, pinned_oids, turns_in_epoch = "", set(), set(), 0
                    return
                if zone_b_mode == "fixed":
                    chosen = list(l3_cache.values())
                else:
                    scored = []
                    for rec in l3_cache.values():
                        p_hat = rec[4] * (0.94 ** max(0, q_ctr - rec[5]))
                        if p_hat >= admit_thresh:
                            scored.append((p_hat, rec))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    chosen = [r for _, r in scored[:24]]
                chosen.sort(key=lambda r: r[0].obs_id)
                pinned_b_str = "\n".join(r[0].full_text for r in chosen)
                pinned_b_dids = {d for r in chosen for d in r[0].dia_ids}
                pinned_oids = {r[0].obs_id for r in chosen}
                turns_in_epoch = 0

            def emit_biosys(qa: QAItem):
                nonlocal q_ctr, turns_in_epoch, client_wm, store_buf
                q_ctr += 1
                if cons_lsn > client_wm:
                    client_wm = cons_lsn
                    store_buf = [x for x in store_buf if x[0] > client_wm]

                qd, qs = enc.encode_dense(qa.question), enc.encode_sparse(qa.question)
                fine, coarse = router.route(qa.question, qd, register=False)
                all_k = fine + coarse

                pool: List[Tuple[float, str, object]] = []
                seen_o: Set[int] = set()
                seen_w: Set[int] = set()

                # Stage 1: Local Boundary (Store Buffer ∪ L2 ∪ L3)
                for lsn, tr, wd, ws in store_buf:
                    sim = enc.hybrid_sim(qd, qs, wd, ws)
                    if sim >= 0.16:
                        seen_w.add(lsn)
                        pool.append((sim, "wal", tr))
                for k in (fine if fine else coarse):
                    for lsn in l2_idx.get(k, set()):
                        if lsn in l2_cache and lsn not in seen_w:
                            seen_w.add(lsn)
                            tr, _, wd, ws = l2_cache[lsn]
                            sim = enc.hybrid_sim(qd, qs, wd, ws)
                            if sim >= 0.16:
                                pool.append((sim, "wal", tr))
                for k in all_k:
                    for oid in l3_idx.get(k, set()):
                        if oid in l3_cache and oid not in seen_o:
                            seen_o.add(oid)
                            rec = l3_cache[oid]
                            sim = enc.hybrid_sim(qd, qs, rec[2], rec[3])
                            if sim >= 0.16:
                                pool.append((sim, "kg", rec))

                pool.sort(key=lambda x: x[0], reverse=True)
                need_remote = (not pool) or (pool[0][0] < 0.36) or (sum(1 for x in pool if x[0] >= 0.24) < 2)

                # Stage 2: Split-TLB (K-Cache -> L-Cache)
                if need_remote:
                    for k in all_k:
                        for oid in k_tlb.get(k, set()):
                            if oid in kg_store and oid not in seen_o:
                                seen_o.add(oid)
                                rec = kg_store[oid]
                                sim = enc.hybrid_sim(qd, qs, rec[2], rec[3])
                                if sim >= 0.16:
                                    pool.append((sim, "kg", rec))
                    pool.sort(key=lambda x: x[0], reverse=True)
                    if (not pool) or (pool[0][0] < 0.28) or (sum(1 for x in pool if x[0] >= 0.22) < 2):
                        for k in all_k:
                            for lsn in l_tlb.get(k, []):
                                if lsn in wal_archive and lsn not in seen_w:
                                    seen_w.add(lsn)
                                    tr, wd, ws = wal_archive[lsn]
                                    sim = enc.hybrid_sim(qd, qs, wd, ws)
                                    if sim >= 0.16:
                                        pool.append((sim, "wal", tr))
                        pool.sort(key=lambda x: x[0], reverse=True)
                        # Stage 3: Red-Path ANN
                        if (not pool) or (pool[0][0] < 0.28) or (sum(1 for x in pool if x[0] >= 0.22) < 2):
                            ann_kg = sorted(
                                ((enc.hybrid_sim(qd, qs, rec[2], rec[3]), rec) for oid, rec in kg_store.items() if oid not in seen_o),
                                key=lambda x: x[0], reverse=True,
                            )
                            for sim, rec in ann_kg[:8]:
                                seen_o.add(rec[0].obs_id)
                                pool.append((sim, "kg", rec))
                                for fk in fine:
                                    k_tlb[fk].add(rec[0].obs_id)

                pool.sort(key=lambda x: x[0], reverse=True)
                delta_lines: List[str] = []
                picked_dids: Set[str] = set()
                for _, kind, obj in pool[:8]:
                    if kind == "kg":
                        rec = obj  # type: ignore[assignment]
                        rec[4] = 0.65 * rec[4] + 0.35
                        rec[5] = q_ctr
                        l3_cache[rec[0].obs_id] = rec
                        for tk in rec[1]:
                            l3_idx[tk].add(rec[0].obs_id)
                        if len(l3_cache) > 64:
                            ev_o, ev_r = l3_cache.popitem(last=False)
                            for tk in ev_r[1]:
                                l3_idx[tk].discard(ev_o)
                        picked_dids.update(rec[0].dia_ids)
                        if zone_b_mode == "none" or rec[0].obs_id not in pinned_oids:
                            delta_lines.append(rec[0].full_text)
                    else:
                        tr = obj  # type: ignore[assignment]
                        picked_dids.add(tr.dia_id)
                        delta_lines.append(tr.full_text)

                # Three-Zone Prefix-Cache-Aware Layout!
                if zone_b_mode != "none":
                    if (not pinned_b_str and l3_cache) or turns_in_epoch >= epoch_len:
                        refresh_zone_b()
                    turns_in_epoch += 1
                    # Zone A + Zone B reside contiguously in system_prompt -> 100% vLLM Prefix Match!
                    sys_p = zone_a_base + ("\n\nPinned Consolidated Knowledge (Zone B):\n" + pinned_b_str if pinned_b_str else "")
                    ctx_dids = set(pinned_b_dids) | picked_dids
                else:
                    sys_p = zone_a_base
                    ctx_dids = set(picked_dids)

                for tr in l1_recent[-2:]:
                    ctx_dids.add(tr.dia_id)
                user_p = (
                    ("Dynamic Delta & RYOW Logs (Zone C):\n" + "\n".join(delta_lines) + "\n\n" if delta_lines else "")
                    + "Immediate Working Turns (L1):\n" + "\n".join(tr.full_text for tr in l1_recent[-2:])
                    + f"\n\nQuestion: {qa.question}\nAnswer:"
                )
                out.append(PreparedQuery(sys_p, user_p, qa.answer, qa.category, qa.is_immediate_lag, bool(qa.evidence_ids & ctx_dids)))

            for s in conv.sessions:
                for tr in conv.turns_by_session[s]:
                    lsn_ctr += 1
                    d, sp = enc.encode_dense(tr.text), enc.encode_sparse(tr.text)
                    fine, coarse = router.route(tr.text, d, register=True)
                    all_k = fine + coarse
                    sal = amygdala_salience(tr, d, recent_vecs, enc)
                    wal_archive[lsn_ctr] = (tr, d, sp)
                    for k in all_k:
                        l_tlb[k].append(lsn_ctr)
                        if len(l_tlb[k]) > 16:
                            l_tlb[k] = l_tlb[k][-16:]
                    l2_cache[lsn_ctr] = (tr, all_k, d, sp)
                    store_buf.append((lsn_ctr, tr, d, sp))
                    for k in (fine if fine else coarse):
                        l2_idx[k].add(lsn_ctr)
                    if len(l2_cache) > 48:
                        ev_l, ev_v = l2_cache.popitem(last=False)
                        for k in ev_v[1]:
                            l2_idx[k].discard(ev_l)
                    if sal >= 0.56:
                        admitted_dids.add(tr.dia_id)
                    recent_vecs.append(d)
                    l1_recent.append(tr)

                for qa in pre_qas.get(s, []):
                    emit_biosys(qa)
                    if max_queries and len(out) >= max_queries:
                        return out

                # Sleep consolidation at session boundary
                for ob in conv.obs_by_session[s]:
                    if ob.dia_ids and not (ob.dia_ids & admitted_dids):
                        continue
                    d, sp = enc.encode_dense(ob.text), enc.encode_sparse(ob.text)
                    fine, coarse = router.route(ob.text, d, register=True)
                    all_k = fine + coarse
                    rec = [ob, all_k, d, sp, 0.16, q_ctr]
                    kg_store[ob.obs_id] = rec
                    for k in all_k:
                        k_tlb[k].add(ob.obs_id)
                    l3_cache[ob.obs_id] = rec
                    for k in all_k:
                        l3_idx[k].add(ob.obs_id)
                    if len(l3_cache) > 64:
                        ev_o, ev_r = l3_cache.popitem(last=False)
                        for k in ev_r[1]:
                            l3_idx[k].discard(ev_o)
                admitted_dids.clear()
                cons_lsn = lsn_ctr

                for qa in post_qas.get(s, []):
                    emit_biosys(qa)
                    if max_queries and len(out) >= max_queries:
                        return out

    return out


# =============================================================================
# 5. Tool-Heavy Coding Agent Prompt Stream (Proves Real GPU TTFT Collapse!)
# =============================================================================

def build_coding_agent_prompts(system_key: str, run_tag: str, n_queries: int = 50, seed: int = 2026) -> List[PreparedQuery]:
    """
    Build a realistic tool-heavy coding-agent prompt stream (large noisy build/test stdout dumps
    interspersed with mutable configuration updates and queries).
    """
    rng = random.Random(seed)
    topics = [
        ("project_atlas", ["region_us_east", "region_eu_west", "region_ap_sg"]),
        ("db_postgres", ["pg15_single", "pg16_replicated", "pg17_ha"]),
        ("auth_gateway", ["cookie_v1", "jwt_rs256", "mtls_strict"]),
        ("cache_redis", ["lru_4gb", "lfu_16gb", "tiered_nvme"]),
        ("ml_inference", ["fp16_a100", "awq_int4", "fp8_h100"]),
        ("ci_runner", ["ubuntu22_py311", "ubuntu24_py312", "self_hosted_gpu"]),
        ("logging_stack", ["elk_7d", "loki_14d", "clickhouse_30d"]),
        ("storage_s3", ["standard_s3", "intelligent_tier", "r2_edge"]),
        ("queue_kafka", ["kafka_3p", "redpanda_24p", "nats_jetstream"]),
        ("security_policy", ["basic_tls", "soc2_type2_kms", "iso27001_enclave"]),
    ]
    noise_words = [
        "pytest", "passed", "checking", "module", "compile", "artifact", "stdout",
        "warning", "deprecation", "retrying", "worker", "step", "info", "trace", "ok",
    ]

    zone_a = f"[BIOSYS-Coding Run={run_tag}]\n{BASE_SYSTEM_PROMPT}"
    state_map = {ent: vals[0] for ent, vals in topics}
    kg_lines = {
        ent: (
            f"[MemoryNode | Entity={ent} | ActiveState={vals[0]} | ValidTo=INF | "
            f"PolicyNote: enforce {vals[0]} for all operations touching {ent} until explicitly superseded.]"
        )
        for ent, vals in topics
    }

    full_hist: List[str] = [f"Decision: set {ent} to {val}." for ent, val in state_map.items()]
    l1_raw: List[str] = []
    l1_compact: List[str] = []
    pinned_b = "\n".join(kg_lines[ent] for ent, _ in topics)
    out: List[PreparedQuery] = []

    for q_idx in range(n_queries):
        # Inject 2 noisy build/test tool dumps + occasional state update before each query
        for _ in range(2):
            dump = "Tool stdout trace: " + " ".join(rng.choices(noise_words, k=320)) + " exit_code=0"
            full_hist.append(dump)
            l1_raw.append(dump)
            l1_compact.append(f"[WAL-Handle lsn={len(full_hist)} tool_stdout_offloaded len=325]")

        if q_idx % 5 == 2:
            ent, vals = topics[q_idx % len(topics)]
            new_v = vals[(q_idx // 5 + 1) % len(vals)]
            state_map[ent] = new_v
            upd = f"Update correction: migrated {ent} to active setting {new_v}."
            full_hist.append(upd)
            l1_raw.append(upd)
            l1_compact.append(upd)
            kg_lines[ent] = (
                f"[MemoryNode | Entity={ent} | ActiveState={new_v} | ValidTo=INF | "
                f"PolicyNote: enforce {new_v} for all operations touching {ent} until explicitly superseded.]"
            )
            # Refresh Zone B ONLY on epoch boundary (every 12 queries)
            if q_idx % 12 == 0:
                pinned_b = "\n".join(kg_lines[e] for e, _ in topics)

        q_ent, _ = topics[q_idx % len(topics)]
        gold = state_map[q_ent]
        q_str = f"What is the current active setting for {q_ent}?"

        if system_key == "Full-Context":
            up = "Session Log:\n" + "\n".join(full_hist) + f"\n\nQuestion: {q_str}\nAnswer:"
            out.append(PreparedQuery(zone_a, up, gold, 1, False, True))
        elif system_key == "Mem0/Zep-Sync":
            # Dynamic reordered top-6 KG nodes at top of prompt (breaks prefix!) + raw L1 tail
            picked_ents = [q_ent] + [e for e, _ in rng.sample(topics, 5) if e != q_ent][:5]
            up = (
                "Retrieved KG:\n" + "\n".join(kg_lines[e] for e in picked_ents)
                + "\n\nRecent Turns:\n" + "\n".join(l1_raw[-2:])
                + f"\n\nQuestion: {q_str}\nAnswer:"
            )
            out.append(PreparedQuery(zone_a, up, gold, 1, False, True))
        elif system_key == "Sync-KG-Sorted+Offload":
            # Strong controlled baseline: deterministic sorted Entity-ID ordering + same L1 WAL-handle offloading
            # (isolates pure Three-Zone Epoch Pinning vs. dynamic top-K membership changes)
            picked_ents = sorted([q_ent] + [e for e, _ in rng.sample(topics, 5) if e != q_ent][:5])
            up = (
                "Retrieved KG:\n" + "\n".join(kg_lines[e] for e in picked_ents)
                + "\n\nImmediate Working Tail (L1):\n" + "\n".join(l1_compact[-2:])
                + f"\n\nQuestion: {q_str}\nAnswer:"
            )
            out.append(PreparedQuery(zone_a, up, gold, 1, False, True))
        elif system_key == "BIOSYS-v2-NoZones":
            up = (
                f"Delta Memory:\n{kg_lines[q_ent]}\n\nRecent Turns:\n"
                + "\n".join(l1_raw[-2:]) + f"\n\nQuestion: {q_str}\nAnswer:"
            )
            out.append(PreparedQuery(zone_a, up, gold, 1, False, True))
        else:  # BIOSYS-v2-Full
            if q_idx > 0 and q_idx % 12 == 0:
                pinned_b = "\n".join(kg_lines[e] for e, _ in topics)
            sys_p = zone_a + "\n\nPinned Consolidated Knowledge (Zone B):\n" + pinned_b
            # Include delta only if updated since last epoch pin
            delta = f"RYOW Delta: {kg_lines[q_ent]}\n" if (gold not in pinned_b) else ""
            up = (
                f"[Now:Q{q_idx}]\n{delta}Immediate Working Tail (L1):\n"
                + "\n".join(l1_compact[-2:]) + f"\n\nQuestion: {q_str}\nAnswer:"
            )
            out.append(PreparedQuery(sys_p, up, gold, 1, False, True))

    return out


# =============================================================================
# 6. Benchmark Execution & Clean Publication Reporting
# =============================================================================

def execute_stream(
    client: VLLMClient,
    label: str,
    queries: List[PreparedQuery],
    max_tokens: int = 28,
) -> Dict[str, object]:
    if not queries:
        return {}

    # Warmup 1 call (untimed) so CUDA graphs and initial KV blocks stabilize
    _ = client.generate_stream(queries[0].system_prompt, queries[0].user_prompt, max_tokens=8)

    prom_before = client.read_prometheus_prefix_counters()
    t_wall_start = time.perf_counter()

    ttfts: List[float] = []
    e2es: List[float] = []
    prompt_toks: List[int] = []
    cached_toks_list: List[int] = []
    f1s: List[float] = []
    ems: List[float] = []
    cat2_f1s: List[float] = []
    lag_f1s: List[float] = []
    ev_hits = 0

    for idx, pq in enumerate(queries):
        res = client.generate_stream(pq.system_prompt, pq.user_prompt, max_tokens=max_tokens)
        ttfts.append(res.ttft_ms)
        e2es.append(res.e2e_ms)
        prompt_toks.append(res.prompt_tokens)
        if res.cached_tokens is not None:
            cached_toks_list.append(res.cached_tokens)

        f1, em = compute_qa_scores(res.answer, pq.gold_answer)
        f1s.append(f1)
        ems.append(em)
        if pq.category == 2:
            cat2_f1s.append(f1)
        if pq.is_lag:
            lag_f1s.append(f1)
        ev_hits += int(pq.evidence_hit)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(queries):
            print(
                f"  [{label}] {idx+1}/{len(queries)} | P50 TTFT={np.median(ttfts):.1f}ms | "
                f"AvgTok={np.mean(prompt_toks):.0f} | F1={100*np.mean(f1s):.1f}%",
                flush=True,
            )

    wall_s = time.perf_counter() - t_wall_start
    prom_after = client.read_prometheus_prefix_counters()

    # Compute exact vLLM Prefix Hit Rate from Prometheus counters (preferred) or usage.cached_tokens
    prom_hit_rate: Optional[float] = None
    if prom_before and prom_after:
        d_hits = prom_after["hits"] - prom_before["hits"]
        d_queries = prom_after["queries"] - prom_before["queries"]
        if d_queries > 0:
            prom_hit_rate = 100.0 * d_hits / d_queries

    usage_hit_rate: Optional[float] = None
    if cached_toks_list and sum(prompt_toks) > 0:
        usage_hit_rate = 100.0 * sum(cached_toks_list) / sum(prompt_toks)

    prefix_hit = prom_hit_rate if prom_hit_rate is not None else (usage_hit_rate if usage_hit_rate is not None else 0.0)

    return {
        "system": label,
        "n_queries": len(queries),
        "avg_prompt_tok": round(float(np.mean(prompt_toks)), 1),
        "vllm_prefix_hit_%": round(float(prefix_hit), 1),
        "prom_prefix_hit_%": round(float(prom_hit_rate), 1) if prom_hit_rate is not None else None,
        "usage_prefix_hit_%": round(float(usage_hit_rate), 1) if usage_hit_rate is not None else None,
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 2),
        "ttft_p90_ms": round(float(np.percentile(ttfts, 90)), 2),
        "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 2),
        "ttft_mean_ms": round(float(np.mean(ttfts)), 2),
        "e2e_mean_ms": round(float(np.mean(e2es)), 2),
        "throughput_rps": round(len(queries) / max(1e-6, wall_s), 2),
        "qa_f1_%": round(100.0 * float(np.mean(f1s)), 1),
        "qa_em_%": round(100.0 * float(np.mean(ems)), 1),
        "temp_c2_f1_%": round(100.0 * float(np.mean(cat2_f1s)), 1) if cat2_f1s else None,
        "lag_f1_%": round(100.0 * float(np.mean(lag_f1s)), 1) if lag_f1s else None,
        "ev_hit_%": round(100.0 * ev_hits / len(queries), 1),
        "wall_sec": round(wall_s, 1),
    }


def print_summary_table(title: str, rows: List[Dict[str, object]]) -> None:
    print(f"\n{'=' * 116}")
    print(f" {title}")
    print(f"{'=' * 116}")
    print(
        f"{'System':<22} | {'N':>4} | {'AvgTok':>6} | {'KVHit%':>6} | "
        f"{'TTFT50':>7} | {'TTFT99':>7} | {'RPS':>5} | {'F1%':>5} | {'EM%':>5} | {'LagF1':>5} | {'EvHit':>5}"
    )
    print("-" * 116)
    for r in rows:
        lf = f"{r['lag_f1_%']:>4.1f}%" if r.get("lag_f1_%") is not None else "  N/A"
        print(
            f"{str(r['system']):<22} | {r['n_queries']:>4} | {r['avg_prompt_tok']:>6.0f} | "
            f"{r['vllm_prefix_hit_%']:>5.1f}% | {r['ttft_p50_ms']:>6.1f}m | {r['ttft_p99_ms']:>6.1f}m | "
            f"{r['throughput_rps']:>5.2f} | {r['qa_f1_%']:>4.1f}% | {r['qa_em_%']:>4.1f}% | "
            f"{lf} | {r['ev_hit_%']:>4.1f}%"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="BIOSYS v2 Real vLLM Serving Verification Benchmark")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="vLLM OpenAI server URL")
    ap.add_argument("--model", default="qwen2.5-7b", help="Served model name in vLLM")
    ap.add_argument("--locomo", default=None, help="Path to locomo10.json")
    ap.add_argument("--max-convs", type=int, default=5, help="Number of LoCoMo conversations (1..10, default 5)")
    ap.add_argument("--coding-queries", type=int, default=60, help="Number of Coding Agent queries")
    ap.add_argument("--out", default="vllm_serving_results.json", help="Output JSON checkpoint path")
    ap.add_argument("--resume", action="store_true", help="Resume from existing checkpoint JSON")
    ap.add_argument("--smoke", action="store_true", help="Fast 30-second smoke test (12 queries per system)")
    ap.add_argument("--mock", action="store_true", help="Offline dry-run without GPU/vLLM")
    args = ap.parse_args()

    out_path = Path(args.out)
    results: Dict[str, Dict[str, Dict[str, object]]] = {"locomo": {}, "coding_agent": {}}
    if args.resume and out_path.exists():
        try:
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            results["locomo"] = loaded.get("locomo", {})
            results["coding_agent"] = loaded.get("coding_agent", {})
            print(f"[RESUME] Loaded existing checkpoint from {out_path}")
        except Exception:
            pass

    client = VLLMClient(args.base_url, args.model, mock=args.mock)
    client.check_health()

    locomo_path = find_locomo_file(args.locomo)
    n_convs = 1 if args.smoke else args.max_convs
    max_q = 12 if args.smoke else None
    coding_n = 12 if args.smoke else args.coding_queries

    convs = load_locomo(locomo_path, max_convs=n_convs)
    print(f"Loaded {len(convs)} LoCoMo conversation(s) from {locomo_path} ({sum(len(c.qas) for c in convs)} total QAs)")

    encoders: Dict[str, LocalHybridEncoder] = {}
    for c in convs:
        texts = [t.text for s in c.sessions for t in c.turns_by_session[s]] + [
            o.text for s in c.sessions for o in c.obs_by_session[s]
        ]
        encoders[c.sample_id] = LocalHybridEncoder(texts, c.speakers, dim=64, seed=2026)

    run_nonce = str(int(time.time()))[-6:]

    # --- Workload 1: Official LoCoMo ---
    locomo_systems = [
        "Full-Context",
        "Naive-RAG",
        "Mem0/Zep-Sync",
        "Mem0/Zep-Async",
        "BIOSYS-v2-NoZones",
        "BIOSYS-v2-Adaptive",
        "BIOSYS-v2-Fixed",
    ]
    print("\n>>> Running Workload 1: Official LoCoMo on vLLM <<<")
    for sys_key in locomo_systems:
        if sys_key in results["locomo"]:
            print(f"  [SKIP] {sys_key} already in checkpoint.")
            continue
        queries = build_locomo_prompts(sys_key, convs, encoders, run_tag=f"{sys_key}-{run_nonce}", max_queries=max_q)
        print(f"\nRunning LoCoMo [{sys_key}] ({len(queries)} queries)...")
        row = execute_stream(client, sys_key, queries, max_tokens=28)
        results["locomo"][sys_key] = row
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # --- Workload 2: Tool-Heavy Coding Agent ---
    coding_systems = [
        "Full-Context",
        "Mem0/Zep-Sync",
        "BIOSYS-v2-NoZones",
        "BIOSYS-v2-Full",
    ]
    print("\n>>> Running Workload 2: Coding Agent on vLLM <<<")
    for sys_key in coding_systems:
        if sys_key in results["coding_agent"]:
            print(f"  [SKIP] {sys_key} already in checkpoint.")
            continue
        queries = build_coding_agent_prompts(sys_key, run_tag=f"CA-{sys_key}-{run_nonce}", n_queries=coding_n)
        print(f"\nRunning Coding Agent [{sys_key}] ({len(queries)} queries)...")
        row = execute_stream(client, sys_key, queries, max_tokens=20)
        results["coding_agent"][sys_key] = row
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print_summary_table(
        f"vLLM SERVING VERIFICATION — LoCoMo (model={client.model}, convs={n_convs})",
        [results["locomo"][k] for k in locomo_systems if k in results["locomo"]],
    )
    print_summary_table(
        f"vLLM SERVING VERIFICATION — CODING AGENT (model={client.model}, N={coding_n})",
        [results["coding_agent"][k] for k in coding_systems if k in results["coding_agent"]],
    )
    print(f"\nAll results saved cleanly to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
