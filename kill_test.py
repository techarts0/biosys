#!/usr/bin/env python3
"""
BIOSYS v2 (Route A) Minimal Kill-Test Prototype
================================================
Zero-cost ($0) local benchmark rigorously testing the 3 core system hypotheses:
  H1. Client Split Semantic TLB (K-Cache / L-Cache) + 5-Tier Cascade:
      - Can a Hybrid Topic Encoder (Entity Hash + Multi-Probe Semantic Buckets)
        achieve high local/TLB interception even under paraphrased & implicit queries,
        while offloading server ANN compute & saving DMP network bandwidth?
  H2. Serving-Agnostic Prefix-Cache-Aware Three-Zone Layout:
      - Under exact block-aligned (B=16) Radix Prefix Cache matching (vLLM/API model),
        does BIOSYS-v2 simultaneously cut raw tokens AND preserve high KV prefix hit rate,
        beating Full-Context, Periodic-Compaction, Naive RAG, and Mem0/Zep-style baselines
        on *Amortized Effective Cost* (including background SLM consolidation cost)?
  H3. Dual-Path Hybrid Consistency (WAL Crash + Sleep Eventual) + DTF RYOW:
      - Under async consolidation lag (L = 5, 10, 20 turns), does DTF + Store-to-Load
        Forwarding eliminate Stale-Read violations to 0% without sacrificing hit rates?

Usage:
    python kill_test.py
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# 1. Realistic Vocabulary & Long-Horizon Agent Trace Generator
# ---------------------------------------------------------------------------

@dataclass
class TopicSpec:
    tid: int
    entity: str
    aliases: List[str]
    vocab: List[str]
    states: List[str]


TOPIC_TEMPLATES = [
    ("project_atlas", ["atlas", "main repo", "core service"], ["deploy", "pipeline", "docker", "kubernetes", "helm", "cluster", "container", "rollout"], ["region_us_east", "region_eu_west", "region_ap_sg", "region_us_west"]),
    ("db_postgres", ["postgres", "primary database", "sql backend"], ["index", "query", "migration", "schema", "pool", "replica", "vacuum", "transaction"], ["pg15_single", "pg16_replicated", "pg16_sharded", "pg17_ha"]),
    ("auth_gateway", ["oauth", "login service", "identity layer"], ["token", "jwt", "session", "refresh", "rbac", "scope", "secret", "sso"], ["cookie_v1", "jwt_rs256", "oauth_pkce", "mtls_strict"]),
    ("billing_api", ["stripe", "payment service", "invoice engine"], ["webhook", "charge", "refund", "currency", "ledger", "payout", "retry", "quota"], ["usd_only", "multi_currency", "usage_metered", "prepaid_credits"]),
    ("cache_redis", ["redis", "in-memory store", "kv cluster"], ["eviction", "ttl", "lru", "sharding", "sentinel", "snapshot", "latency", "memory"], ["lru_4gb", "lfu_16gb", "cluster_64gb", "tiered_nvme"]),
    ("ml_inference", ["vllm", "model server", "serving cluster"], ["prefill", "decode", "batch", "tensor", "quantization", "prefix", "gpu", "throughput"], ["fp16_a100", "awq_int4", "fp8_h100", "speculative_v2"]),
    ("user_location", ["home city", "residence", "office hub"], ["commute", "timezone", "weather", "flight", "address", "neighborhood", "local", "branch"], ["beijing", "shanghai", "shenzhen", "singapore"]),
    ("user_diet", ["food preference", "meal rule", "allergy profile"], ["restaurant", "menu", "peanut", "seafood", "vegan", "coffee", "spicy", "dinner"], ["no_restriction", "peanut_allergy", "low_carb", "vegetarian_no_nut"]),
    ("ci_runner", ["github actions", "build pipeline", "test runner"], ["pytest", "lint", "artifact", "matrix", "timeout", "cache", "coverage", "workflow"], ["ubuntu22_py311", "ubuntu24_py312", "arm64_py313", "self_hosted_gpu"]),
    ("logging_stack", ["grafana", "observability", "telemetry stack"], ["trace", "metric", "alert", "prometheus", "otel", "dashboard", "retention", "span"], ["elk_7d", "loki_14d", "clickhouse_30d", "clickhouse_90d"]),
    ("storage_s3", ["object store", "bucket storage", "archive tier"], ["parquet", "lifecycle", "glacier", "encryption", "multipart", "cdn", "etag", "prefix"], ["standard_s3", "intelligent_tier", "r2_edge", "minio_local"]),
    ("frontend_app", ["nextjs", "web client", "dashboard ui"], ["bundle", "hydration", "ssr", "component", "tailwind", "route", "state", "viewport"], ["react18_spa", "next14_app", "next15_rsc", "vite_edge"]),
    ("queue_kafka", ["kafka", "event bus", "message broker"], ["partition", "consumer", "offset", "dlq", "throughput", "retention", "avro", "lag"], ["kafka_3p", "kafka_12p", "redpanda_24p", "nats_jetstream"]),
    ("search_index", ["opensearch", "vector engine", "retrieval index"], ["hnsw", "bm25", "rerank", "embedding", "chunk", "recall", "shard", "filter"], ["bm25_only", "hnsw_fp32", "hybrid_pq", "diskann_binary"]),
    ("security_policy", ["compliance", "audit policy", "firewall rules"], ["soc2", "gdpr", "kms", "rotation", "pii", "masking", "zero_trust", "acl"], ["basic_tls", "soc2_type1", "soc2_type2_kms", "iso27001_enclave"]),
    ("team_schedule", ["sprint cadence", "release calendar", "standup plan"], ["milestone", "deadline", "review", "freeze", "hotfix", "demo", "roadmap", "owner"], ["biweekly_fri", "weekly_wed", "continuous_daily", "monthly_train"]),
    ("mobile_sdk", ["ios client", "android sdk", "app telemetry"], ["crash", "offline", "sync", "push", "battery", "sqlite", "release", "deeplink"], ["sdk_v2", "sdk_v3_offline", "sdk_v4_quic", "sdk_v5_edge"]),
    ("cost_budget", ["cloud spend", "finops cap", "api budget"], ["spot", "reserved", "quota", "alert", "token", "egress", "savings", "forecast"], ["cap_2k_usd", "cap_5k_usd", "cap_1k_strict", "zero_grant_local"]),
    ("docs_wiki", ["knowledge base", "runbook", "api spec"], ["openapi", "markdown", "diagram", "onboarding", "changelog", "rfc", "adr", "template"], ["confluence", "notion_sync", "git_docs_v2", "mintlify_auto"]),
    ("data_etl", ["airflow", "batch pipeline", "warehouse job"], ["dag", "dbt", "spark", "backfill", "warehouse", "sla", "partition", "lineage"], ["airflow_nightly", "dbt_hourly", "flink_streaming", "duckdb_micro"]),
]

NOISE_VOCAB = [
    "thanks", "okay", "sure", "got", "it", "please", "continue", "looks", "good",
    "hmm", "wait", "checking", "step", "output", "stdout", "info", "debug",
    "line", "warning", "retrying", "progress", "status", "done", "next",
]


@dataclass
class TurnEvent:
    turn_id: int
    session_id: int
    kind: str  # "fact_init" | "fact_update" | "query" | "chitchat" | "tool_noise"
    topic_id: int
    text: str
    tokens: List[str]
    token_count: int
    salience_gt: float  # true semantic importance (not visible to system directly)
    has_explicit_entity: bool
    state_value: Optional[str] = None  # ground truth state set or queried
    is_update: bool = False
    is_tool_dump: bool = False


def make_topic_specs() -> List[TopicSpec]:
    return [TopicSpec(i, ent, aliases, vocab, states) for i, (ent, aliases, vocab, states) in enumerate(TOPIC_TEMPLATES)]


def build_trace(
    regime: str,
    n_turns: int = 200,
    seed: int = 42,
) -> Tuple[List[TurnEvent], List[TopicSpec], Dict[int, str]]:
    """
    Generate a realistic long-horizon agent trace with:
      - Zipfian topic locality + session topic clustering + long-tail callbacks
      - Mutable facts (updates that supersede older states)
      - Both explicit-entity queries and paraphrased/implicit queries (~30% implicit!)
      - Regime 'assistant': multi-session QA/personal assistant (LoCoMo/LongMemEval style)
      - Regime 'coding_agent': long-horizon coding/tool-use workflow with large noisy tool outputs
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    specs = make_topic_specs()
    n_topics = len(specs)

    # Zipf weights for realistic temporal/topic locality
    ranks = np.arange(1, n_topics + 1, dtype=float)
    zipf_p = (1.0 / (ranks ** 1.05))
    zipf_p /= zipf_p.sum()

    current_state: Dict[int, str] = {}
    state_idx: Dict[int, int] = {i: 0 for i in range(n_topics)}
    initialized: Set[int] = set()

    events: List[TurnEvent] = []
    n_sessions = 10
    turns_per_session = max(1, n_turns // n_sessions)

    for t in range(n_turns):
        session_id = t // turns_per_session
        # Each session focuses on 3-4 hot topics plus occasional callbacks
        sess_rng = random.Random(seed * 100 + session_id)
        hot_topics = sess_rng.sample(range(n_topics), k=4)

        def pick_topic(require_init: bool = False) -> int:
            pool = list(initialized) if require_init else list(range(n_topics))
            if not pool:
                return hot_topics[0]
            # 68% locality to session hot topics, 32% global Zipfian long-tail
            hot_in_pool = [x for x in hot_topics if x in pool]
            if hot_in_pool and rng.random() < 0.68:
                return rng.choice(hot_in_pool)
            probs = np.array([zipf_p[x] for x in pool], dtype=float)
            probs /= probs.sum()
            return int(np_rng.choice(pool, p=probs))

        # First 12 turns seed core topics
        if t < 12:
            tid = t % n_topics
            kind = "fact_init"
        else:
            r = rng.random()
            if regime == "coding_agent":
                if r < 0.10 and len(initialized) < n_topics:
                    kind = "fact_init"
                elif r < 0.25 and initialized:
                    kind = "fact_update"
                elif r < 0.50 and initialized:
                    kind = "query"
                elif r < 0.82:
                    kind = "tool_noise"
                else:
                    kind = "chitchat"
            else:
                if r < 0.10 and len(initialized) < n_topics:
                    kind = "fact_init"
                elif r < 0.27 and initialized:
                    kind = "fact_update"
                elif r < 0.57 and initialized:
                    kind = "query"
                elif r < 0.72:
                    kind = "tool_noise"
                else:
                    kind = "chitchat"

        if kind == "fact_init":
            uninit = [i for i in range(n_topics) if i not in initialized]
            tid = uninit[0] if uninit else pick_topic(require_init=False)
            spec = specs[tid]
            st = spec.states[0]
            current_state[tid] = st
            state_idx[tid] = 0
            initialized.add(tid)
            kws = " ".join(rng.sample(spec.vocab, 4))
            text = (
                f"Decision record for {spec.entity} ({spec.aliases[0]}): "
                f"we configured active setting to {st} covering {kws}. "
                f"Please remember this requirement permanently."
            )
            tokens = text.split()
            events.append(TurnEvent(t, session_id, kind, tid, text, tokens, len(tokens), 0.90, True, st, False, False))

        elif kind == "fact_update":
            tid = pick_topic(require_init=True)
            spec = specs[tid]
            old_st = current_state[tid]
            nxt = (state_idx[tid] + 1) % len(spec.states)
            state_idx[tid] = nxt
            new_st = spec.states[nxt]
            current_state[tid] = new_st
            kws = " ".join(rng.sample(spec.vocab, 4))
            use_explicit = rng.random() < 0.78
            ref = spec.entity if use_explicit else rng.choice(spec.aliases)
            text = (
                f"Important update correction for {ref}: instead of {old_st}, "
                f"we migrated and switched current policy to {new_st} regarding {kws}."
            )
            tokens = text.split()
            events.append(TurnEvent(t, session_id, kind, tid, text, tokens, len(tokens), 0.95, use_explicit, new_st, True, False))

        elif kind == "query":
            tid = pick_topic(require_init=True)
            spec = specs[tid]
            gt_st = current_state[tid]
            # 30% implicit/paraphrased queries WITHOUT exact entity string!
            use_explicit = rng.random() < 0.70
            kws = " ".join(rng.sample(spec.vocab, 4))
            if use_explicit:
                text = f"Question about {spec.entity}: what is our latest active configuration for {kws}?"
            else:
                alias = rng.choice(spec.aliases)
                text = f"Question about the {alias} we discussed earlier: what is the current setup for {kws}?"
            tokens = text.split()
            events.append(TurnEvent(t, session_id, kind, tid, text, tokens, len(tokens), 0.35, use_explicit, gt_st, False, False))

        elif kind == "tool_noise":
            tid = pick_topic(require_init=False)
            spec = specs[tid]
            # Large verbose tool output (stdout/build logs) simulating context bloat!
            dump_len = rng.randint(240, 560) if regime == "coding_agent" else rng.randint(90, 200)
            body_words = [rng.choice(NOISE_VOCAB) for _ in range(dump_len - 12)]
            # Sprinkle occasional topic words into verbose log
            for pos in range(0, len(body_words), 35):
                body_words[pos] = rng.choice(spec.vocab)
            text = (
                f"Tool stdout log trace during {spec.aliases[0]} check: "
                + " ".join(body_words)
                + " exit_code=0"
            )
            tokens = text.split()
            events.append(TurnEvent(t, session_id, kind, tid, text, tokens, len(tokens), 0.12, False, None, False, True))

        else:  # chitchat
            tid = pick_topic(require_init=False)
            spec = specs[tid]
            filler = " ".join(rng.choices(NOISE_VOCAB, k=rng.randint(14, 32)))
            text = f"Quick acknowledgement note: {filler} regarding {rng.choice(spec.vocab)}."
            tokens = text.split()
            events.append(TurnEvent(t, session_id, kind, tid, text, tokens, len(tokens), 0.08, False, None, False, False))

    return events, specs, current_state


# ---------------------------------------------------------------------------
# 2. Exact Block-Aligned Radix Prefix Cache Simulator (vLLM / API Model)
# ---------------------------------------------------------------------------

class RadixPrefixCache:
    """
    Simulates real serving-engine / API automatic prefix caching:
      - Prompt tokens are hashed in contiguous blocks of size B=16 (PagedAttention block size).
      - A block hits ONLY IF all preceding blocks match identically (exact prefix stability).
      - Bounded capacity with LRU eviction across cached prompt prefixes.
    """

    def __init__(self, block_size: int = 16, max_blocks: int = 4096):
        self.block_size = block_size
        self.max_blocks = max_blocks
        # Maps cumulative prefix hash -> LRU timestamp
        self.blocks: OrderedDict[Tuple[int, ...], int] = OrderedDict()
        self.step = 0

    def query_and_insert(self, tokens: List[str]) -> Tuple[int, int]:
        """Return (cached_tokens, uncached_tokens) for this prompt."""
        self.step += 1
        n = len(tokens)
        n_full_blocks = n // self.block_size
        cached = 0
        prefix_sig: List[int] = []
        still_matching = True

        for b in range(n_full_blocks):
            block_tup = tuple(tokens[b * self.block_size : (b + 1) * self.block_size])
            h = hash(block_tup)
            prefix_sig.append(h)
            key = tuple(prefix_sig)
            if still_matching and key in self.blocks:
                cached += self.block_size
                self.blocks.move_to_end(key)
                self.blocks[key] = self.step
            else:
                still_matching = False
                self.blocks[key] = self.step
                if len(self.blocks) > self.max_blocks:
                    self.blocks.popitem(last=False)

        uncached = n - cached
        return cached, uncached


# ---------------------------------------------------------------------------
# 3. Real Lightweight Semantic Embedder & Topic Encoders
# ---------------------------------------------------------------------------

class LocalEmbedder:
    """
    Lightweight local embedder (TF-IDF + TruncatedSVD 64-d unit vectors),
    analogous to a tiny local client encoder (~few MB, <1ms per query).
    """

    def __init__(self, corpus_texts: List[str], dim: int = 48, seed: int = 42):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=2048)
        X = self.vec.fit_transform(corpus_texts)
        actual_dim = min(dim, max(2, min(X.shape) - 1))
        self.svd = TruncatedSVD(n_components=actual_dim, random_state=seed)
        self.svd.fit(X)
        self.dim = actual_dim
        # Random hyperplanes for Multi-Probe SimHash (LSH)
        rng = np.random.default_rng(seed + 7)
        self.planes = rng.normal(size=(12, self.dim))

    def encode(self, text: str) -> np.ndarray:
        x = self.vec.transform([text])
        v = self.svd.transform(x)[0].astype(np.float32)
        norm = float(np.linalg.norm(v))
        if norm > 1e-9:
            v /= norm
        return v

    def simhash_probes(self, vec: np.ndarray, n_bits: int = 8, multi_probe: int = 0) -> List[int]:
        proj = self.planes[:n_bits] @ vec
        bits = (proj >= 0).astype(int)
        base = 0
        for i, b in enumerate(bits):
            base |= (int(b) << i)
        if multi_probe <= 0:
            return [base]
        # Flip the `multi_probe` weakest-margin bits (classic Multi-Probe LSH)
        margins = np.argsort(np.abs(proj))[:multi_probe]
        probes = [base]
        for idx in margins:
            probes.append(base ^ (1 << int(idx)))
        return probes


class TopicEncoder:
    """
    Supports 3 modes to honestly test the Topic Encoder upgrade:
      - 'keyword_only'  : original Fig 2 BERT Keywords (exact known entity/alias match only)
      - 'pure_lsh'      : original idea.md pure 8-bit SimHash (single bucket)
      - 'hybrid_v2'     : design_v2.md Hybrid (Entity Hash + Online Centroid Buckets + 2-Probe SimHash)
    """

    def __init__(self, mode: str, embedder: LocalEmbedder, specs: List[TopicSpec], centroid_thresh: float = 0.42):
        self.mode = mode
        self.embedder = embedder
        self.entity_vocab: Set[str] = {s.entity for s in specs}
        self.alias_to_entity: Dict[str, str] = {}
        for s in specs:
            self.alias_to_entity[s.entity] = s.entity
            for al in s.aliases:
                self.alias_to_entity[al] = s.entity
        self.centroids: List[Tuple[str, np.ndarray]] = []
        self.centroid_thresh = centroid_thresh

    def extract_entity_key(self, text: str) -> Optional[str]:
        low = text.lower()
        for phrase, canon in self.alias_to_entity.items():
            if phrase in low:
                return f"ent:{canon}"
        return None

    def encode_keys(self, text: str, vec: np.ndarray, register: bool = False) -> Tuple[List[str], List[str]]:
        """
        Return (fine_keys, coarse_keys):
          - fine_keys: exact entity/alias hashes and tight semantic centroids (safe for DTF dirty bits & exact tag match)
          - coarse_keys: multi-probe SimHash buckets (only used as candidate index, MUST pass Tag Verification!)
        """
        fine_keys: List[str] = []
        coarse_keys: List[str] = []

        if self.mode == "keyword_only":
            low = text.lower()
            # Strict exact canonical entity only (fails on aliases/paraphrases)
            for ent in self.entity_vocab:
                if ent in low:
                    fine_keys.append(f"ent:{ent}")
            return (fine_keys if fine_keys else ["kw:unknown"]), []

        if self.mode == "pure_lsh":
            h = self.embedder.simhash_probes(vec, n_bits=8, multi_probe=0)[0]
            return [], [f"lsh:{h}"]

        # hybrid_v2: exact entity/alias hash + tight online semantic centroid + multi-probe SimHash
        ek = self.extract_entity_key(text)
        if ek:
            fine_keys.append(ek)

        best_cid, best_sim = None, -1.0
        for cid, cvec in self.centroids:
            sim = float(np.dot(vec, cvec))
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        if best_cid is not None and best_sim >= self.centroid_thresh:
            fine_keys.append(best_cid)
        elif register:
            new_cid = f"cen:{len(self.centroids)}"
            self.centroids.append((new_cid, vec.copy()))
            fine_keys.append(new_cid)

        for h in self.embedder.simhash_probes(vec, n_bits=8, multi_probe=2):
            coarse_keys.append(f"mph:{h}")

        return fine_keys, coarse_keys


# ---------------------------------------------------------------------------
# 4. Data Records for BIOSYS Storage Hierarchy & Consistency
# ---------------------------------------------------------------------------

@dataclass
class RawLogEntry:
    lsn: int
    turn_id: int
    topic_id: int
    topic_keys: List[str]
    text: str
    tokens: List[str]
    vec: np.ndarray
    state_value: Optional[str]
    is_update: bool
    salience_score: float


@dataclass
class KGNode:
    node_id: int
    topic_id: int
    topic_keys: List[str]
    summary_text: str
    tokens: List[str]
    vec: np.ndarray
    active_state: Optional[str]
    valid_from: int
    valid_to: Optional[int]  # None => infinity (active)
    source_lsn: int
    version: int = 1


# ---------------------------------------------------------------------------
# 5. System Simulators (Baselines + BIOSYS-v1 + BIOSYS-v2 & Ablations)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TOKENS = (
    "You are an autonomous enterprise engineering assistant. Follow safety rules, "
    "use tool outputs carefully, track mutable user constraints over time, resolve temporal "
    "conflicts by preferring newer verified facts, and keep responses concise and accurate. "
    "Available tools: read_file write_file run_tests query_db deploy_service inspect_logs "
    "check_metrics update_config search_docs notify_team rollback_release verify_auth "
).split() * 4  # ~256 tokens realistic system + tool schemas prefix


@dataclass
class EvalStats:
    name: str
    regime: str
    total_raw_tokens: int = 0
    total_cached_tokens: int = 0
    total_uncached_tokens: int = 0
    slm_consolidation_tokens: int = 0
    slm_calls: int = 0
    slm_skipped_by_amygdala: int = 0
    dmp_network_bytes: int = 0
    query_count: int = 0
    hit_funnel: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Accuracy / Consistency on ground-truth queries
    gt_recall_hits: int = 0          # Context contains current latest ground-truth state
    stale_read_violations: int = 0   # Context has outdated state WITHOUT latest state!
    missing_fact_misses: int = 0     # Neither latest nor outdated state retrieved
    client_lookup_us_total: float = 0.0

    def summary(self, cache_discount: float = 0.10, slm_cost_ratio: float = 0.15) -> Dict[str, object]:
        prefix_hit_rate = self.total_cached_tokens / max(1, self.total_raw_tokens)
        eff_infer_cost = cache_discount * self.total_cached_tokens + self.total_uncached_tokens
        amort_cost_slm = eff_infer_cost + slm_cost_ratio * self.slm_consolidation_tokens
        amort_cost_worst = eff_infer_cost + 1.0 * self.slm_consolidation_tokens
        q = max(1, self.query_count)
        return {
            "name": self.name,
            "avg_raw_tok": round(self.total_raw_tokens / q, 1),
            "prefix_hit_%": round(100.0 * prefix_hit_rate, 1),
            "eff_infer_tok/q": round(eff_infer_cost / q, 1),
            "slm_tok/q": round(self.slm_consolidation_tokens / q, 1),
            "amort_cost_0.15x": round(amort_cost_slm / q, 1),
            "amort_cost_1.0x": round(amort_cost_worst / q, 1),
            "net_kb/q": round(self.dmp_network_bytes / (1024.0 * q), 2),
            "latest_recall_%": round(100.0 * self.gt_recall_hits / q, 1),
            "stale_read_%": round(100.0 * self.stale_read_violations / q, 2),
            "ann_rate_%": round(100.0 * self.hit_funnel.get("ANN", 0) / q, 1),
            "funnel": dict(self.hit_funnel),
        }


def check_context_ground_truth(
    context_text: str,
    gt_state: str,
    spec: TopicSpec,
) -> Tuple[bool, bool]:
    """
    Evaluate whether assembled context contains:
      - has_latest: latest ground-truth state is present
      - is_stale_violation: an OLD superseded state for this topic is present WITHOUT the latest state
    """
    has_latest = gt_state in context_text
    has_old = any((st != gt_state and st in context_text) for st in spec.states)
    is_stale = has_old and (not has_latest)
    return has_latest, is_stale


def run_full_context(events: List[TurnEvent], specs: List[TopicSpec], regime: str, compact_limit: Optional[int] = None) -> EvalStats:
    name = "Full-Context" if compact_limit is None else f"Full+Compact({compact_limit})"
    stats = EvalStats(name=name, regime=regime)
    cache = RadixPrefixCache(block_size=16, max_blocks=16384)
    history: List[List[str]] = []
    history_texts: List[str] = []

    for ev in events:
        if ev.kind == "query":
            stats.query_count += 1
            prompt_tokens = list(SYSTEM_PROMPT_TOKENS)
            for h in history:
                prompt_tokens.extend(h)
            prompt_tokens.extend(ev.tokens)

            cached, uncached = cache.query_and_insert(prompt_tokens)
            stats.total_raw_tokens += len(prompt_tokens)
            stats.total_cached_tokens += cached
            stats.total_uncached_tokens += uncached
            stats.hit_funnel["FULL"] += 1

            ctx_text = " ".join(history_texts)
            has_latest, is_stale = check_context_ground_truth(ctx_text, ev.state_value or "", specs[ev.topic_id])
            stats.gt_recall_hits += int(has_latest)
            stats.stale_read_violations += int(is_stale)
            stats.missing_fact_misses += int((not has_latest) and (not is_stale))

        history.append(ev.tokens)
        history_texts.append(ev.text)

        # Periodic compaction when exceeding budget (nukes prefix & compresses older turns lossily!)
        if compact_limit is not None:
            cur_len = len(SYSTEM_PROMPT_TOKENS) + sum(len(h) for h in history)
            if cur_len > compact_limit:
                keep_recent = 6
                older = history[:-keep_recent]
                older_txt = history_texts[:-keep_recent]
                # Summarization call consumes older tokens as SLM/LLM compaction cost
                older_tok_cnt = sum(len(x) for x in older)
                stats.slm_consolidation_tokens += older_tok_cnt
                stats.slm_calls += 1
                # Lossy compaction keeps ~65% of facts (realistic summary drift/compression loss)
                kept_lines = [txt for i, txt in enumerate(older_txt) if ("Decision" in txt or "update" in txt) and (i % 3 != 0)]
                summary_txt = "Summary: " + " | ".join(kept_lines[-10:])
                history = [summary_txt.split()] + history[-keep_recent:]
                history_texts = [summary_txt] + history_texts[-keep_recent:]

    return stats


def run_naive_rag(events: List[TurnEvent], specs: List[TopicSpec], embedder: LocalEmbedder, regime: str, top_k: int = 5) -> EvalStats:
    stats = EvalStats(name="Naive-RAG (Top-K)", regime=regime)
    cache = RadixPrefixCache(block_size=16, max_blocks=8192)
    store: List[Tuple[np.ndarray, TurnEvent]] = []
    l1_window: List[TurnEvent] = []

    for ev in events:
        vec = embedder.encode(ev.text)
        if ev.kind == "query":
            stats.query_count += 1
            stats.hit_funnel["ANN"] += 1
            # Full vector scan on server
            scored = [(float(np.dot(vec, sv)), sev) for sv, sev in store]
            scored.sort(key=lambda x: x[0], reverse=True)
            picked = [sev for _, sev in scored[:top_k]]

            # Standard RAG puts retrieved chunks right after system prompt -> breaks downstream prefix!
            prompt_tokens = list(SYSTEM_PROMPT_TOKENS)
            ctx_parts = []
            for p in picked:
                prompt_tokens.extend(p.tokens)
                ctx_parts.append(p.text)
                stats.dmp_network_bytes += len(p.text.encode("utf-8"))
            for w in l1_window[-3:]:
                prompt_tokens.extend(w.tokens)
                ctx_parts.append(w.text)
            prompt_tokens.extend(ev.tokens)

            cached, uncached = cache.query_and_insert(prompt_tokens)
            stats.total_raw_tokens += len(prompt_tokens)
            stats.total_cached_tokens += cached
            stats.total_uncached_tokens += uncached

            has_latest, is_stale = check_context_ground_truth(" ".join(ctx_parts), ev.state_value or "", specs[ev.topic_id])
            stats.gt_recall_hits += int(has_latest)
            stats.stale_read_violations += int(is_stale)
            stats.missing_fact_misses += int((not has_latest) and (not is_stale))

        store.append((vec, ev))
        l1_window.append(ev)

    return stats


def format_kg_summary(nid: int, spec: TopicSpec, state_val: str, turn_or_lsn: int) -> str:
    """
    Realistic structured consolidated knowledge slice (~52 tokens),
    used identically by BOTH Mem0/Zep baselines and BIOSYS for fair comparison.
    """
    aliases_str = ", ".join(spec.aliases)
    vocab_str = ", ".join(spec.vocab)
    return (
        f"[MemoryNode#{nid} | Entity={spec.entity} | Aliases=({aliases_str}) | "
        f"ActiveState={state_val} | Scope=({vocab_str}) | ValidFrom={turn_or_lsn} | "
        f"ValidTo=INF | Confidence=0.95 | Provenance=verified_session_consolidation "
        f"PolicyNote: enforce {state_val} for all operations involving {spec.entity} and {spec.aliases[0]} "
        f"until explicitly superseded by a newer timestamped user instruction.]"
    )


def run_mem0_zep_style(
    events: List[TurnEvent],
    specs: List[TopicSpec],
    embedder: LocalEmbedder,
    regime: str,
    async_lag: int = 10,
    sync_mode: bool = False,
) -> EvalStats:
    """
    Represents external fact/KG memory systems (Mem0 / Zep-style):
      - Extracts structured facts per turn (sync or async sleep batch) without salience write gating
      - Every query hits server ANN/KG and injects dynamic top-K memory block right after System Prompt
        (unstable prefix layout).
    """
    label = "Mem0/Zep (Sync-KG)" if sync_mode else f"Mem0/Zep (Async-Lag{async_lag})"
    stats = EvalStats(name=label, regime=regime)
    cache = RadixPrefixCache(block_size=16, max_blocks=8192)
    kg_by_topic: Dict[int, KGNode] = {}
    pending_buf: List[Tuple[TurnEvent, np.ndarray]] = []
    l1_window: List[TurnEvent] = []
    next_nid = 1

    def consolidate(batch: List[Tuple[TurnEvent, np.ndarray]]):
        nonlocal next_nid
        for bev, bvec in batch:
            # No write admission gate: every turn pays SLM extraction tokens!
            stats.slm_consolidation_tokens += bev.token_count + 40
            stats.slm_calls += 1
            if bev.state_value is not None:
                spec = specs[bev.topic_id]
                summary = format_kg_summary(next_nid, spec, bev.state_value, bev.turn_id)
                kg_by_topic[bev.topic_id] = KGNode(
                    node_id=next_nid,
                    topic_id=bev.topic_id,
                    topic_keys=[],
                    summary_text=summary,
                    tokens=summary.split(),
                    vec=bvec,
                    active_state=bev.state_value,
                    valid_from=bev.turn_id,
                    valid_to=None,
                    source_lsn=bev.turn_id,
                )
                next_nid += 1

    for ev in events:
        vec = embedder.encode(ev.text)
        if ev.kind == "query":
            stats.query_count += 1
            stats.hit_funnel["ANN"] += 1
            # Server KG retrieval (top-6 nodes by semantic similarity, ~300+ tok memory block)
            nodes = list(kg_by_topic.values())
            scored = [(float(np.dot(vec, nd.vec)), nd) for nd in nodes]
            scored.sort(key=lambda x: x[0], reverse=True)
            picked = [nd for _, nd in scored[:6]]

            # Dynamic top-of-prompt injection (standard Mem0/Zep integration) -> breaks prefix!
            prompt_tokens = list(SYSTEM_PROMPT_TOKENS)
            ctx_parts = []
            for nd in picked:
                prompt_tokens.extend(nd.tokens)
                ctx_parts.append(nd.summary_text)
                stats.dmp_network_bytes += len(nd.summary_text.encode("utf-8"))
            for w in l1_window[-2:]:
                prompt_tokens.extend(w.tokens)
                ctx_parts.append(w.text)
            prompt_tokens.extend(ev.tokens)

            cached, uncached = cache.query_and_insert(prompt_tokens)
            stats.total_raw_tokens += len(prompt_tokens)
            stats.total_cached_tokens += cached
            stats.total_uncached_tokens += uncached

            has_latest, is_stale = check_context_ground_truth(" ".join(ctx_parts), ev.state_value or "", specs[ev.topic_id])
            stats.gt_recall_hits += int(has_latest)
            stats.stale_read_violations += int(is_stale)
            stats.missing_fact_misses += int((not has_latest) and (not is_stale))

        if ev.kind != "query":
            pending_buf.append((ev, vec))
            if sync_mode or len(pending_buf) >= async_lag:
                consolidate(pending_buf)
                pending_buf = []
        l1_window.append(ev)

    return stats


def compute_amygdala_salience(
    ev: TurnEvent,
    vec: np.ndarray,
    recent_vecs: List[np.ndarray],
) -> float:
    """
    Zero-LLM lightweight salience gate S(x) in [0, 1] as defined in design_v2.md:
      S(x) = w1*Novelty + w2*Conflict + w3*ExplicitSignal + w4*ToolOutcome
    """
    # Novelty: min cosine distance against recent cached vectors
    if not recent_vecs:
        novelty = 1.0
    else:
        sims = [float(np.dot(vec, rv)) for rv in recent_vecs[-20:]]
        novelty = max(0.0, min(1.0, 1.0 - max(sims)))

    low = ev.text.lower()
    conflict = 1.0 if any(k in low for k in ["instead of", "switched", "migrated", "correction", "update"]) else 0.0
    explicit = 1.0 if any(k in low for k in ["remember", "permanently", "decision", "requirement", "policy"]) else 0.0
    # Filter verbose raw stdout dumps unless error/failure signal is present
    if ev.is_tool_dump:
        tool_sig = 0.8 if any(k in low for k in ["error", "exception", "failed", "root cause"]) else 0.0
    else:
        tool_sig = 0.25

    score = 0.25 * novelty + 0.35 * conflict + 0.25 * explicit + 0.15 * tool_sig
    return min(1.0, score)


def run_biosys(
    events: List[TurnEvent],
    specs: List[TopicSpec],
    embedder: LocalEmbedder,
    regime: str,
    name: str = "BIOSYS-v2 (Full)",
    encoder_mode: str = "hybrid_v2",
    enable_prefix_zones: bool = True,
    enable_dtf_forwarding: bool = True,
    enable_amygdala: bool = True,
    enable_l_cache: bool = True,
    consolidation_lag: int = 10,
    l2_capacity: int = 8,
    l3_capacity: int = 16,
    epoch_len: int = 16,
    tag_sim_threshold: float = 0.50,
) -> EvalStats:
    """
    Upgraded BIOSYS v2.1 faithful implementation (fixing 3 Kill-Test findings):
      1. Fine-grained DTF isolation (tracks entity/tight-centroid keys only -> eliminates False Sharing!)
      2. Hardware-style Semantic TLB Tag Verification (Index + Tag Match -> eliminates Aliasing Hits!)
      3. Low-salience L1 tool-dump offloading to Fast-Path WAL handles (keeps Zone C lean & Prefix Hit high!)
    """
    stats = EvalStats(name=name, regime=regime)
    cache = RadixPrefixCache(block_size=16, max_blocks=8192)
    encoder = TopicEncoder(mode=encoder_mode, embedder=embedder, specs=specs, centroid_thresh=0.52)

    # Client caches
    l1_window: List[List[str]] = []                                # compact L1 working tail tokens
    l1_texts: List[str] = []
    l2_cache: OrderedDict[str, List[RawLogEntry]] = OrderedDict()  # fine_key -> recent raw entries
    l3_cache: OrderedDict[int, KGNode] = OrderedDict()             # node_id -> KGNode
    l3_key_index: Dict[str, Set[int]] = defaultdict(set)           # any_key -> node_ids cached locally
    k_tlb: Dict[str, Set[int]] = defaultdict(set)                  # K-Cache: key -> remote NodeIDs
    l_tlb: Dict[str, List[int]] = defaultdict(list)                # L-Cache: key -> remote LSN offsets
    dtf_dirty: Dict[str, int] = {}                                 # DTF: FINE key only -> latest dirty LSN
    client_known_watermark: int = 0
    recent_vecs: List[np.ndarray] = []

    # Zone B Epoch Pinning state
    pinned_zone_b_tokens: List[str] = []
    pinned_zone_b_text: str = ""
    pinned_node_ids: Set[int] = set()
    turns_since_epoch_refresh: int = 0

    # Server Adapter & Persister state
    lsn_counter: int = 0
    consolidated_lsn: int = 0
    wal_archive: Dict[int, RawLogEntry] = {}                       # O(1) LSN offset direct read
    replay_buffer: List[RawLogEntry] = []
    temporal_kg: Dict[int, KGNode] = {}                            # node_id -> KGNode
    kg_by_topic: Dict[int, int] = {}                               # topic_id -> active node_id
    next_node_id: int = 1

    def tag_matches(query_fine: List[str], query_vec: np.ndarray, cand_keys: List[str], cand_vec: np.ndarray) -> bool:
        """Hardware-inspired TLB Tag Verification: exact entity match OR high semantic similarity."""
        q_ents = {k for k in query_fine if k.startswith("ent:")}
        c_ents = {k for k in cand_keys if k.startswith("ent:")}
        if q_ents and c_ents:
            return bool(q_ents & c_ents)
        return float(np.dot(query_vec, cand_vec)) >= tag_sim_threshold

    def refresh_zone_b():
        nonlocal pinned_zone_b_tokens, pinned_zone_b_text, pinned_node_ids, turns_since_epoch_refresh
        # Deterministic ascending NodeID serialization -> maximizes cross-turn Prefix Cache stability!
        ordered_nodes = sorted(l3_cache.values(), key=lambda nd: nd.node_id)
        parts = [nd.summary_text for nd in ordered_nodes]
        pinned_zone_b_text = " ".join(parts)
        pinned_zone_b_tokens = pinned_zone_b_text.split()
        pinned_node_ids = {nd.node_id for nd in ordered_nodes}
        turns_since_epoch_refresh = 0

    def sync_watermark(server_wm: int):
        nonlocal client_known_watermark
        if server_wm > client_known_watermark:
            client_known_watermark = server_wm
            clean_keys = [k for k, dlsn in dtf_dirty.items() if dlsn <= client_known_watermark]
            for k in clean_keys:
                del dtf_dirty[k]

    def server_sleep_consolidate():
        nonlocal consolidated_lsn, next_node_id
        if not replay_buffer:
            return
        max_lsn = max(entry.lsn for entry in replay_buffer)
        for entry in replay_buffer:
            stats.slm_consolidation_tokens += len(entry.tokens) + 32
            stats.slm_calls += 1
            if entry.state_value is not None:
                spec = specs[entry.topic_id]
                if entry.topic_id in kg_by_topic:
                    old_nid = kg_by_topic[entry.topic_id]
                    if old_nid in temporal_kg:
                        temporal_kg[old_nid].valid_to = entry.turn_id
                nid = next_node_id
                next_node_id += 1
                summary = format_kg_summary(nid, spec, entry.state_value, entry.lsn)
                nd = KGNode(
                    node_id=nid,
                    topic_id=entry.topic_id,
                    topic_keys=list(entry.topic_keys),
                    summary_text=summary,
                    tokens=summary.split(),
                    vec=entry.vec,
                    active_state=entry.state_value,
                    valid_from=entry.turn_id,
                    valid_to=None,
                    source_lsn=entry.lsn,
                )
                temporal_kg[nid] = nd
                kg_by_topic[entry.topic_id] = nid
                for tk in entry.topic_keys:
                    k_tlb[tk] = {x for x in k_tlb[tk] if temporal_kg[x].topic_id != entry.topic_id}
                    k_tlb[tk].add(nid)
                # Refresh L3 entry if topic already resident locally
                for old_id in list(l3_cache.keys()):
                    if l3_cache[old_id].topic_id == entry.topic_id:
                        del l3_cache[old_id]
                        for tk_set in l3_key_index.values():
                            tk_set.discard(old_id)
                l3_cache[nid] = nd
                for tk in entry.topic_keys:
                    l3_key_index[tk].add(nid)
                if len(l3_cache) > l3_capacity:
                    evicted_id, _ = l3_cache.popitem(last=False)
                    for tk_set in l3_key_index.values():
                        tk_set.discard(evicted_id)
        replay_buffer.clear()
        consolidated_lsn = max(consolidated_lsn, max_lsn)

    def write_back(ev: TurnEvent, vec: np.ndarray, fine_keys: List[str], coarse_keys: List[str]):
        nonlocal lsn_counter, consolidated_lsn
        lsn_counter += 1
        all_keys = fine_keys + coarse_keys
        sal = compute_amygdala_salience(ev, vec, recent_vecs) if enable_amygdala else 1.0
        entry = RawLogEntry(
            lsn=lsn_counter,
            turn_id=ev.turn_id,
            topic_id=ev.topic_id,
            topic_keys=all_keys,
            text=ev.text,
            tokens=ev.tokens,
            vec=vec,
            state_value=ev.state_value,
            is_update=ev.is_update,
            salience_score=sal,
        )

        # 1. Local L2 insert indexed by fine keys (or primary coarse key if pure_lsh)
        l2_keys = fine_keys if fine_keys else coarse_keys[:1]
        for lk in l2_keys:
            if lk in l2_cache:
                l2_cache.move_to_end(lk)
                l2_cache[lk].append(entry)
                l2_cache[lk] = l2_cache[lk][-2:]
            else:
                l2_cache[lk] = [entry]
                if len(l2_cache) > l2_capacity:
                    l2_cache.popitem(last=False)

        # 2. Async DMP.Store -> Server Fast Path (WAL Crash Consistency, O(1) archive)
        wal_archive[entry.lsn] = entry
        for tk in all_keys:
            l_tlb[tk].append(entry.lsn)
            l_tlb[tk] = l_tlb[tk][-4:]

        # 3. Amygdala Admission Gate into Smart Path Replay Buffer
        if (not enable_amygdala) or sal >= 0.35:
            replay_buffer.append(entry)
            # Fine-grained DTF isolation: ONLY mark fine_keys dirty (NEVER coarse SimHash buckets!)
            if ev.state_value is not None or sal >= 0.55:
                for fk in fine_keys:
                    dtf_dirty[fk] = entry.lsn
        else:
            stats.slm_skipped_by_amygdala += 1
            if not replay_buffer:
                consolidated_lsn = max(consolidated_lsn, entry.lsn)

        # Compact L1 working tail: low-salience verbose tool dumps paged to WAL handle (O(1) via L-TLB)
        if enable_prefix_zones and ev.is_tool_dump and sal < 0.35:
            handle = f"[WAL-Handle lsn={entry.lsn} tool_stdout_offloaded len={ev.token_count}]"
            l1_window.append(handle.split())
            l1_texts.append(handle)
        else:
            l1_window.append(ev.tokens)
            l1_texts.append(ev.text)

        if len(replay_buffer) >= consolidation_lag:
            server_sleep_consolidate()
            sync_watermark(consolidated_lsn)

    for ev in events:
        vec = embedder.encode(ev.text)
        fine_keys, coarse_keys = encoder.encode_keys(
            ev.text, vec, register=(ev.kind in ("fact_init", "fact_update"))
        )
        all_keys = fine_keys + coarse_keys

        if ev.kind == "query":
            stats.query_count += 1
            t0 = time.perf_counter()
            sync_watermark(consolidated_lsn)

            retrieved_texts: List[str] = []
            retrieved_tokens: List[str] = []
            tier_used: Optional[str] = None
            verified_override: bool = False

            # Fine-grained DTF check (zero False Sharing!)
            dirty_lsns = [
                dtf_dirty[fk]
                for fk in fine_keys
                if fk in dtf_dirty and dtf_dirty[fk] > client_known_watermark
            ]
            is_dirty = bool(dirty_lsns) and enable_dtf_forwarding

            # [Tier 1] L2-Cache (Local Raw, ~0ms) with Tag Verification
            l2_hits: List[RawLogEntry] = []
            for lk in (fine_keys if fine_keys else coarse_keys):
                if lk in l2_cache:
                    for e in l2_cache[lk]:
                        if e.state_value is not None and tag_matches(fine_keys, vec, e.topic_keys, e.vec):
                            l2_hits.append(e)
            if l2_hits:
                l2_hits.sort(key=lambda x: x.lsn, reverse=True)
                best_l2 = l2_hits[0]
                if (not is_dirty) or best_l2.lsn >= max(dirty_lsns):
                    tier_used = "L2"
                    verified_override = True
                    retrieved_texts.append(best_l2.text)
                    retrieved_tokens.extend(best_l2.tokens)

            # [Dirty Branch: Store-to-Load Forwarding]
            if tier_used is None and is_dirty:
                fwd_lsns: Set[int] = set()
                for fk in fine_keys:
                    for lsn in l_tlb.get(fk, []):
                        if lsn > client_known_watermark:
                            fwd_lsns.add(lsn)
                fwd_entries = [
                    wal_archive[lsn]
                    for lsn in sorted(fwd_lsns)
                    if lsn in wal_archive
                    and wal_archive[lsn].state_value is not None
                    and tag_matches(fine_keys, vec, wal_archive[lsn].topic_keys, wal_archive[lsn].vec)
                ]
                if fwd_entries:
                    tier_used = "DTF-FWD"
                    verified_override = True
                    # Only forward the freshest matching log record (minimal bandwidth!)
                    newest = fwd_entries[-1]
                    retrieved_texts.append(newest.text)
                    retrieved_tokens.extend(newest.tokens)
                    stats.dmp_network_bytes += len(newest.text.encode("utf-8"))

            # [Tier 2] L3-Cache (Local Consolidated KG, ~0ms) with Tag Verification
            if tier_used is None:
                l3_cands: List[KGNode] = []
                seen_nids: Set[int] = set()
                for tk in all_keys:
                    for nid in l3_key_index.get(tk, set()):
                        if nid in l3_cache and nid not in seen_nids:
                            seen_nids.add(nid)
                            nd = l3_cache[nid]
                            if tag_matches(fine_keys, vec, nd.topic_keys, nd.vec):
                                l3_cands.append(nd)
                if l3_cands:
                    l3_cands.sort(key=lambda nd: float(np.dot(vec, nd.vec)), reverse=True)
                    top_nd = l3_cands[0]
                    tier_used = "L3"
                    l3_cache.move_to_end(top_nd.node_id)
                    if (not enable_prefix_zones) or (top_nd.node_id not in pinned_node_ids):
                        retrieved_texts.append(top_nd.summary_text)
                        retrieved_tokens.extend(top_nd.tokens)

            # [Tier 3] Split-TLB: K-Cache (O(1) KG NodeID direct read -> Small Payload) with Tag Verification
            if tier_used is None:
                k_cands: List[KGNode] = []
                seen_k: Set[int] = set()
                for tk in all_keys:
                    for nid in k_tlb.get(tk, set()):
                        if nid not in seen_k and nid in temporal_kg and temporal_kg[nid].valid_to is None:
                            seen_k.add(nid)
                            nd = temporal_kg[nid]
                            if tag_matches(fine_keys, vec, nd.topic_keys, nd.vec):
                                k_cands.append(nd)
                if k_cands:
                    k_cands.sort(key=lambda nd: float(np.dot(vec, nd.vec)), reverse=True)
                    top_nd = k_cands[0]
                    tier_used = "K-TLB"
                    stats.dmp_network_bytes += len(top_nd.summary_text.encode("utf-8"))
                    retrieved_texts.append(top_nd.summary_text)
                    retrieved_tokens.extend(top_nd.tokens)
                    l3_cache[top_nd.node_id] = top_nd
                    for tk in top_nd.topic_keys:
                        l3_key_index[tk].add(top_nd.node_id)

            # [Tier 4] Split-TLB: L-Cache (O(1) Log Offset direct read -> Offload Compute to I/O)
            if tier_used is None and enable_l_cache:
                cand_lsns: Set[int] = set()
                for tk in all_keys:
                    cand_lsns.update(l_tlb.get(tk, []))
                cand_logs = [
                    wal_archive[lsn]
                    for lsn in cand_lsns
                    if lsn in wal_archive
                    and wal_archive[lsn].state_value is not None
                    and tag_matches(fine_keys, vec, wal_archive[lsn].topic_keys, wal_archive[lsn].vec)
                ]
                if cand_logs:
                    cand_logs.sort(key=lambda e: e.lsn, reverse=True)
                    top_log = cand_logs[0]
                    tier_used = "L-TLB"
                    verified_override = True
                    stats.dmp_network_bytes += len(top_log.text.encode("utf-8"))
                    retrieved_texts.append(top_log.text)
                    retrieved_tokens.extend(top_log.tokens)

            # [Tier 5] Red Path: High-Compute Server ANN Search (Last Resort)
            if tier_used is None:
                tier_used = "ANN"
                active_nodes = [nd for nd in temporal_kg.values() if nd.valid_to is None]
                best_nd = None
                if active_nodes:
                    scored_kg = [(float(np.dot(vec, nd.vec)), nd) for nd in active_nodes]
                    scored_kg.sort(key=lambda x: x[0], reverse=True)
                    best_nd = scored_kg[0][1]
                    for _, nd in scored_kg[:2]:
                        retrieved_texts.append(nd.summary_text)
                        retrieved_tokens.extend(nd.tokens)
                        stats.dmp_network_bytes += len(nd.summary_text.encode("utf-8"))
                        for tk in all_keys:
                            k_tlb[tk].add(nd.node_id)
                        l3_cache[nd.node_id] = nd
                        for tk in all_keys:
                            l3_key_index[tk].add(nd.node_id)
                # If DTF enabled, also check WAL for any fresher un-consolidated entry matching ANN's topic!
                if enable_dtf_forwarding and best_nd is not None:
                    fresher = [
                        e for e in wal_archive.values()
                        if e.lsn > consolidated_lsn and e.topic_id == best_nd.topic_id and e.state_value is not None
                    ]
                    if fresher:
                        fresher.sort(key=lambda e: e.lsn, reverse=True)
                        verified_override = True
                        retrieved_texts = [fresher[0].text]
                        retrieved_tokens = list(fresher[0].tokens)
                        stats.dmp_network_bytes += len(fresher[0].text.encode("utf-8"))
                elif not retrieved_texts and wal_archive:
                    scored_w = [(float(np.dot(vec, e.vec)), e) for e in wal_archive.values() if e.state_value is not None]
                    scored_w.sort(key=lambda x: x[0], reverse=True)
                    if scored_w:
                        best_e = scored_w[0][1]
                        retrieved_texts.append(best_e.text)
                        retrieved_tokens.extend(best_e.tokens)
                        stats.dmp_network_bytes += len(best_e.text.encode("utf-8"))

            stats.client_lookup_us_total += (time.perf_counter() - t0) * 1e6
            stats.hit_funnel[tier_used] += 1

            # --- Context Assembly ---
            if enable_prefix_zones:
                # Refresh Zone B ONLY at Epoch boundaries -> rock-solid Prefix Stability!
                if (not pinned_zone_b_tokens and l3_cache) or turns_since_epoch_refresh >= epoch_len:
                    refresh_zone_b()
                turns_since_epoch_refresh += 1

                # [Zone A: Immutable System] + [Zone B: Epoch-Pinned L3] + [Zone C: Time + Delta + L1 + Query]
                prompt_tokens = list(SYSTEM_PROMPT_TOKENS) + list(pinned_zone_b_tokens)
                prompt_tokens.append(f"[Now:T{ev.turn_id}]")
                prompt_tokens.extend(retrieved_tokens)
                for wt in l1_window[-2:]:
                    prompt_tokens.extend(wt)
                prompt_tokens.extend(ev.tokens)

                # Recency-Wins Overlay: fresher L2/DTF-FWD/L-TLB supersedes older Zone B state for this topic
                if verified_override and retrieved_texts:
                    eval_ctx = " ".join(retrieved_texts)
                else:
                    eval_ctx = pinned_zone_b_text + " " + " ".join(retrieved_texts)
            else:
                prompt_tokens = list(SYSTEM_PROMPT_TOKENS)
                prompt_tokens.append(f"[Now:T{ev.turn_id}]")
                prompt_tokens.extend(retrieved_tokens)
                for wt in l1_window[-2:]:
                    prompt_tokens.extend(wt)
                prompt_tokens.extend(ev.tokens)
                eval_ctx = " ".join(retrieved_texts) + " " + " ".join(l1_texts[-2:])

            cached, uncached = cache.query_and_insert(prompt_tokens)
            stats.total_raw_tokens += len(prompt_tokens)
            stats.total_cached_tokens += cached
            stats.total_uncached_tokens += uncached

            has_latest, is_stale = check_context_ground_truth(eval_ctx, ev.state_value or "", specs[ev.topic_id])
            stats.gt_recall_hits += int(has_latest)
            stats.stale_read_violations += int(is_stale)
            stats.missing_fact_misses += int((not has_latest) and (not is_stale))

            l1_window.append(ev.tokens)
            l1_texts.append(ev.text)
        else:
            write_back(ev, vec, fine_keys, coarse_keys)
            recent_vecs.append(vec)

    return stats


# ---------------------------------------------------------------------------
# 6. Runner & Clean Reporting
# ---------------------------------------------------------------------------

def print_table(title: str, rows: List[Dict[str, object]]):
    print(f"\n{'=' * 112}")
    print(f" {title}")
    print(f"{'=' * 112}")
    header = (
        f"{'System':<28} | {'RawTok':>6} | {'PrefHit':>7} | {'EffInf':>6} | {'SLM/q':>5} | "
        f"{'Cost.15':>7} | {'Cost1.0':>7} | {'NetKB':>5} | {'Latest%':>7} | {'Stale%':>6} | {'ANN%':>5}"
    )
    print(header)
    print("-" * 112)
    for r in rows:
        print(
            f"{str(r['name']):<28} | {r['avg_raw_tok']:>6.0f} | {r['prefix_hit_%']:>6.1f}% | "
            f"{r['eff_infer_tok/q']:>6.0f} | {r['slm_tok/q']:>5.0f} | {r['amort_cost_0.15x']:>7.0f} | "
            f"{r['amort_cost_1.0x']:>7.0f} | {r['net_kb/q']:>5.2f} | {r['latest_recall_%']:>6.1f}% | "
            f"{r['stale_read_%']:>5.1f}% | {r['ann_rate_%']:>4.1f}%"
        )


def main():
    all_results: Dict[str, object] = {}

    for regime in ("assistant", "coding_agent"):
        events, specs, _ = build_trace(regime=regime, n_turns=240, seed=2026)
        embedder = LocalEmbedder([ev.text for ev in events], dim=48, seed=2026)

        runs = [
            run_full_context(events, specs, regime, compact_limit=None),
            run_full_context(events, specs, regime, compact_limit=6000),
            run_naive_rag(events, specs, embedder, regime, top_k=5),
            run_mem0_zep_style(events, specs, embedder, regime, async_lag=10, sync_mode=True),
            run_mem0_zep_style(events, specs, embedder, regime, async_lag=10, sync_mode=False),
            # BIOSYS-v1 (Original Jan 2026 idea: pure LSH, no Prefix Zones, no DTF, no Amygdala)
            run_biosys(
                events, specs, embedder, regime,
                name="BIOSYS-v1 (Orig: LSH,noDTF)",
                encoder_mode="pure_lsh",
                enable_prefix_zones=False,
                enable_dtf_forwarding=False,
                enable_amygdala=False,
                consolidation_lag=10,
            ),
            # BIOSYS-v2 Full
            run_biosys(
                events, specs, embedder, regime,
                name="BIOSYS-v2 (Full Route A)",
                encoder_mode="hybrid_v2",
                enable_prefix_zones=True,
                enable_dtf_forwarding=True,
                enable_amygdala=True,
                consolidation_lag=10,
            ),
            # Key Ablations on BIOSYS-v2
            run_biosys(
                events, specs, embedder, regime,
                name="  - w/o Prefix Zones",
                encoder_mode="hybrid_v2",
                enable_prefix_zones=False,
                enable_dtf_forwarding=True,
                enable_amygdala=True,
                consolidation_lag=10,
            ),
            run_biosys(
                events, specs, embedder, regime,
                name="  - w/o DTF Forwarding",
                encoder_mode="hybrid_v2",
                enable_prefix_zones=True,
                enable_dtf_forwarding=False,
                enable_amygdala=True,
                consolidation_lag=10,
            ),
            run_biosys(
                events, specs, embedder, regime,
                name="  - w/o Amygdala Gate",
                encoder_mode="hybrid_v2",
                enable_prefix_zones=True,
                enable_dtf_forwarding=True,
                enable_amygdala=False,
                consolidation_lag=10,
            ),
            run_biosys(
                events, specs, embedder, regime,
                name="  - Keyword-Only Encoder",
                encoder_mode="keyword_only",
                enable_prefix_zones=True,
                enable_dtf_forwarding=True,
                enable_amygdala=True,
                consolidation_lag=10,
            ),
        ]

        summaries = [r.summary(cache_discount=0.10, slm_cost_ratio=0.15) for r in runs]
        print_table(f"REGIME: {regime.upper()} (240 turns, Lag=10, Prefix Block=16, Cache Discount=0.10x)", summaries)

        # Print detailed 5-tier funnel for BIOSYS-v2 vs Encoders
        print(f"\n[5-Tier Hit Funnel - {regime}]")
        for r in runs[5:]:
            s = r.summary()
            print(f"  {r.name:<28} -> {s['funnel']}")

        all_results[regime] = summaries

    # Lag sensitivity sweep for H3 (Stale-Read Consistency under Consolidation Lag = 5, 10, 20)
    lag_rows = []
    events_a, specs_a, _ = build_trace(regime="assistant", n_turns=240, seed=2026)
    embedder_a = LocalEmbedder([ev.text for ev in events_a], dim=48, seed=2026)
    for lag in (5, 10, 20):
        for dtf in (False, True):
            st = run_biosys(
                events_a, specs_a, embedder_a, "assistant",
                name=f"Lag={lag:>2} | DTF={'ON ' if dtf else 'OFF'}",
                encoder_mode="hybrid_v2",
                enable_prefix_zones=True,
                enable_dtf_forwarding=dtf,
                enable_amygdala=True,
                consolidation_lag=lag,
            )
            lag_rows.append(st.summary())
    print_table("CONSISTENCY SWEEP: Stale-Read Rate vs. Consolidation Lag (Assistant Trace)", lag_rows)
    all_results["lag_sweep"] = lag_rows

    out_path = Path(__file__).parent / "kill_test_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved structured results to: {out_path}")


if __name__ == "__main__":
    main()
