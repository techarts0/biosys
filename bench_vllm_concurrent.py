#!/usr/bin/env python3
"""
BIOSYS v2 — Multi-Session Concurrent Shared-vLLM Stress Benchmark (PAI-DSW Ready)
=================================================================================
Evaluates multi-tenant shared GPU serving over a single unmodified vLLM instance
(`--enable-prefix-caching`) hosting Qwen2.5-7B-Instruct on NVIDIA A10 (24GB).

Measures across concurrency levels C in {1, 4, 8, 16, 32} independent agent sessions
(with identical turns per session across all C to eliminate cold-start ratio confounds):
  - Real Concurrent Serving Throughput (req/s)
  - SLO Attainment (%) & Effective Goodput (req/s meeting TTFT <= 200ms, 300ms, 500ms)
  - Wall-clock P50 / P90 / P99 TTFT (ms) under multi-session GPU prefill queueing
  - Real vLLM Prometheus `/metrics` KV-Cache Prefix Hit Rate (%) under cache contention

Upload alongside `bench_vllm_serving.py` and `locomo10.json` in `/mnt/workspace/biosys/`:
  1) 20-second smoke test:
     python bench_vllm_concurrent.py --smoke --base-url http://127.0.0.1:8000 --model qwen2.5-7b
  2) Full concurrent sweep (~8-12 minutes on A10, auto-checkpoints with --resume):
     python bench_vllm_concurrent.py --base-url http://127.0.0.1:8000 --model qwen2.5-7b --out vllm_concurrent_results.json --resume
  3) Local offline dry-run verification:
     python bench_vllm_concurrent.py --mock --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Reuse exact prompt builders & zero-dependency vLLM HTTP streaming client from bench_vllm_serving.py
import bench_vllm_serving as bv


def prepare_session_streams_locomo(
    system_key: str,
    convs: List[bv.ParsedConversation],
    encoders: Dict[str, bv.LocalHybridEncoder],
    concurrency: int,
    turns_per_session: int,
    run_nonce: str,
) -> List[List[bv.PreparedQuery]]:
    """
    Build `concurrency` independent multi-session streams from official LoCoMo conversations.
    Each concurrent session gets a unique session header nonce inside Zone A so sessions do NOT
    artificially share each other's personal memory prefixes (realistic multi-tenant KV contention!),
    while each session's own turns reuse its own Zone A + Zone B prefix across epochs.
    """
    streams: List[List[bv.PreparedQuery]] = []
    # Pre-build base streams per unique conversation once to save CPU time
    cache_by_conv: Dict[int, List[bv.PreparedQuery]] = {}
    for s_idx in range(concurrency):
        c_idx = s_idx % len(convs)
        if c_idx not in cache_by_conv:
            # Skip first 30 cold-start queries so Zone B is realistically populated during stress test
            raw_qs = bv.build_locomo_prompts(
                system_key, [convs[c_idx]], encoders, run_tag=f"BASE-{c_idx}", max_queries=30 + turns_per_session
            )
            cache_by_conv[c_idx] = raw_qs[30 : 30 + turns_per_session] if len(raw_qs) > 30 else raw_qs[:turns_per_session]

        # Stamp unique per-policy and per-tenant session ID into Zone A so policies and sessions never share KV blocks
        sess_tag = f"[TenantSession={run_nonce}-{system_key}-C{concurrency}-S{s_idx}]\n"
        tenant_qs = [
            bv.PreparedQuery(
                system_prompt=sess_tag + q.system_prompt,
                user_prompt=q.user_prompt,
                gold_answer=q.gold_answer,
                category=q.category,
                is_lag=q.is_lag,
                evidence_hit=q.evidence_hit,
            )
            for q in cache_by_conv[c_idx]
        ]
        streams.append(tenant_qs)
    return streams


def prepare_session_streams_coding(
    system_key: str,
    concurrency: int,
    turns_per_session: int,
    run_nonce: str,
) -> List[List[bv.PreparedQuery]]:
    """Build `concurrency` independent tool-heavy Coding Agent sessions competing for GPU KV cache."""
    streams: List[List[bv.PreparedQuery]] = []
    for s_idx in range(concurrency):
        qs = bv.build_coding_agent_prompts(
            system_key,
            run_tag=f"CA-{run_nonce}-{system_key}-C{concurrency}-S{s_idx}",
            n_queries=turns_per_session,
            seed=2026 + s_idx * 17,
        )
        streams.append(qs)
    return streams


def run_concurrent_point(
    client: bv.VLLMClient,
    label: str,
    concurrency: int,
    session_streams: List[List[bv.PreparedQuery]],
    max_tokens: int = 24,
) -> Dict[str, object]:
    """
    Launch `concurrency` worker threads; each thread executes its own session's turns sequentially
    (closed-loop multi-tenant agent execution) against the shared vLLM instance.
    """
    # Neutral untimed warmup (does not pre-warm any tenant's Zone A/B prefix)
    _ = client.generate_stream("Warmup check.", "Reply OK.", max_tokens=4)

    prom_before = client.read_prometheus_prefix_counters()
    t_wall_start = time.perf_counter()

    def run_one_session(s_idx: int, qs: List[bv.PreparedQuery]) -> List[bv.InferenceResult]:
        # Tiny deterministic stagger (0..120ms) to avoid artificial lock-step synchronization
        if not client.mock and concurrency > 1:
            time.sleep((s_idx % concurrency) * 0.015)
        res_list: List[bv.InferenceResult] = []
        for q in qs:
            res_list.append(client.generate_stream(q.system_prompt, q.user_prompt, max_tokens=max_tokens))
        return res_list

    all_res: List[bv.InferenceResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(run_one_session, i, st) for i, st in enumerate(session_streams)]
        for fut in as_completed(futs):
            all_res.extend(fut.result())

    wall_s = time.perf_counter() - t_wall_start
    prom_after = client.read_prometheus_prefix_counters()

    ttfts = [r.ttft_ms for r in all_res]
    e2es = [r.e2e_ms for r in all_res]
    ptoks = [r.prompt_tokens for r in all_res]
    ctoks = [r.cached_tokens for r in all_res if r.cached_tokens is not None]
    n_req = len(all_res)

    prom_hit = None
    if prom_before and prom_after:
        dq = prom_after["queries"] - prom_before["queries"]
        if dq > 0:
            prom_hit = 100.0 * (prom_after["hits"] - prom_before["hits"]) / dq
    usage_hit = (100.0 * sum(ctoks) / sum(ptoks)) if (ctoks and sum(ptoks) > 0) else None
    kv_hit = prom_hit if prom_hit is not None else (usage_hit if usage_hit is not None else 0.0)

    slo_200_cnt = sum(1 for x in ttfts if x <= 200.0)
    slo_300_cnt = sum(1 for x in ttfts if x <= 300.0)
    slo_500_cnt = sum(1 for x in ttfts if x <= 500.0)
    throughput = n_req / max(1e-6, wall_s)
    goodput_200 = slo_200_cnt / max(1e-6, wall_s)
    goodput_300 = slo_300_cnt / max(1e-6, wall_s)
    goodput_500 = slo_500_cnt / max(1e-6, wall_s)

    return {
        "system": label,
        "concurrency": concurrency,
        "n_requests": n_req,
        "avg_prompt_tok": round(float(np.mean(ptoks)), 1),
        "vllm_kv_hit_%": round(float(kv_hit), 1),
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 2),
        "ttft_p90_ms": round(float(np.percentile(ttfts, 90)), 2),
        "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 2),
        "ttft_mean_ms": round(float(np.mean(ttfts)), 2),
        "e2e_mean_ms": round(float(np.mean(e2es)), 2),
        "throughput_rps": round(throughput, 2),
        "slo200_att_%": round(100.0 * slo_200_cnt / max(1, n_req), 1),
        "goodput200_rps": round(goodput_200, 2),
        "slo300_att_%": round(100.0 * slo_300_cnt / max(1, n_req), 1),
        "goodput300_rps": round(goodput_300, 2),
        "slo500_att_%": round(100.0 * slo_500_cnt / max(1, n_req), 1),
        "goodput500_rps": round(goodput_500, 2),
        "wall_sec": round(wall_s, 1),
    }


def print_concurrent_table(title: str, rows: List[Dict[str, object]]) -> None:
    print(f"\n{'=' * 118}")
    print(f" {title}")
    print(f"{'=' * 118}")
    print(
        f"{'System':<20} | {'C':>2} | {'N':>3} | {'Tok':>5} | {'KVHit%':>6} | "
        f"{'P50ms':>6} | {'P99ms':>6} | {'Thr(r/s)':>8} | {'SLO200':>6} | {'GP200':>6} | {'SLO300':>6} | {'GP300':>6}"
    )
    print("-" * 118)
    for r in rows:
        print(
            f"{str(r['system']):<20} | {r['concurrency']:>2} | {r['n_requests']:>3} | {r['avg_prompt_tok']:>5.0f} | "
            f"{r['vllm_kv_hit_%']:>5.1f}% | {r['ttft_p50_ms']:>6.1f} | {r['ttft_p99_ms']:>6.1f} | "
            f"{r['throughput_rps']:>8.2f} | {r['slo200_att_%']:>5.1f}% | {r['goodput200_rps']:>6.2f} | "
            f"{r['slo300_att_%']:>5.1f}% | {r['goodput300_rps']:>6.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="BIOSYS v2 Concurrent Multi-Session Shared-vLLM Stress Benchmark")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--locomo", default=None)
    ap.add_argument("--concurrencies", default="1,4,8,16,32", help="Comma-separated concurrent session counts")
    ap.add_argument("--turns", type=int, default=20, help="Fixed turns per session across all C (controlled horizon)")
    ap.add_argument("--verify-r1", action="store_true",
                    help="Targeted ~25-min TSC R1 verification: Sync-KG-Sorted+Offload (C=16,32) + 3x isolated repeats of 4 headline C=32 points")
    ap.add_argument("--repeats", type=int, default=3, help="Number of independent isolated repeats in --verify-r1 mode")
    ap.add_argument("--out", default="vllm_concurrent_results.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Fast 20-second smoke test (C=1,4 with 6 turns/session)")
    ap.add_argument("--mock", action="store_true", help="Offline dry-run without GPU/vLLM")
    args = ap.parse_args()

    out_path = Path(args.out)
    results: Dict[str, Dict[str, Dict[str, object]]] = {"locomo_concurrent": {}, "coding_concurrent": {}, "r1_verify": {}}
    if args.resume and out_path.exists():
        try:
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            results["locomo_concurrent"] = loaded.get("locomo_concurrent", {})
            results["coding_concurrent"] = loaded.get("coding_concurrent", {})
            results["r1_verify"] = loaded.get("r1_verify", {})
            print(f"[RESUME] Loaded existing checkpoint from {out_path}")
        except Exception:
            pass

    client = bv.VLLMClient(args.base_url, args.model, mock=args.mock)
    client.check_health()

    c_list = [1, 4] if args.smoke else [int(x.strip()) for x in args.concurrencies.split(",") if x.strip()]
    locomo_path = bv.find_locomo_file(args.locomo)
    convs = bv.load_locomo(locomo_path, max_convs=10)
    encoders = {
        c.sample_id: bv.LocalHybridEncoder(
            [t.text for s in c.sessions for t in c.turns_by_session[s]]
            + [o.text for s in c.sessions for o in c.obs_by_session[s]],
            c.speakers, dim=64, seed=2026,
        )
        for c in convs
    }

    run_nonce = str(int(time.time()))[-6:]

    if args.verify_r1:
        # Targeted TSC verification: strictly isolated per-policy & per-repeat prefix tags
        n_rep = 2 if args.smoke else args.repeats
        turns = 6 if args.smoke else args.turns
        c_hi = 4 if args.smoke else 32
        c_mid = 2 if args.smoke else 16
        plan = [
            ("coding", "Sync-KG-Sorted+Offload", "Sync-KG-Sorted+Offload", c_mid, 20),
            ("coding", "Sync-KG-Sorted+Offload", "Sync-KG-Sorted+Offload", c_hi, 20),
            ("coding", "Mem0/Zep-Sync", "Sync-KG", c_hi, 20),
            ("coding", "BIOSYS-v2-Full", "BIOSYS-Full", c_hi, 20),
            ("locomo", "BIOSYS-v2-Adaptive", "BIOSYS-Adaptive", c_hi, 24),
            ("locomo", "BIOSYS-v2-Fixed", "BIOSYS-Fixed", c_hi, 24),
        ]
        print(f"\n>>> TSC R1 Verification Suite ({len(plan)} configs x {n_rep} isolated repeats, turns={turns}) <<<")
        summary_rows: List[Dict[str, object]] = []
        for wl, internal_key, display_name, C, max_tok in plan:
            rep_rows: List[Dict[str, object]] = []
            for r_idx in range(1, n_rep + 1):
                rk = f"{wl}_{display_name}_C{C}_R{r_idx}"
                if rk in results["r1_verify"]:
                    print(f"  [SKIP] {rk}")
                    rep_rows.append(results["r1_verify"][rk])
                    continue
                rep_nonce = f"{run_nonce}R{r_idx}"
                if wl == "coding":
                    streams = prepare_session_streams_coding(internal_key, C, turns, rep_nonce)
                else:
                    streams = prepare_session_streams_locomo(internal_key, convs, encoders, C, turns, rep_nonce)
                print(f"  Running {wl.upper()} [{display_name}] C={C} R{r_idx}/{n_rep} ({sum(len(s) for s in streams)} reqs)...", flush=True)
                row = run_concurrent_point(client, f"{display_name} (R{r_idx})", C, streams, max_tokens=max_tok)
                row["repeat"] = r_idx
                row["workload"] = wl
                results["r1_verify"][rk] = row
                rep_rows.append(row)
                out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

            thrs = [float(x["throughput_rps"]) for x in rep_rows]
            p50s = [float(x["ttft_p50_ms"]) for x in rep_rows]
            p99s = [float(x["ttft_p99_ms"]) for x in rep_rows]
            kvs = [float(x["vllm_kv_hit_%"]) for x in rep_rows]
            gp3 = [float(x["goodput300_rps"]) for x in rep_rows]
            summary_rows.append({
                "wl": wl, "system": display_name, "C": C, "reps": n_rep,
                "kv_mean": round(float(np.mean(kvs)), 1), "kv_min": round(min(kvs), 1), "kv_max": round(max(kvs), 1),
                "p50_mean": round(float(np.mean(p50s)), 1), "p50_min": round(min(p50s), 1), "p50_max": round(max(p50s), 1),
                "p99_mean": round(float(np.mean(p99s)), 1), "p99_min": round(min(p99s), 1), "p99_max": round(max(p99s), 1),
                "thr_mean": round(float(np.mean(thrs)), 2), "thr_min": round(min(thrs), 2), "thr_max": round(max(thrs), 2),
                "gp3_mean": round(float(np.mean(gp3)), 2), "gp3_min": round(min(gp3), 2), "gp3_max": round(max(gp3), 2),
            })

        print(f"\n{'=' * 124}")
        print(f" TSC R1 ISOLATED MULTI-REPEAT SUMMARY (model={client.model}, repeats={n_rep}, turns={turns})")
        print(f"{'=' * 124}")
        print(f"{'WL':<6} | {'System':<22} | {'C':>2} | {'KVHit% (min-max)':<18} | {'P50ms (min-max)':<22} | {'Thr r/s (min-max)':<20} | {'GP300 (min-max)':<18}")
        print("-" * 124)
        for s in summary_rows:
            print(
                f"{s['wl']:<6} | {s['system']:<22} | {s['C']:>2} | "
                f"{s['kv_mean']:>5.1f} ({s['kv_min']:.1f}-{s['kv_max']:.1f})    | "
                f"{s['p50_mean']:>7.1f} ({s['p50_min']:.1f}-{s['p50_max']:.1f}) | "
                f"{s['thr_mean']:>5.2f} ({s['thr_min']:.2f}-{s['thr_max']:.2f})   | "
                f"{s['gp3_mean']:>5.2f} ({s['gp3_min']:.2f}-{s['gp3_max']:.2f})"
            )
        results["r1_verify_summary"] = summary_rows  # type: ignore[assignment]
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved R1 verification results cleanly to: {out_path.resolve()}")
        return

    # --- Concurrent Workload 1: Multi-Session LoCoMo (KV Cache Contention + Tail Latency) ---
    lc_systems = [
        ("Mem0/Zep-Sync", "Sync-KG"),
        ("BIOSYS-v2-NoZones", "BIOSYS-NoZones"),
        ("BIOSYS-v2-Adaptive", "BIOSYS-Adaptive"),
        ("BIOSYS-v2-Fixed", "BIOSYS-Fixed"),
    ]
    print("\n>>> Concurrent Workload 1: Shared-GPU Multi-Session LoCoMo <<<")
    lc_rows: List[Dict[str, object]] = []
    for C in c_list:
        turns_per_sess = 6 if args.smoke else args.turns
        for internal_key, display_name in lc_systems:
            rk = f"{display_name}_C{C}"
            if rk in results["locomo_concurrent"]:
                print(f"  [SKIP] {rk}")
                lc_rows.append(results["locomo_concurrent"][rk])
                continue
            streams = prepare_session_streams_locomo(internal_key, convs, encoders, C, turns_per_sess, run_nonce)
            print(f"  Running LoCoMo [{display_name}] C={C} ({sum(len(s) for s in streams)} reqs)...", flush=True)
            row = run_concurrent_point(client, display_name, C, streams, max_tokens=24)
            results["locomo_concurrent"][rk] = row
            lc_rows.append(row)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # --- Concurrent Workload 2: Multi-Session Coding Agent (SLO Goodput & Scaling) ---
    ca_systems = [
        ("Mem0/Zep-Sync", "Sync-KG"),
        ("BIOSYS-v2-NoZones", "BIOSYS-NoZones"),
        ("BIOSYS-v2-Full", "BIOSYS-Full"),
    ]
    print("\n>>> Concurrent Workload 2: Shared-GPU Multi-Session Coding Agent <<<")
    ca_rows: List[Dict[str, object]] = []
    for C in c_list:
        turns_per_sess = 6 if args.smoke else args.turns
        for internal_key, display_name in ca_systems:
            rk = f"{display_name}_C{C}"
            if rk in results["coding_concurrent"]:
                print(f"  [SKIP] {rk}")
                ca_rows.append(results["coding_concurrent"][rk])
                continue
            streams = prepare_session_streams_coding(internal_key, C, turns_per_sess, run_nonce)
            print(f"  Running Coding [{display_name}] C={C} ({sum(len(s) for s in streams)} reqs)...", flush=True)
            row = run_concurrent_point(client, display_name, C, streams, max_tokens=20)
            results["coding_concurrent"][rk] = row
            ca_rows.append(row)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print_concurrent_table(f"CONCURRENT SHARED-vLLM — LoCoMo (model={client.model})", lc_rows)
    print_concurrent_table(f"CONCURRENT SHARED-vLLM — CODING AGENT (model={client.model})", ca_rows)
    print(f"\nSaved concurrent results cleanly to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
