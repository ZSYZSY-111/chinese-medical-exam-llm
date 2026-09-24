"""Harness pipeline: load messages data → select items (with a hard refusal of any evaluation-set stem) → staged generation → filters → export (data, provenance, manifest, funnel)."""
import csv
import hashlib
import json
import platform
import re
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

try:
    from ..cmexam_prompts import parse_user_content, stable_fraction
    from ..build_cmb_train_sft import near_duplicate_scores, normalize_stem
    from .providers import Budget, OpenAICompatibleProvider, ResponseCache, complete_many
    from .tasks import TASKS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cmexam_prompts import parse_user_content, stable_fraction
    from build_cmb_train_sft import near_duplicate_scores, normalize_stem
    from providers import Budget, OpenAICompatibleProvider, ResponseCache, complete_many
    from tasks import TASKS

HARNESS_VERSION = "0.1"
TOKENS_PER_CHAR = 0.75  # rough estimate for mostly-Chinese text, used only by the dry-run budget


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


EXPLANATION_RECORD = re.compile(r"解析[:：]([\s\S]+)\n答案[:：]\s*([A-Z]{1,6})")


def load_items(path, metadata_path=None):
    """Load messages-format data; sample_id matches build_cmb_train_sft (hash of the user content). Returns (items, counts)."""
    meta = {}
    if metadata_path:
        with open(metadata_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    meta[record["sample_id"]] = record
    items, counts = [], Counter()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages") or []
            users = [m["content"] for m in messages if m.get("role") == "user"]
            if not users or not messages or messages[-1].get("role") != "assistant":
                counts["skipped_bad_messages"] += 1
                continue
            answer = messages[-1]["content"].strip()
            explanation = None
            explained = EXPLANATION_RECORD.fullmatch(answer)
            if explained:
                # rationale-style SFT rows (解析：…\n答案：X): keep the official rationale for the segment-scoring task
                explanation, answer = explained.group(1).strip(), explained.group(2)
                counts["loaded_with_explanation"] += 1
            if not re.fullmatch(r"[A-Z]{1,6}", answer):
                counts["skipped_not_direct_answer"] += 1
                continue
            try:
                parsed = parse_user_content(users[0])
            except ValueError:
                counts["skipped_unparsable_prompt"] += 1
                continue
            sample_id = sha256_text("\n".join(users))
            extra = meta.get(sample_id, {})
            items.append({"sample_id": sample_id, "question": parsed["question"], "options": parsed["options"],
                          "gold": "".join(sorted(set(answer))), "stem_hash": sha256_text(normalize_stem(parsed["question"])),
                          "source": extra.get("source"), "variant": extra.get("variant"),
                          "exam_type": extra.get("exam_type"), "question_type": extra.get("question_type"),
                          "explanation": explanation, "messages": messages if explanation else None})
            counts["loaded"] += 1
    return items, counts


def load_reference_stems(paths):
    """Evaluation-set stems: returns (set of exact hashes, list of normalised stems). Reads `question` from JSON/JSONL and `Question` from CSV."""
    hashes, keys = set(), []
    for path in paths:
        path = str(path)
        if path.endswith(".json"):
            rows = json.load(open(path, encoding="utf-8"))
            questions = [r.get("question", "") for r in rows]
        elif path.endswith(".csv"):
            with open(path, encoding="utf-8", newline="") as handle:
                questions = [r.get("Question") or r.get("question") or "" for r in csv.DictReader(handle)]
        elif path.endswith(".jsonl"):
            questions = []
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        if "question" in record:
                            questions.append(record["question"])
                        elif "messages" in record:
                            users = [m["content"] for m in record["messages"] if m.get("role") == "user"]
                            try:
                                questions.append(parse_user_content(users[0])["question"])
                            except (ValueError, IndexError):
                                pass
        else:
            raise ValueError(f"unrecognised reference file format: {path}")
        for question in questions:
            key = normalize_stem(question)
            if key:
                hashes.add(sha256_text(key))
                keys.append(key)
    return hashes, keys


def load_scores(path):
    scores = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                scores[record["sample_id"]] = record
    return scores


def select_items(items, scores=None, p_max=None, limit=0, seed=42, exclude_hashes=frozenset(), only_original=True,
                 exam_types=None, question_types=None, order="hash"):
    """order="hash": the first `limit` items in stable-hash order; order="p_gold": filter the same way, then take the `limit` items with the lowest p_gold (hardest first)."""
    counts = Counter()
    ordered = sorted(items, key=lambda it: stable_fraction(seed, "harness", it["sample_id"]))
    seen, selected = set(), []
    cut = limit if order == "hash" else 0
    for item in ordered:
        counts["total"] += 1
        if exam_types and item.get("exam_type") not in exam_types:
            counts["filtered_exam_type"] += 1
            continue
        if question_types and item.get("question_type") not in question_types:
            counts["filtered_question_type"] += 1
            continue
        if only_original and item["stem_hash"] in seen:
            counts["duplicate_stem_skipped"] += 1
            continue
        if scores is not None:
            score = scores.get(item["sample_id"])
            if score is None:
                counts["no_score_skipped"] += 1
                continue
            item["p_gold"] = score.get("p_gold")
            if p_max is not None and item["p_gold"] is not None and item["p_gold"] > p_max:
                counts["already_learned_skipped"] += 1
                continue
        if item["stem_hash"] in exclude_hashes:
            counts["refused_reference_overlap"] += 1
            continue
        seen.add(item["stem_hash"])
        selected.append(item)
        counts["selected"] += 1
        if cut and len(selected) >= cut:
            break
    if order == "p_gold":
        if scores is None:
            raise ValueError("order=p_gold requires --scores")
        selected.sort(key=lambda it: (it.get("p_gold") if it.get("p_gold") is not None else 1.0, stable_fraction(seed, "harness", it["sample_id"])))
        if limit:
            counts["candidates_before_cut"] = len(selected)
            selected = selected[:limit]
            counts["selected"] = len(selected)
    return selected, counts


def build_providers(cfg):
    providers = {}
    for name, spec in cfg["providers"].items():
        kind = spec.get("type", "openai_compatible")
        if kind != "openai_compatible":
            raise ValueError(f"provider {name}: unsupported type {kind}")
        providers[name] = OpenAICompatibleProvider(base_url=spec["base_url"], model=spec["model"],
                                                   api_key_env=spec.get("api_key_env", "DEEPSEEK_API_KEY"),
                                                   timeout=spec.get("timeout", 60), max_retries=spec.get("max_retries", 5),
                                                   extra_body=spec.get("extra_body"),
                                                   first_token_deadline=spec.get("first_token_deadline", 90),
                                                   wall_deadline=spec.get("wall_deadline", 300))
    return providers


def build_tasks(cfg, names):
    tasks = []
    for name in names:
        if name not in TASKS:
            raise ValueError(f"unknown task {name}; available: {sorted(TASKS)}")
        tasks.append(TASKS[name](cfg.get("tasks", {}).get(name, {})))
    return tasks


def params_for(cfg, provider_name, overrides=None):
    spec = cfg["providers"][provider_name]
    params = {"temperature": spec.get("temperature", 0.7), "max_tokens": spec.get("max_tokens", 300)}
    if spec.get("seed") is not None:
        params["seed"] = spec["seed"]
    params.update(overrides or {})
    return params


def estimate_cost(cfg, tasks, items):
    """Dry run: estimate tokens and cost from character counts without sending requests."""
    rows, total = [], 0.0
    for task in tasks:
        tcfg = cfg["tasks"][task.name]
        prices = cfg["providers"][tcfg["provider"]].get("prices_cny_per_million", {})
        in_chars = sum(sum(len(m["content"]) for m in task.stage1_messages(item)) for item in items)
        in_tokens = in_chars * TOKENS_PER_CHAR + 8 * len(items)
        out_tokens = task.expected_output_chars * TOKENS_PER_CHAR * len(items)
        cost = (in_tokens * prices.get("input", 0) + out_tokens * prices.get("output", 0)) / 1e6
        if task.stages == 2:
            vprices = cfg["providers"][tcfg.get("verifier", tcfg["provider"])].get("prices_cny_per_million", {})
            in2 = (in_chars * 0.6) * TOKENS_PER_CHAR
            out2 = task.stage2_output_chars * TOKENS_PER_CHAR * len(items)
            cost += (in2 * vprices.get("input", 0) + out2 * vprices.get("output", 0)) / 1e6
        rows.append({"task": task.name, "items": len(items), "est_input_tokens": int(in_tokens), "est_output_tokens": int(out_tokens),
                     "est_cost_cny": round(cost, 3)})
        total += cost
    return rows, round(total, 3)


def run_task(task, items, cfg, providers, budget, cache, workers, decontam_refs, exclude_hashes):
    tcfg = cfg["tasks"][task.name]
    provider_name = tcfg["provider"]
    provider = providers[provider_name]
    params = params_for(cfg, provider_name, tcfg.get("params"))
    for item in items:
        if item["stem_hash"] in exclude_hashes:
            raise RuntimeError(f"refusing to send sample {item['sample_id'][:12]}: its stem matches an evaluation question")
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    records = OrderedDict()
    for item in items:
        records[item["sample_id"]] = {
            "gen_id": sha256_text(f"{task.version}|{item['sample_id']}")[:16], "sample_id": item["sample_id"], "task": task.name,
            "task_version": task.version, "provider": provider_name, "model": provider.model, "model_seen": None, "fingerprint": None,
            "request_hash": None, "cached": None, "params": params, "usage": None, "cost_cny": 0.0, "parsed": None, "stage2": None,
            "filters": {}, "kept": False, "drop_reason": None, "created_at": now, "p_gold": item.get("p_gold"),
            "exam_type": item.get("exam_type"), "question_type": item.get("question_type"), "source": item.get("source")}
    requests = [{"key": item["sample_id"], "messages": task.stage1_messages(item), "params": params} for item in items]
    results = complete_many(provider, requests, budget, cache, workers)
    for item in items:
        record = records[item["sample_id"]]
        result = results.get(item["sample_id"])
        if result is None:
            record["drop_reason"] = "budget_stopped"
            continue
        if result.get("queued"):
            record["drop_reason"] = "queued"
            record["request_hash"] = result.get("request_hash")
            continue
        record.update(model_seen=result.get("model"), fingerprint=result.get("fingerprint"), request_hash=result["request_hash"],
                      cached=result["cached"], usage=result.get("usage"), cost_cny=result.get("cost_cny", 0.0))
        parsed = task.parse_stage1(result.get("text"))
        if parsed is None:
            record["drop_reason"] = "parse_failed"
            continue
        record["parsed"] = parsed

    stage2_items = [item for item in items if records[item["sample_id"]]["parsed"] is not None
                    and task.stage2_messages(item, records[item["sample_id"]]["parsed"]) is not None]
    if stage2_items:
        verifier_name = tcfg.get("verifier", provider_name)
        verifier = providers[verifier_name]
        vparams = params_for(cfg, verifier_name, tcfg.get("verifier_params"))
        requests = [{"key": item["sample_id"], "messages": task.stage2_messages(item, records[item["sample_id"]]["parsed"]),
                     "params": vparams} for item in stage2_items]
        results2 = complete_many(verifier, requests, budget, cache, workers)
        for item in stage2_items:
            record = records[item["sample_id"]]
            result = results2.get(item["sample_id"])
            if result is None or result.get("queued"):
                record["drop_reason"] = "budget_stopped_stage2" if result is None else "queued_stage2"
                record["parsed"] = None
                continue
            record["stage2"] = {"provider": verifier_name, "model": verifier.model, "request_hash": result["request_hash"],
                                "cached": result["cached"], "usage": result.get("usage"), **(task.parse_stage2(result.get("text")) or {})}
            record["cost_cny"] += result.get("cost_cny", 0.0)

    context = {"decontam": {}, "decontam_method": None}
    to_check = []
    for item in items:
        parsed = records[item["sample_id"]]["parsed"]
        if parsed is None:
            continue
        text = task.needs_decontam(parsed)
        if text:
            to_check.append((item["sample_id"], normalize_stem(text)))
    if to_check and decontam_refs:
        dcfg = cfg.get("decontam", {})
        scores, method = near_duplicate_scores([key for _, key in to_check], decontam_refs, dcfg.get("threshold", 0.7),
                                               dcfg.get("min_chars", 20), dcfg.get("num_perm", 128))
        context["decontam_method"] = method
        for index, (score, ref) in scores.items():
            context["decontam"][to_check[index][0]] = {"jaccard": round(score, 3), "ref": ref[:60]}

    filter_names = []
    for item in items:
        record = records[item["sample_id"]]
        if record["parsed"] is None or record["drop_reason"]:
            continue
        outcome = task.filters(item, record["parsed"], record["stage2"], context)
        if not filter_names:
            filter_names = list(outcome)
        record["filters"] = {name: {"pass": bool(ok), "detail": detail} for name, (ok, detail) in outcome.items()}
        failed = [name for name, (ok, _) in outcome.items() if not ok]
        record["kept"] = not failed
        record["drop_reason"] = failed[0] if failed else None

    funnel = OrderedDict()
    funnel["selected"] = len(items)
    funnel["requested"] = sum(1 for r in records.values() if r["request_hash"])
    funnel["queued"] = sum(1 for r in records.values() if str(r["drop_reason"] or "").startswith("queued"))
    funnel["cache_hits"] = sum(1 for r in records.values() if r["cached"])
    funnel["budget_stopped"] = sum(1 for r in records.values() if str(r["drop_reason"] or "").startswith("budget_stopped"))
    funnel["parsed_ok"] = sum(1 for r in records.values() if r["parsed"] is not None)
    for name in filter_names:
        funnel[f"pass_{name}"] = sum(1 for r in records.values() if r["filters"].get(name, {}).get("pass"))
    funnel["kept"] = sum(1 for r in records.values() if r["kept"])
    funnel["drop_reasons"] = dict(Counter(r["drop_reason"] for r in records.values() if r["drop_reason"]))
    funnel["decontam_method"] = context["decontam_method"]
    funnel["cost_cny"] = round(sum(r["cost_cny"] for r in records.values()), 4)
    exports = [task.export(item, records[item["sample_id"]]["parsed"]) for item in items if records[item["sample_id"]]["kept"]]
    return list(records.values()), exports, funnel


def run(cfg, items, tasks, providers, budget, cache=None, workers=4, decontam_refs=(), exclude_hashes=frozenset()):
    all_records, exports, funnels = [], {}, OrderedDict()
    for task in tasks:
        records, task_exports, funnel = run_task(task, items, cfg, providers, budget, cache, workers, decontam_refs, exclude_hashes)
        all_records.extend(records)
        exports[task.name] = task_exports
        funnels[task.name] = funnel
    return all_records, exports, funnels


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def funnel_markdown(funnels, budget):
    lines = []
    for name, funnel in funnels.items():
        lines.append(f"### {name}\n")
        lines.append("| step | count |")
        lines.append("|---|---:|")
        for key, value in funnel.items():
            if key in ("drop_reasons", "decontam_method", "cost_cny"):
                continue
            lines.append(f"| {key} | {value} |")
        lines.append(f"| cost (CNY) | {funnel['cost_cny']} |")
        if funnel["drop_reasons"]:
            lines.append("\ndrop reasons: " + ", ".join(f"{k} {v}" for k, v in sorted(funnel["drop_reasons"].items(), key=lambda kv: -kv[1])))
        if funnel.get("decontam_method"):
            lines.append(f"\ndecontamination: {funnel['decontam_method']}")
        lines.append("")
    summary = budget.summary()
    lines.append(f"{summary['calls']} calls, cost {summary['cost_cny']} CNY (cap {summary['max_cost_cny']}), "
                 f"input tokens {summary['tokens']['input']} (cache hits {summary['tokens']['input_cache_hit']}), output tokens {summary['tokens']['output']}")
    return "\n".join(lines)


def write_outputs(out_dir, cfg, cfg_text, input_paths, tasks, records, exports, funnels, budget, select_counts, load_counts,
                  cache_path, started_at):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "provenance.jsonl", records)
    for name, rows in exports.items():
        write_jsonl(out / f"data_{name}.jsonl", rows)
    manifest = {
        "harness_version": HARNESS_VERSION, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_s": round(time.time() - started_at, 1), "python": platform.python_version(),
        "config_sha256": sha256_text(cfg_text), "config": cfg,
        "inputs": {str(p): sha256_file(p) for p in input_paths if p and Path(p).exists()},
        "tasks": {t.name: {"version": t.version, "provider": cfg["tasks"][t.name]["provider"],
                           "model": cfg["providers"][cfg["tasks"][t.name]["provider"]]["model"]} for t in tasks},
        "models_seen": sorted({r["model_seen"] for r in records if r.get("model_seen")}),
        "load_counts": dict(load_counts), "selection_counts": dict(select_counts), "funnels": funnels,
        "budget": budget.summary(), "cache_file": str(cache_path) if cache_path else None,
        "exports": {name: len(rows) for name, rows in exports.items()},
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with open(out / "funnel.md", "w", encoding="utf-8") as handle:
        handle.write(funnel_markdown(funnels, budget) + "\n")
    return manifest
