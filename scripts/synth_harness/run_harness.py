#!/usr/bin/env python3
"""Command-line entry point of the annotation / synthesis harness.

Example (20-question smoke test: dry-run for the budget first, then the real call):
  python scripts/synth_harness/run_harness.py --config configs/synth_harness_deepseek.json \
    --input data/cmb_sft_v4/cmb_sft_train.jsonl --metadata data/cmb_sft_v4/cmb_sft_metadata.jsonl \
    --exclude-stems <CMB-test json> <CMB-val json> <CMExam test csv> <CMExam val csv> \
    --tasks rationale,paraphrase,statement --limit 20 --output-dir data/synth_smoke --dry-run
Drop --dry-run to make the calls. The API key is read only from the environment variable named by api_key_env in the config.
"""
import argparse
import json
import sys
import time
from pathlib import Path

try:
    from .pipeline import (build_providers, build_tasks, estimate_cost, load_items, load_reference_stems, load_scores, run,
                           select_items, write_outputs)
    from .providers import Budget, ResponseCache
    from ..sentence_split import split_spans
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sentence_split import split_spans
    from pipeline import (build_providers, build_tasks, estimate_cost, load_items, load_reference_stems, load_scores, run,
                          select_items, write_outputs)
    from providers import Budget, ResponseCache


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="messages-format training data (direct-answer or rationale-style rows)")
    parser.add_argument("--metadata", help="matching metadata.jsonl, optional, provides exam_type / variant")
    parser.add_argument("--scores", help="per-item score JSONL (sample_id, p_gold), optional")
    parser.add_argument("--p-max", type=float, default=None, help="keep only items with p_gold <= this value (needs --scores)")
    parser.add_argument("--exclude-stems", nargs="*", default=[], help="evaluation-set files: their stems are never sent to the model and are also used for decontamination")
    parser.add_argument("--allow-no-exclude", action="store_true", help="allow running without reference files (local tests only)")
    parser.add_argument("--tasks", default="rationale,paraphrase,statement")
    parser.add_argument("--exam-types", default="")
    parser.add_argument("--question-types", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--order", choices=("hash", "p_gold"), default="hash", help="p_gold: pick the `limit` hardest items by ascending p_gold (needs --scores)")
    parser.add_argument("--keep-shuffled", action="store_true", help="by default one row per stem (shuffled copies skipped)")
    parser.add_argument("--min-sentences", type=int, default=0, help="keep only items whose rationale has at least this many segments (sentence_score task; 0 = no limit)")
    parser.add_argument("--max-sentences", type=int, default=0, help="keep only items whose rationale has at most this many segments (0 = no limit)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cost", type=float, default=None, help="override the cost cap from the config (CNY)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache", help="response cache file, default <output-dir>/cache.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="select items and estimate the cost without sending requests")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    cfg_text = Path(args.config).read_text(encoding="utf-8")
    cfg = json.loads(cfg_text)
    if not args.exclude_stems and not args.allow_no_exclude:
        print("--exclude-stems must list the evaluation-set files; add --allow-no-exclude for local tests only", file=sys.stderr)
        return 2
    exclude_hashes, decontam_refs = load_reference_stems(args.exclude_stems) if args.exclude_stems else (set(), [])
    items, load_counts = load_items(args.input, args.metadata)
    if args.min_sentences or args.max_sentences:
        # count segments with exactly the scoring task's splitter settings so the filter matches the numbering sent to the model
        score_cfg = cfg.get("tasks", {}).get("sentence_score", {})
        def count(item):
            return len(split_spans(item["explanation"], score_cfg.get("min_chars", 8), score_cfg.get("level", "clause")))
        before = len(items)
        items = [item for item in items if item.get("explanation") and count(item) >= args.min_sentences
                 and (not args.max_sentences or count(item) <= args.max_sentences)]
        load_counts["filtered_sentence_count"] = before - len(items)
    scores = load_scores(args.scores) if args.scores else None
    exam_types = {t for t in args.exam_types.split(",") if t} or None
    question_types = {t for t in args.question_types.split(",") if t} or None
    selected, select_counts = select_items(items, scores, args.p_max, args.limit, args.seed, frozenset(exclude_hashes),
                                           only_original=not args.keep_shuffled, exam_types=exam_types, question_types=question_types,
                                           order=args.order)
    tasks = build_tasks(cfg, [t for t in args.tasks.split(",") if t])
    print(f"loaded {load_counts.get('loaded', 0)} rows, selected {len(selected)}; {len(exclude_hashes)} reference stems set to refuse", file=sys.stderr)
    rows, total = estimate_cost(cfg, tasks, selected)
    for row in rows:
        print(f"[estimate] {row['task']}: {row['items']} items, ~{row['est_input_tokens']} input tokens, ~{row['est_output_tokens']} output tokens, "
              f"~{row['est_cost_cny']} CNY", file=sys.stderr)
    print(f"[estimate] total ~{total} CNY (config list prices; check the provider's current prices)", file=sys.stderr)
    if args.dry_run:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / "dry_run.json", "w", encoding="utf-8") as handle:
            json.dump({"selected": len(selected), "selection_counts": dict(select_counts), "load_counts": dict(load_counts),
                       "estimate": rows, "estimate_total_cny": total}, handle, ensure_ascii=False, indent=2)
        return 0
    if not selected:
        print("no usable items", file=sys.stderr)
        return 1
    max_cost = args.max_cost if args.max_cost is not None else cfg.get("budget", {}).get("max_cost_cny")
    prices = cfg["providers"][cfg["tasks"][tasks[0].name]["provider"]].get("prices_cny_per_million", {})
    budget = Budget(prices, max_cost)
    cache_path = args.cache or str(Path(args.output_dir) / "cache.jsonl")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(cache_path)
    providers = build_providers(cfg)
    records, exports, funnels = run(cfg, selected, tasks, providers, budget, cache, args.workers, decontam_refs, frozenset(exclude_hashes))
    manifest = write_outputs(args.output_dir, cfg, cfg_text, [args.input, args.metadata, args.scores] + list(args.exclude_stems),
                             tasks, records, exports, funnels, budget, select_counts, load_counts, cache_path, started)
    print((Path(args.output_dir) / "funnel.md").read_text(encoding="utf-8"))
    print(f"wrote {args.output_dir}: provenance.jsonl, manifest.json, funnel.md, " + ", ".join(f"data_{k}.jsonl" for k in exports), file=sys.stderr)
    return 0 if not manifest["budget"]["exceeded"] else 3


if __name__ == "__main__":
    sys.exit(main())
