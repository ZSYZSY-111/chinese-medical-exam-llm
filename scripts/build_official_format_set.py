#!/usr/bin/env python3
"""Re-render a direct-answer training file in the official CMB evaluation prompt (the evaluator's system message
plus the official user template), so that training and evaluation share exactly the same format.

exam_type / exam_class / question_type: CMB rows are looked up in the raw CMB-train file by stem + option-set hash;
CMExam rows have no metadata and use --cmexam-exam-type / --cmexam-exam-class, with the question type inferred from
the number of answer letters. Options keep their order, and an empty option renders as "E. " exactly as in eval_cmb.py.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

try:
    from .build_cmb_train_sft import normalize_stem, stem_option_hash
    from .cmexam_prompts import parse_user_content
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_cmb_train_sft import normalize_stem, stem_option_hash
    from cmexam_prompts import parse_user_content

OFFICIAL_SYSTEM = "你是一个人工智能助手。"


def official_user_prompt(exam_type, exam_class, question_type, question, options):
    option_text = "\n".join(f"{letter}. {content}" for letter, content in options)
    return (f"以下是中国{exam_type}中{exam_class}考试的一道{question_type}，不需要做任何分析和解释，直接输出答案选项。\n"
            f"{question}\n{option_text}\n只能输出选项字母。单选题例如：A；多选题例如：ABC。")


def index_cmb(rows):
    index = {}
    for row in rows:
        option_map = row.get("option") or {}
        options = [(letter, " ".join(str(option_map[letter]).split())) for letter in sorted(option_map)]
        key = stem_option_hash(normalize_stem(" ".join((row.get("question") or "").split())), options)
        index.setdefault(key, row)
    return index


def convert_rows(rows, cmb_index, cmexam_exam_type, cmexam_exam_class):
    out, counts = [], Counter()
    for record in rows:
        messages = record["messages"]
        gold = messages[-1]["content"].strip()
        parsed = parse_user_content([m for m in messages if m["role"] == "user"][0]["content"])
        qtype = "多项选择题" if len(gold) > 1 else "单项选择题"
        source = cmb_index.get(stem_option_hash(normalize_stem(parsed["question"]), parsed["options"]))
        if source is not None:
            counts["cmb_matched"] += 1
            exam_type, exam_class = source.get("exam_type", ""), source.get("exam_class", "")
            if source.get("question_type") in ("单项选择题", "多项选择题"):
                qtype = source["question_type"]
        else:
            counts["unmatched_as_cmexam"] += 1
            exam_type, exam_class = cmexam_exam_type, cmexam_exam_class
        user = official_user_prompt(exam_type, exam_class, qtype, parsed["question"], parsed["options"])
        out.append({"messages": [{"role": "system", "content": OFFICIAL_SYSTEM}, {"role": "user", "content": user},
                                 {"role": "assistant", "content": gold}]})
    return out, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--cmb-train", required=True)
    parser.add_argument("--cmexam-exam-type", default="医师考试")
    parser.add_argument("--cmexam-exam-class", default="执业医师")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(l) for l in open(args.train_file, encoding="utf-8") if l.strip()]
    cmb_index = index_cmb(json.load(open(args.cmb_train, encoding="utf-8")))
    out, counts = convert_rows(rows, cmb_index, args.cmexam_exam_type, args.cmexam_exam_class)
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with open(target / "train.jsonl", "w", encoding="utf-8") as handle:
        for row in out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(open(args.train_file, "rb").read()).hexdigest()
    manifest = {"rows": len(out), "counts": dict(counts), "source_train": args.train_file, "source_sha256": digest,
                "cmexam_labels": {"exam_type": args.cmexam_exam_type, "exam_class": args.cmexam_exam_class}, "format": "cmb_official_prompt"}
    json.dump(manifest, open(target / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
