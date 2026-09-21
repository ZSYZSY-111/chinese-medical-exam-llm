#!/usr/bin/env python3
"""把 messages 格式的 CMB 内部验证集还原成 CMB 官方 JSON（含 exam_class），供 eval_cmb.py 用官方 prompt 评测。

按“归一化题干 + 选项集合”哈希回查 CMB-train-merge.json 取 exam_type / exam_class / exam_subject / question_type；
输出 questions.json（官方结构，无答案）与 answers.json（[{id, answer}]）。id 直接用 64 位 sample_id，与 headroom_records 的 sample_id 完全一致。
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


def sample_id_of(messages):
    users = [m["content"] for m in messages if m.get("role") == "user"]
    return hashlib.sha256("\n".join(users).encode("utf-8")).hexdigest()


def index_cmb(rows):
    """hash -> 第一条原始行（同哈希的重复行元数据几乎相同，取第一条）。"""
    index = {}
    for row in rows:
        option_map = row.get("option") or {}
        options = [(letter, " ".join(str(option_map[letter]).split())) for letter in sorted(option_map)]
        key = stem_option_hash(normalize_stem(" ".join((row.get("question") or "").split())), options)
        index.setdefault(key, row)
    return index


def convert(val_rows, cmb_index):
    questions, answers, counts = [], [], Counter()
    for record in val_rows:
        messages = record["messages"]
        try:
            parsed = parse_user_content([m for m in messages if m["role"] == "user"][0]["content"])
        except (ValueError, IndexError):
            counts["unparsable"] += 1
            continue
        gold = messages[-1]["content"].strip()
        key = stem_option_hash(normalize_stem(parsed["question"]), parsed["options"])
        source = cmb_index.get(key)
        if source is None:
            counts["no_source_match"] += 1
            exam_type, exam_class, subject, qtype = "医学", "", "", "多项选择题" if len(gold) > 1 else "单项选择题"
        else:
            counts["matched"] += 1
            exam_type, exam_class, subject, qtype = (source.get("exam_type", ""), source.get("exam_class", ""),
                                                     source.get("exam_subject", ""), source.get("question_type", ""))
        qid = sample_id_of(messages)
        questions.append({"id": qid, "exam_type": exam_type, "exam_class": exam_class, "exam_subject": subject,
                          "question": parsed["question"], "question_type": qtype,
                          "option": {letter: text for letter, text in parsed["options"]}})
        answers.append({"id": qid, "answer": gold})
    return questions, answers, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--cmb-train", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    val_rows = [json.loads(l) for l in open(args.val_file, encoding="utf-8") if l.strip()]
    cmb_index = index_cmb(json.load(open(args.cmb_train, encoding="utf-8")))
    questions, answers, counts = convert(val_rows, cmb_index)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(questions, open(out / "questions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(answers, open(out / "answers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({"questions": len(questions), **counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
