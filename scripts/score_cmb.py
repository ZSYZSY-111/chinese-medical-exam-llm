import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


ANSWER_FIELDS = (
    "answer",
    "model_answer",
    "correct_answer",
    "standard_answer",
    "gold_answer",
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_records(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("data", "test", "questions", "records", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value

        # 兼容 {"1": {...}, "2": {...}} 形式
        if all(isinstance(v, dict) for v in data.values()):
            records = []
            for key, value in data.items():
                item = dict(value)
                item.setdefault("id", key)
                records.append(item)
            return records

    raise ValueError("无法识别 JSON 记录结构")


def normalize_answer(value):
    if value is None:
        return ""

    if isinstance(value, list):
        value = "".join(str(x) for x in value)

    text = str(value).upper()

    text = text.translate(
        str.maketrans(
            "ＡＢＣＤＥＦＧａｂｃｄｅｆｇ",
            "ABCDEFGabcdefg",
        )
    ).upper()

    letters = re.findall(r"[A-G]", text)

    # 去重并固定为 A、B、C……顺序
    selected = set(letters)
    return "".join(
        letter for letter in "ABCDEFG"
        if letter in selected
    )


def get_answer_from_item(item):
    for field in ANSWER_FIELDS:
        if field in item:
            return normalize_answer(item[field])
    return ""


def load_questions(path):
    records = extract_records(load_json(path))
    result = {}

    for index, item in enumerate(records):
        qid = str(item.get("id", index + 1))
        result[qid] = item

    return result


def load_ground_truth(path):
    data = load_json(path)
    answers = {}

    if isinstance(data, dict):
        # 先检查是否是 {"data": [...]} 等结构
        for key in ("data", "test", "answers", "records", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if isinstance(data, list):
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            qid = str(item.get("id", index + 1))
            answer = get_answer_from_item(item)

            if answer:
                answers[qid] = answer

        return answers

    if isinstance(data, dict):
        # 兼容 {"1": "A", "2": "BCD"}
        for qid, value in data.items():
            if isinstance(value, dict):
                answer = get_answer_from_item(value)
            else:
                answer = normalize_answer(value)

            if answer:
                answers[str(qid)] = answer

        return answers

    raise ValueError("无法识别标准答案结构")


def load_predictions(path):
    records = extract_records(load_json(path))
    predictions = {}

    for index, item in enumerate(records):
        qid = str(item.get("id", index + 1))

        if qid in predictions:
            raise ValueError(f"{path} 中存在重复 ID：{qid}")

        predictions[qid] = normalize_answer(
            item.get("model_answer", item.get("answer", ""))
        )

    return predictions


def exact_mcnemar_pvalue(improved, regressed):
    discordant = improved + regressed

    if discordant == 0:
        return 1.0

    try:
        from scipy.stats import binomtest

        return float(
            binomtest(
                improved,
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    except ImportError:
        return None


def group_statistics(rows, field):
    groups = defaultdict(list)

    for row in rows:
        group_name = str(row.get(field) or "未知")
        groups[group_name].append(row)

    output = []

    for group_name, group_rows in groups.items():
        total = len(group_rows)
        base_correct = sum(x["base_correct"] for x in group_rows)
        sft_correct = sum(x["sft_correct"] for x in group_rows)

        base_acc = base_correct / total if total else 0
        sft_acc = sft_correct / total if total else 0

        output.append(
            {
                "group": group_name,
                "total": total,
                "base_correct": base_correct,
                "base_accuracy": base_acc,
                "sft_correct": sft_correct,
                "sft_accuracy": sft_acc,
                "delta_percentage_points": (sft_acc - base_acc) * 100,
            }
        )

    output.sort(
        key=lambda x: (
            -x["delta_percentage_points"],
            -x["total"],
            x["group"],
        )
    )

    return output


def write_group_csv(path, rows):
    fieldnames = [
        "group",
        "total",
        "base_correct",
        "base_accuracy",
        "sft_correct",
        "sft_accuracy",
        "delta_percentage_points",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            formatted = dict(row)
            formatted["base_accuracy"] = round(
                formatted["base_accuracy"], 6
            )
            formatted["sft_accuracy"] = round(
                formatted["sft_accuracy"], 6
            )
            formatted["delta_percentage_points"] = round(
                formatted["delta_percentage_points"], 4
            )
            writer.writerow(formatted)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--questions", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--out-dir", required=True)

    args = parser.parse_args()

    questions = load_questions(args.questions)
    gold = load_ground_truth(args.answers)
    base = load_predictions(args.base)
    sft = load_predictions(args.sft)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_ids = set(gold)
    base_ids = set(base)
    sft_ids = set(sft)

    missing_base = sorted(gold_ids - base_ids)
    missing_sft = sorted(gold_ids - sft_ids)

    if missing_base:
        raise ValueError(
            f"原始模型缺少 {len(missing_base)} 道题，"
            f"例如：{missing_base[:10]}"
        )

    if missing_sft:
        raise ValueError(
            f"SFT 模型缺少 {len(missing_sft)} 道题，"
            f"例如：{missing_sft[:10]}"
        )

    rows = []

    for qid in sorted(gold_ids):
        gold_answer = gold[qid]
        base_answer = base[qid]
        sft_answer = sft[qid]

        question = questions.get(qid, {})

        base_correct = base_answer == gold_answer
        sft_correct = sft_answer == gold_answer

        rows.append(
            {
                "id": qid,
                "exam_type": question.get("exam_type", "未知"),
                "exam_class": question.get("exam_class", "未知"),
                "question_type": question.get(
                    "question_type", "未知"
                ),
                "question": question.get("question", ""),
                "gold_answer": gold_answer,
                "base_answer": base_answer,
                "sft_answer": sft_answer,
                "base_correct": base_correct,
                "sft_correct": sft_correct,
            }
        )

    total = len(rows)
    base_correct_count = sum(row["base_correct"] for row in rows)
    sft_correct_count = sum(row["sft_correct"] for row in rows)

    improved_rows = [
        row for row in rows
        if not row["base_correct"] and row["sft_correct"]
    ]

    regressed_rows = [
        row for row in rows
        if row["base_correct"] and not row["sft_correct"]
    ]

    both_correct = sum(
        row["base_correct"] and row["sft_correct"]
        for row in rows
    )

    both_wrong = sum(
        not row["base_correct"] and not row["sft_correct"]
        for row in rows
    )

    base_accuracy = base_correct_count / total
    sft_accuracy = sft_correct_count / total

    pvalue = exact_mcnemar_pvalue(
        len(improved_rows),
        len(regressed_rows),
    )

    summary = {
        "total": total,
        "base": {
            "correct": base_correct_count,
            "accuracy": base_accuracy,
        },
        "sft": {
            "correct": sft_correct_count,
            "accuracy": sft_accuracy,
        },
        "comparison": {
            "accuracy_delta": sft_accuracy - base_accuracy,
            "delta_percentage_points": (
                sft_accuracy - base_accuracy
            ) * 100,
            "improved_questions": len(improved_rows),
            "regressed_questions": len(regressed_rows),
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "mcnemar_exact_pvalue": pvalue,
        },
    }

    with open(
        out_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(
        out_dir / "question_level_results.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    with open(
        out_dir / "sft_improved_questions.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for row in improved_rows:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    with open(
        out_dir / "sft_regressed_questions.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for row in regressed_rows:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    write_group_csv(
        out_dir / "by_exam_type.csv",
        group_statistics(rows, "exam_type"),
    )

    write_group_csv(
        out_dir / "by_exam_class.csv",
        group_statistics(rows, "exam_class"),
    )

    write_group_csv(
        out_dir / "by_question_type.csv",
        group_statistics(rows, "question_type"),
    )

    print("=" * 65)
    print(f"总题数：{total}")
    print()
    print(
        f"原始模型：{base_correct_count}/{total} "
        f"= {base_accuracy:.4%}"
    )
    print(
        f"SFT 模型：{sft_correct_count}/{total} "
        f"= {sft_accuracy:.4%}"
    )
    print()
    print(
        "准确率变化："
        f"{(sft_accuracy - base_accuracy) * 100:+.4f} 个百分点"
    )
    print(f"SFT 新增答对：{len(improved_rows)}")
    print(f"SFT 导致退步：{len(regressed_rows)}")
    print(f"两者都答对：{both_correct}")
    print(f"两者都答错：{both_wrong}")

    if pvalue is not None:
        print(f"McNemar 精确检验 p-value：{pvalue:.6g}")
    else:
        print("未安装 scipy，暂未计算显著性检验。")

    print("=" * 65)
    print(f"结果目录：{out_dir}")


if __name__ == "__main__":
    main()
