import argparse
import ast
import json
import pandas as pd

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    preds = load_jsonl(args.jsonl_file)
    gt_df = pd.read_csv(args.csv_file)

    correct_all = 0
    total_all = 0
    correct_no_unk = 0
    total_no_unk = 0
    correct_unk = 0
    total_unk = 0
    wrong_preds = []

    for pred in preds:
        gt_labels = ast.literal_eval(gt_df.iloc[pred["index"]]["gt"])
        imdbid = gt_df.iloc[pred["index"]]["imdbid"]

        if imdbid == "tt0119822":
            continue

        pred_text = pred["response_text"]
        pred_labels = pred["response_json"]

        if pred_labels is None:
            pred_labels = {}

        for name in gt_labels:
            pred_label = pred_labels.get(name, "unknown")
            is_correct = pred_label == gt_labels[name]

            total_all += 1
            if is_correct:
                correct_all += 1

            if gt_labels[name] != "unknown":
                total_no_unk += 1
                if is_correct:
                    correct_no_unk += 1
            else:
                total_unk += 1
                if is_correct:
                    correct_unk += 1

            if not is_correct:
                wrong_preds.append({"index": pred["index"], "pred_labels": pred_labels, "gt_labels": gt_labels, "wrong_name": name, "predicted_value": pred_label, "gt_value": gt_labels[name], "pred_text": pred_text})

    if args.log_file is not None:
        with open(args.log_file, 'w') as f:
            json.dump(wrong_preds, f)

    acc_all = correct_all / total_all if total_all > 0 else 0.0
    acc_no_unk = correct_no_unk / total_no_unk if total_no_unk > 0 else 0.0
    acc_unk = correct_unk / total_unk if total_unk > 0 else 0.0
    print(f"[RESULT] Overall Accuracy: {correct_all}/{total_all} = {acc_all*100:.2f}%")
    print(f"[RESULT] Known Accuracy:   {correct_no_unk}/{total_no_unk} = {acc_no_unk*100:.2f}%")
    print(f"[RESULT] Unknown Accuracy: {correct_unk}/{total_unk} = {acc_unk*100:.2f}%")

if __name__ == "__main__":
    main()