"""LLM 라벨과 모델 예측이 갈리는 (리뷰, 속성)만 사람이 판정하도록 판정 큐를 만든다.

- 둘 다 같은 속성·같은 감정 → 합의로 보고 정답에 넣는다
  (단, 둘이 같이 틀리는 경우를 재기 위해 합의 중 일부를 무작위로 감사 항목으로 섞는다)
- 한쪽만 잡은 속성, 또는 감정이 다른 속성 → 사람이 판정

판정 화면에는 어느 쪽이 무슨 답을 냈는지 보여주지 않는다(블라인드).

    python src/make_review_queue.py            # → data/review_queue.csv
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT_N = 20
SEED = 42


def main():
    sample = pd.read_csv(ROOT / "data" / "label_sample.csv", encoding="utf-8-sig").set_index("review_id")
    llm = {}
    for line in (ROOT / "data" / "llm_labels.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        for a, s in rec["aspects"].items():
            llm[(rec["review_id"], a)] = s
    pred = pd.read_csv(ROOT / "outputs" / "predictions.csv", encoding="utf-8-sig")
    pred = pred[pred["review_id"].isin(sample.index)]
    model = {(r.review_id, r.속성): r.감정 for r in pred.itertuples()}

    rows, agreed = [], []
    for k in sorted(set(llm) | set(model)):
        a, b = llm.get(k), model.get(k)
        if a == b:
            agreed.append(k)
            continue
        kind = "감정 불일치" if a and b else ("LLM만 속성" if a else "모델만 속성")
        rows.append((*k, kind))
    audit = pd.Series(agreed).sample(min(AUDIT_N, len(agreed)), random_state=SEED).tolist()
    rows += [(*k, "합의 감사") for k in audit]

    q = pd.DataFrame(rows, columns=["review_id", "속성", "유형"])
    q["llm"] = [llm.get((r, a)) for r, a in zip(q.review_id, q.속성)]
    q["model"] = [model.get((r, a)) for r, a in zip(q.review_id, q.속성)]
    q["제품명"] = sample.loc[q.review_id, "제품명"].values
    q["리뷰텍스트"] = sample.loc[q.review_id, "리뷰텍스트"].values
    q = q.sample(frac=1, random_state=SEED).reset_index(drop=True)   # 유형이 티 나지 않게 섞기
    q.insert(0, "item_id", q.index)
    q.to_csv(ROOT / "data" / "review_queue.csv", index=False, encoding="utf-8-sig")

    print(f"LLM 쌍 {len(llm)} · 모델 쌍 {len(model)} · 합의 {len(agreed)}")
    print(q["유형"].value_counts().to_string())
    print(f"판정할 항목 {len(q)}개 (리뷰 {q.review_id.nunique()}건)")


if __name__ == "__main__":
    main()
