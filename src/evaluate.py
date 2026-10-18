"""사람이 붙인 올리브영 라벨(data/labels.jsonl)로 모델의 실제 성능을 잰다.

AI Hub 테스트셋 점수(속성 Macro F1 0.932 @0.6, 감정 Macro F1 0.892)가
학습에 안 쓴 올리브영 리뷰에서는 얼마인지 — 도메인 갭을 숫자로 본다.

1. 속성 감지: 리뷰마다 정답 속성 집합 vs 모델이 0.6 넘게 잡은 속성 집합
2. 감정 분류: 정답 속성을 그대로 넣었을 때 감정이 맞는지 (속성 오류와 분리해서 본다)
3. 끝까지(end-to-end): (속성, 감정) 쌍이 통째로 맞는지

    python src/evaluate.py      # 결과 출력 + outputs/eval_errors.csv (틀린 사례)
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer

from pipeline import LABEL_MAP, load, predict_sentiments

ROOT = Path(__file__).resolve().parents[1]
SENTS = ["부정", "중립", "긍정"]


def jsonl(path, key):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec[key]] = rec                     # 같은 항목은 마지막 저장이 이긴다
    return out


def read_labels():
    """정답 만들기.
    - data/labels.jsonl이 있으면: 사람이 처음부터 붙인 라벨을 그대로 쓴다.
    - 없으면 (방식 C): LLM과 모델이 합의한 쌍 + 갈린 쌍은 사람 판정(data/judgments.jsonl).
      합의 감사 항목도 사람 판정을 따른다.
    """
    if (ROOT / "data" / "labels.jsonl").exists():
        return jsonl(ROOT / "data" / "labels.jsonl", "review_id"), None

    queue = pd.read_csv(ROOT / "data" / "review_queue.csv", encoding="utf-8-sig")
    judged = jsonl(ROOT / "data" / "judgments.jsonl", "item_id")
    llm = jsonl(ROOT / "data" / "llm_labels.jsonl", "review_id")
    labels = {rid: {"aspects": dict(r["aspects"]), "note": ""} for rid, r in llm.items()}

    for q in queue.itertuples():
        j = judged.get(q.item_id)
        if j is None:
            raise SystemExit(f"판정이 안 끝났어요: {len(judged)}/{len(queue)}개 판정됨")
        asp = labels[q.review_id]["aspects"]
        asp.pop(q.속성, None)                       # LLM 답을 지우고 사람 판정으로 덮는다
        if j["value"] != "언급 없음":
            asp[q.속성] = j["value"]
        if j.get("note"):
            labels[q.review_id]["note"] = j["note"]
    # 판정 결과에서 누가 맞았는지 (LLM vs 모델, 합의 감사)
    queue["사람"] = [judged[i]["value"] for i in queue.item_id]
    return labels, queue


def judge_summary(queue):
    norm = lambda v: v if isinstance(v, str) else "언급 없음"
    q = queue.assign(llm=queue["llm"].map(norm), model=queue["model"].map(norm))
    print("[판정 요약] 갈린 항목에서 사람 판정과 일치한 비율")
    for kind, g in q.groupby("유형"):
        print(f"  {kind:8s} {len(g):3d}개 · LLM 맞음 {(g.llm == g.사람).mean():.0%} · 모델 맞음 {(g.model == g.사람).mean():.0%}")
    print()


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    labels, queue = read_labels()
    if queue is not None:
        judge_summary(queue)
    sample = pd.read_csv(ROOT / "data" / "label_sample.csv", encoding="utf-8-sig").set_index("review_id")
    pred = pd.read_csv(ROOT / "outputs" / "predictions.csv", encoding="utf-8-sig")
    pred = pred[pred["review_id"].isin(labels)]
    ids = sorted(labels)
    print(f"라벨 {len(ids)}건 (샘플 {len(sample)}건 중)\n")

    gold_pairs = {(rid, a): s for rid in ids for a, s in labels[rid]["aspects"].items()}
    pred_pairs = {(r.review_id, r.속성): r.감정 for r in pred.itertuples()}

    # 1. 속성 감지
    gold_asp, pred_asp = set(gold_pairs), set(pred_pairs)
    per = []
    for a in sorted({a for _, a in gold_asp | pred_asp}):
        tp = len({k for k in gold_asp & pred_asp if k[1] == a})
        fp = len({k for k in pred_asp - gold_asp if k[1] == a})
        fn = len({k for k in gold_asp - pred_asp if k[1] == a})
        per.append((a, tp + fn, *prf(tp, fp, fn)))
    per = pd.DataFrame(per, columns=["속성", "정답수", "P", "R", "F1"]).sort_values("정답수", ascending=False)
    p, r, f = prf(len(gold_asp & pred_asp), len(pred_asp - gold_asp), len(gold_asp - pred_asp))
    macro = per.loc[per["정답수"] > 0, "F1"].mean()
    print(f"[1] 속성 감지  micro P {p:.3f} · R {r:.3f} · F1 {f:.3f} | macro F1 {macro:.3f} (정답 있는 속성 {int((per['정답수']>0).sum())}개)")
    print(per.round(3).to_string(index=False), "\n")

    # 2. 감정 분류 — 정답 속성을 넣어서
    keys = list(gold_pairs)
    tok = AutoTokenizer.from_pretrained("klue/roberta-base")
    model = load(3, "klueroberta_sentiment_best.pth")
    probs = predict_sentiments([(sample.loc[rid, "리뷰텍스트"], a) for rid, a in keys], model, tok)
    y_true = [gold_pairs[k] for k in keys]
    y_pred = [LABEL_MAP[i] for i in probs.argmax(1)]
    print("[2] 감정 분류 (정답 속성 입력)")
    print(classification_report(y_true, y_pred, labels=SENTS, digits=3, zero_division=0))
    print("혼동행렬 (행=정답, 열=예측)", SENTS)
    print(confusion_matrix(y_true, y_pred, labels=SENTS), "\n")

    # 3. end-to-end
    hit = sum(1 for k, s in gold_pairs.items() if pred_pairs.get(k) == s)
    p, r, f = prf(hit, len(pred_pairs) - hit, len(gold_pairs) - hit)
    print(f"[3] 속성+감정 end-to-end  P {p:.3f} · R {r:.3f} · F1 {f:.3f}")

    # 틀린 사례 모으기 (오류 유형 분석용)
    rows = []
    for (rid, a), s in gold_pairs.items():
        sp = y_pred[keys.index((rid, a))]
        kind = "속성 놓침" if (rid, a) not in pred_pairs else None
        if sp != s:
            kind = (kind + " + " if kind else "") + f"감정 {s}→{sp}"
        if kind:
            rows.append((rid, a, kind, s, pred_pairs.get((rid, a)), labels[rid].get("note", ""), sample.loc[rid, "리뷰텍스트"]))
    for (rid, a) in pred_asp - gold_asp:
        rows.append((rid, a, "속성 과잉", None, pred_pairs[(rid, a)], labels[rid].get("note", ""), sample.loc[rid, "리뷰텍스트"]))
    err = pd.DataFrame(rows, columns=["review_id", "속성", "오류", "정답감정", "모델감정(파이프라인)", "메모", "리뷰"])
    err.to_csv(ROOT / "outputs" / "eval_errors.csv", index=False, encoding="utf-8-sig")
    print(f"\n틀린 사례 {len(err)}건 → outputs/eval_errors.csv")
    print(err["오류"].str.split(" \\+ ").explode().str.replace(r"감정 .*", "감정 오분류", regex=True).value_counts().to_string())


if __name__ == "__main__":
    main()
