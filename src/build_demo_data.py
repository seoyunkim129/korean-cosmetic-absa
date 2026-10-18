"""웹 데모(docs/demo)가 읽을 집계 데이터를 만든다.

리뷰 원문은 넣지 않고, 제품×속성별 긍정·부정·중립 건수만 담는다.
만족도·개인화 지수와 추천 순위는 페이지에서 노트북과 같은 규칙으로 계산한다.

    python src/build_demo_data.py      # outputs/predictions.csv → docs/demo/data.js
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

pred = pd.read_csv(ROOT / "outputs" / "predictions.csv", encoding="utf-8-sig")
reviews = pd.read_csv(ROOT / "data" / "oliveyoung_reviews_clean_vfff.csv", encoding="utf-8-sig")
n_reviews = reviews.drop_duplicates(["제품명", "리뷰텍스트"]).groupby("제품명").size()

MIN_MENTIONS = 5      # 속성별 결과에 표시하는 최소 언급 수 (노트북과 같음)


def indices(counts):
    """노트북 run_pipeline의 지수 계산을 그대로 옮긴 것 (반올림 순서까지 같게).
    기본 지수 = 속성별 긍정 비율을 '이 제품에서 그 속성이 언급된 비율'로 가중 평균.
    개인화 지수 = 고른 속성 긍정 비율 × 0.5 + 나머지 속성 동적 가중 평균 × 0.5."""
    scores = {}
    for a, (pos, neg, neu) in counts.items():
        total = pos + neg + neu
        if total >= MIN_MENTIONS:
            scores[a] = {"긍정": round(pos / total * 100, 1), "총언급": total}
    tm = sum(s["총언급"] for s in scores.values())
    base = round(sum(s["긍정"] * s["총언급"] / tm for s in scores.values()), 1) if tm else 0
    personal = {}
    for my in scores:
        ot = sum(s["총언급"] for a, s in scores.items() if a != my)
        other = sum(s["긍정"] * s["총언급"] / ot for a, s in scores.items() if a != my) if ot else 0
        personal[my] = round(scores[my]["긍정"] * 0.5 + other * 0.5, 1)
    return base, personal


cnt = (pred.groupby(["제품명", "속성"])["감정"].value_counts().unstack(fill_value=0)
           .reindex(columns=["긍정", "부정", "중립"], fill_value=0))
products = []
for name in sorted(n_reviews.index):
    rows = cnt.loc[name] if name in cnt.index.get_level_values(0) else pd.DataFrame()
    counts = {a: [int(r["긍정"]), int(r["부정"]), int(r["중립"])] for a, r in rows.iterrows()}
    base, personal = indices(counts)
    products.append({"name": name, "reviews": int(n_reviews[name]), "aspects": counts,
                     "base": base, "personal": personal})

out = ROOT / "docs" / "demo" / "data.js"
out.parent.mkdir(parents=True, exist_ok=True)
payload = json.dumps({
    "source": "올리브영 5월 인기 기초케어 34개 제품 · 추천순 상위 리뷰 (2026.05 수집)",
    "total_reviews": int(n_reviews.sum()),
    "products": products,
}, ensure_ascii=False, separators=(",", ":"))
# <script src>로 읽어서 file://로 열어도, GitHub Pages에서도 동작하게
out.write_text(f"window.ABSA_DATA = {payload};\n", encoding="utf-8")
print(f"{len(products)}개 제품, 리뷰 {n_reviews.sum()}건 → {out} ({out.stat().st_size // 1024}KB)")
