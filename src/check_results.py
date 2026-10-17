"""predictions.csv를 노트북과 같은 규칙으로 집계해 ① 노트북 결과(oliveyoung_results.csv) 재현 여부,
② 라운드랩 독도 토너 향·제형의 실제 긍·부·중립 분포를 확인한다."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pred = pd.read_csv(ROOT / "outputs" / "predictions.csv", encoding="utf-8-sig")
nb = pd.read_csv(ROOT / "data" / "oliveyoung_results.csv", encoding="utf-8-sig")

MIN_MENTIONS = 5
cnt = (pred.groupby(["제품명", "속성"])["감정"].value_counts().unstack(fill_value=0)
       .reindex(columns=["긍정", "부정", "중립"], fill_value=0))
cnt["총언급"] = cnt.sum(axis=1)
pct = cnt[["긍정", "부정", "중립"]].div(cnt["총언급"], axis=0).mul(100).round(1)
agg = pct.join(cnt["총언급"])
agg = agg[agg["총언급"] >= MIN_MENTIONS].reset_index()

# ① 재현 확인
m = nb.merge(agg, on=["제품명", "속성"], how="left", suffixes=("_nb", "_local"))
same = ((m["총언급_nb"] == m["총언급_local"])
        & (m["긍정_nb"] - m["긍정_local"]).abs().le(0.1)
        & (m["부정_nb"] - m["부정_local"]).abs().le(0.1))
print(f"[재현] 노트북 결과 {len(m)}행 중 일치 {same.sum()}행")
if (~same).any():
    print(m.loc[~same, ["제품명", "속성", "총언급_nb", "총언급_local", "긍정_nb", "긍정_local",
                         "부정_nb", "부정_local"]].head(20).to_string())

# ② 독도 토너
TONER = "라운드랩 1025 독도 토너 500ml 기획"
print(f"\n[{TONER}]")
t = cnt.loc[TONER].sort_values("총언급", ascending=False)
print(t[t["총언급"] >= MIN_MENTIONS].to_string())
for asp in ["향", "제형"]:
    print(f"\n--- {asp} 리뷰 ---")
    for _, r in pred[(pred["제품명"] == TONER) & (pred["속성"] == asp)].iterrows():
        print(f"[{r['감정']} {max(r['p_부정'], r['p_중립'], r['p_긍정']):.2f}] {r['리뷰텍스트'][:110]}")

# 전체 감정 분포
print("\n[전체 리뷰×속성 감정 분포]")
print(pred["감정"].value_counts(normalize=True).round(3).to_string())
