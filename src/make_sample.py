"""올리브영 리뷰에서 사람이 직접 라벨을 붙일 평가용 샘플을 뽑는다.

제품마다 같은 수만큼 뽑아(층화) 특정 제품에 쏠리지 않게 하고, 같은 문장이 반복된 리뷰는 한 번만 넣는다.

    python src/make_sample.py            # 제품당 5건 → data/label_sample.csv
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--per-product", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

df = pd.read_csv(ROOT / "data" / "oliveyoung_reviews_clean_vfff.csv", encoding="utf-8-sig")
# predictions.csv의 review_id와 같은 번호를 쓰도록 pipeline.py와 같은 순서로 중복 제거
df = df.drop_duplicates(["제품명", "리뷰텍스트"]).reset_index(drop=True)
df["review_id"] = df.index
df = df.drop_duplicates("리뷰텍스트")

sample = (df.groupby("제품명").sample(frac=1, random_state=args.seed)
            .assign(rank=lambda d: d.groupby("제품명").cumcount())
            .loc[lambda d: d["rank"] < args.per_product]
            .sample(frac=1, random_state=args.seed)       # 제품 순서대로 나오지 않게 섞기
            [["review_id", "제품명", "리뷰텍스트"]])
out = ROOT / "data" / "label_sample.csv"
sample.to_csv(out, index=False, encoding="utf-8-sig")
print(f"{len(sample)}건 ({sample['제품명'].nunique()}개 제품) → {out}")
