"""올리브영 리뷰 전체에 속성 분류기 → 감정 분류기를 돌려 리뷰×속성 단위 예측을 저장한다.

원본: Colab `0529_pipeline_oliveyoung.ipynb` (STEP 1~3)를 그대로 옮긴 것.
노트북과 같은 모델 구조·토크나이저·MAX_LEN·THRESHOLD를 쓴다.

    python src/pipeline.py              # data/ 아래 파일을 읽어 outputs/predictions.csv 저장
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

MAX_LEN = 192
THRESHOLD = 0.6
LABEL_MAP = {0: "부정", 1: "중립", 2: "긍정"}
EXCLUDE = ["밀착력/접착력", "커버력", "색상"]
BATCH = 32


class RobertaClassifier(nn.Module):
    """속성 분류기(21)·감정 분류기(3) 공용 구조: RoBERTa pooler → dropout → linear."""

    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("klue/roberta-base")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.pooler_output))


def aspect_classes():
    # 노트북과 같은 방식: 학습 데이터에서 3개 제외 후 MultiLabelBinarizer 순서(= 정렬)
    df = pd.read_csv(DATA / "bert_train_skincare_12k.csv")
    return sorted(df.loc[~df["Aspect"].isin(EXCLUDE), "Aspect"].unique())


def load(num_classes, filename):
    model = RobertaClassifier(num_classes)
    state = torch.load(DATA / filename, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def predict_aspects(texts, model, tok, classes):
    probs = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], max_length=MAX_LEN, padding="max_length",
                  truncation=True, return_tensors="pt")
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs.append(torch.sigmoid(logits).numpy())
    probs = np.vstack(probs)
    return probs, [[classes[j] for j in np.where(p > THRESHOLD)[0]] for p in probs]


@torch.no_grad()
def predict_sentiments(pairs, model, tok):
    rows = []
    for i in range(0, len(pairs), BATCH):
        texts, aspects = zip(*pairs[i:i + BATCH])
        enc = tok(list(texts), list(aspects), max_length=MAX_LEN, padding="max_length",
                  truncation=True, return_tensors="pt")
        p = torch.softmax(model(enc["input_ids"], enc["attention_mask"]), dim=1).numpy()
        rows.extend(p)
    return np.array(rows)


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained("klue/roberta-base")
    classes = aspect_classes()
    print(f"속성 {len(classes)}개: {classes}")

    reviews = pd.read_csv(DATA / "oliveyoung_reviews_clean_vfff.csv", encoding="utf-8-sig")
    # 노트북 run_pipeline과 같은 단위: 제품별 unique 리뷰
    reviews = reviews.drop_duplicates(["제품명", "리뷰텍스트"]).reset_index(drop=True)
    print(f"리뷰 {len(reviews)}건 / 제품 {reviews['제품명'].nunique()}개")

    aspect_model = load(len(classes), "aspect_classifier_v2_21cls.pth")
    _, detected = predict_aspects(reviews["리뷰텍스트"].tolist(), aspect_model, tok, classes)
    del aspect_model

    pairs, keys = [], []
    for idx, (text, asps) in enumerate(zip(reviews["리뷰텍스트"], detected)):
        for a in asps:
            pairs.append((text, a))
            keys.append(idx)
    print(f"감지된 리뷰×속성 쌍 {len(pairs)}개")

    sent_model = load(3, "klueroberta_sentiment_best.pth")
    probs = predict_sentiments(pairs, sent_model, tok)

    out = pd.DataFrame({
        "review_id": keys,
        "제품명": reviews.loc[keys, "제품명"].values,
        "리뷰텍스트": [t for t, _ in pairs],
        "속성": [a for _, a in pairs],
        "감정": [LABEL_MAP[i] for i in probs.argmax(1)],
        "p_부정": probs[:, 0].round(4),
        "p_중립": probs[:, 1].round(4),
        "p_긍정": probs[:, 2].round(4),
    })
    out.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {OUT / 'predictions.csv'}")


if __name__ == "__main__":
    main()
