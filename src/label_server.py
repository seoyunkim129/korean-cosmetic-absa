"""라벨링 화면(web/labeler.html)을 띄우고, 붙인 라벨을 data/labels.jsonl에 바로바로 저장하는 로컬 서버.

    python src/label_server.py          # http://localhost:8770

저장은 한 줄씩 덧붙이기(append)라 중간에 꺼도 잃지 않는다. 같은 리뷰를 고치면 마지막 줄이 이긴다.
모델 예측은 화면에 보내지 않는다 — 정답을 붙일 때 모델 답에 끌려가지 않게(블라인드 라벨링).
"""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "label_sample.csv"
LABELS = ROOT / "data" / "labels.jsonl"
PAGE = ROOT / "web" / "labeler.html"
PORT = 8770

ASPECTS = ["가격", "기능/효과", "디자인", "발림성", "보습력/수분감", "사용감", "성분", "용기", "용량",
           "유통기한", "윤기/피부(톤)", "자극성", "제품구성", "제형", "지속력", "탄력",
           "편의성/활용성", "품질", "피부타입", "향", "흡수력"]


def load_labels():
    labels = {}
    if LABELS.exists():
        for line in LABELS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                labels[str(rec["review_id"])] = rec
    return labels


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json; charset=utf-8", code=200):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/data":
            sample = pd.read_csv(SAMPLE, encoding="utf-8-sig")
            return self._send(json.dumps({
                "aspects": ASPECTS,
                "sample": sample.to_dict("records"),
                "labels": load_labels(),
            }, ensure_ascii=False))
        self._send('{"error":"not found"}', code=404)

    def do_POST(self):
        if self.path != "/api/label":
            return self._send('{"error":"not found"}', code=404)
        rec = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        bad = [a for a in rec.get("aspects", {}) if a not in ASPECTS]
        if bad or any(s not in ("긍정", "부정", "중립") for s in rec.get("aspects", {}).values()):
            return self._send(json.dumps({"error": f"잘못된 값: {bad}"}, ensure_ascii=False), code=400)
        rec["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with LABELS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._send(json.dumps({"ok": True, "saved_at": rec["saved_at"]}))

    def log_message(self, fmt, *args):
        if "POST" in (args[0] if args else ""):
            print(fmt % args)


if __name__ == "__main__":
    print(f"라벨링 화면: http://localhost:{PORT}  (저장 위치: {LABELS})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
