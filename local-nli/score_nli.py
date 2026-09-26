"""Local, offline scorer for the Jev study: a zero-shot NLI classifier on the same items as Jev and Haiku.

Runs inside the jev-study-nli container (no network). It reads the question wording directly from
run.py, so the pre-registered criteria are identical for every model, and writes results/nli.jsonl
in the same format run.py uses, so analyze.py scores it like any other model.

How a decision is made: each option's criteria text is a hypothesis, the item text is the premise.
The model scores how strongly the premise entails each hypothesis; a softmax over those entailment
scores gives one probability per option (the standard single-label zero-shot method). The model reads
at most 512 tokens, so a long premise is cut from the end; every result records whether it was cut.

  python /app/score_nli.py --smoke     # 3 items per task
  python /app/score_nli.py             # everything; resumes if stopped
  python /app/score_nli.py --data-dir kit/<repo>/data --results-dir kit/<repo>/results   # own-repo kit
"""
import argparse, json, math, os, sys, time
from pathlib import Path

HARNESS = Path(os.environ.get("NLI_HARNESS", "/harness"))
MODEL_DIR = os.environ.get("NLI_MODEL_DIR", "/models/deberta")
MODEL_ID = "moritzlaurer/deberta-v3-large-zeroshot-v2.0"
MODEL_REVISION = os.environ.get("NLI_MODEL_REVISION", "cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2")
MAX_LEN = 512
DATA = HARNESS / "data"
OUT = HARNESS / "results" / "nli.jsonl"

sys.path.insert(0, str(HARNESS))
from run import QUESTIONS, TASKS  # noqa: E402  the pre-registered wording


def options_for(task):
    """(label, hypothesis) pairs, in a fixed order, from the pre-registered criteria."""
    Q = QUESTIONS[task]
    if Q["kind"] == "choice":
        return list(Q["criteria"].items())
    return [("yes", Q["criteria"]["true"]), ("no", Q["criteria"]["false"])]


class Scorer:
    def __init__(self, model_dir=MODEL_DIR):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tensors = "pt"
        torch.set_num_threads(int(os.environ.get("NLI_THREADS") or os.cpu_count() or 4))
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.entail = label2id["entailment"]

    def encode(self, premise, hyps):
        full = max(len(self.tok(premise, h)["input_ids"]) for h in hyps)
        enc = self.tok([premise] * len(hyps), hyps, truncation="only_first", max_length=MAX_LEN,
                       padding=True, return_tensors=self.tensors)
        return enc, full > MAX_LEN

    def forward(self, enc):
        with self.torch.inference_mode():
            logits = self.model(**enc).logits
        return logits[:, self.entail].tolist()


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [v / sum(e) for v in e]


def decide(scorer, item):
    opts = options_for(item["task"])
    enc, truncated = scorer.encode(item["text"], [h for _, h in opts])
    t0 = time.monotonic()
    ent = scorer.forward(enc)
    ms = (time.monotonic() - t0) * 1000
    probs = dict(zip([k for k, _ in opts], softmax(ent)))
    label = max(probs, key=probs.get)
    return label, probs[label], probs, truncated, ms


def load_items():
    items = []
    for t in TASKS.values():
        f = DATA / t["file"]
        if f.exists():
            items += [json.loads(l) for l in f.open(encoding="utf-8")]
        else:
            print(f"skipping {f.name}: not found")
    return items


def done_ids():
    if not OUT.exists():
        return set()
    return {json.loads(l)["id"] for l in OUT.open(encoding="utf-8") if json.loads(l).get("ok")}


def main(smoke=False, scorer=None):
    items = load_items()
    if smoke:
        items = [i for t in TASKS for i in [x for x in items if x["task"] == t][:3]]
    done = done_ids()
    todo = [i for i in items if i["id"] not in done]
    print(f"nli: {len(todo)} to do ({len(done)} already done)", flush=True)
    scorer = scorer or Scorer()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as out:
        for n, item in enumerate(todo, 1):
            rec = {"id": item["id"], "task": item["task"], "split": item["split"], "gold": item["label"],
                   "model": MODEL_ID, "served_model": f"{MODEL_ID}@{MODEL_REVISION}", "provider": "local",
                   "cost_usd": 0.0, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                label, conf, probs, truncated, ms = decide(scorer, item)
                rec.update(ok=True, pred=label, confidence=conf, probs=probs, truncated=truncated, latency_ms=round(ms, 1))
            except Exception as e:  # kept and scored as wrong, like any failed call
                rec.update(ok=False, final=True, error=f"{type(e).__name__}: {e}", latency_ms=0.0)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if smoke:
                print(f"  {item['id']}: gold={item['label']} pred={rec.get('pred')} conf={rec.get('confidence', 0):.2f} "
                      f"{rec['latency_ms']:.0f}ms truncated={rec.get('truncated')} {rec.get('error', '')}", flush=True)
            elif n % 50 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-dir", default="data", help="relative to the harness folder")
    ap.add_argument("--results-dir", default="results", help="relative to the harness folder")
    a = ap.parse_args()
    DATA, OUT = HARNESS / a.data_dir, HARNESS / a.results_dir / "nli.jsonl"
    main(a.smoke)
