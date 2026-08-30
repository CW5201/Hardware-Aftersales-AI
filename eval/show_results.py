import csv
import json
from collections import Counter

with open("eval/dataset/rag_eval.jsonl", encoding="utf-8") as f:
    lines = [l for l in f if l.strip() and not l.strip().startswith("#")]
cases = [json.loads(l) for l in lines]
qt = {c["id"]: set(c.get("source", [])) for c in cases}
print("=== TEST SET ===")
print("Total:", len(cases))
print("Types:", Counter(c.get("type", "?") for c in cases))
print("Diff:", Counter(c.get("difficulty", "?") for c in cases))
for c in cases:
    print(" ", c["id"], c.get("type", "?"), c.get("difficulty", "?"), c.get("source", []))

rows = []
with open("eval/results/retrieval_results.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)
print("CSV rows:", len(rows))

methods = ["hybrid", "hybrid_rrf", "hybrid_rrf_rerank", "hyde_hybrid_rrf_rerank"]
topks = [2, 3, 5, 10]
grid = {(m, k): [] for m in methods for k in topks}
for r in rows:
    m, k = r["method"], int(r["top_k"])
    el = float(r["elapsed_sec"]) if r["elapsed_sec"] else 0.0
    pre = int(r["pre_rerank"]) if r["pre_rerank"] else None
    post = int(r["post_rerank"]) if r["post_rerank"] else None
    cliff = int(r["cliff_cutoff"]) if r["cliff_cutoff"] else None
    rid = json.loads(r["retrieved_ids"]) if r["retrieved_ids"] else []
    grid[(m, k)].append({"qid": r["id"], "el": el, "pre": pre, "post": post, "cliff": cliff, "ids": rid})

print()
hdr = "%-30s %3s %7s %7s %8s %9s %8s %8s %8s" % ("METHOD", "K", "Hit@K", "MRR", "Recall@K", "Delay(s)", "PreRerank", "PostRerank", "Cliff")
print(hdr)
print("-" * 100)
for m in methods:
    for k in topks:
        items = grid[(m, k)]
        if not items:
            continue
        n = len(items)
        h = mrr = r = el = 0.0
        tp = pp = po = pc = tc = ccl = 0
        for it in items:
            src = qt.get(it["qid"], set())
            if not src:
                continue
            ids = [str(x) for x in it["ids"]]
            hit = 1 if any(any(s in iid for iid in ids) for s in src) else 0
            fr = 0
            for ri, iid in enumerate(ids):
                if any(s in iid for s in src):
                    fr = ri + 1
                    break
            rec = sum(1 for s in src if any(s in iid for iid in ids)) / len(src)
            h += hit
            mrr += (1.0 / fr if fr > 0 else 0.0)
            r += rec
            el += it["el"]
            if it["pre"] is not None:
                tp += it["pre"]
                pp += 1
            if it["post"] is not None:
                po += it["post"]
                pc += 1
            if it["cliff"] is not None:
                tc += it["cliff"]
                ccl += 1
        lbl = m.replace("hyde_", "HYDE_").replace("_", " ")
        pr = "%.2f" % (tp / pp) if pp else "N/A"
        po2 = "%.2f" % (po / pc) if pc else "N/A"
        cl = "%.2f" % (tc / ccl) if ccl else "N/A"
        print("%-30s %3d %7.2f %7.3f %8.3f %9.3f %8s %8s %8s" % (lbl, k, h / n, mrr / n, r / n, el / n, pr, po2, cl))