import json
with open('eval/dataset/rag_eval.jsonl', encoding='utf-8') as f:
    lines = [l for l in f if l.strip() and not l.strip().startswith('#')]
data = json.loads('\n'.join(lines))
cases = data if isinstance(data, list) else [data]
print('总题数:', len(cases))
for c in cases:
    print(f"{c['id']}: {c['question'][:40]}... | source={c.get('source',[])}")