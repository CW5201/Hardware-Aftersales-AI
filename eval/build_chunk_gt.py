"""
Chunk-level Ground Truth 构建脚本

逻辑：
1. 读取 rag_eval.jsonl 测试集
2. 从 Milvus 加载所有 chunk
3. 对每个问题，按 source 文件过滤，用关键词+内容匹配定位 gold chunk
4. 输出 rag_eval_chunk_level.jsonl
"""
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymilvus import MilvusClient
import numpy as np
from config.milvus_config import milvus_config


def load_test_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = [line for line in content.split("\n")
             if line.strip() and not line.strip().startswith("#")]
    return json.loads("\n".join(lines))


def load_all_chunks() -> list:
    c = MilvusClient(milvus_config.milvus_url)
    res = c.search(
        collection_name=milvus_config.chunks_collection,
        data=[np.zeros(1024, dtype=np.float32)],
        anns_field='dense_vector',
        limit=500,
        output_fields=['chunk_id', 'content', 'file_title', 'item_name', 'title', 'parent_title']
    )
    chunks = []
    for hits in res:
        for h in hits:
            e = h.entity
            chunks.append({
                "chunk_id": e.get("chunk_id"),
                "content": e.get("content", ""),
                "file_title": e.get("file_title", ""),
                "item_name": e.get("item_name", ""),
                "title": e.get("title", ""),
                "parent_title": e.get("parent_title", ""),
                "score": h.distance,
            })
    return chunks


def normalize_key(s: str) -> str:
    """标准化字符串用于匹配"""
    return re.sub(r'[^\w一-鿿]', '', s.lower()).strip()


def chunk_matches_question(chunk: dict, question: str, answer: str, keywords: list, source_files: list) -> tuple:
    """
    判断 chunk 是否匹配问题
    返回 (score, reason)
    score 越高越匹配，0 表示不匹配
    """
    reasons = []
    score = 0

    content = chunk["content"]
    title = chunk["title"] or ""
    full_text = f"{title} {content}"

    # 1. 文件名匹配（支持多源时任意一个匹配即可）
    matched_source = False
    for sf in source_files:
        sf_norm = normalize_key(sf)
        ft_norm = normalize_key(chunk["file_title"])
        if sf_norm in ft_norm or ft_norm in sf_norm:
            matched_source = True
            break
    if not matched_source:
        return 0, "source file mismatch"

    # 2. 关键词匹配
    kw_matched = []
    q_norm = normalize_key(question)
    a_norm = normalize_key(answer)
    text_norm = normalize_key(full_text)

    for kw in keywords:
        kw_norm = normalize_key(kw)
        if kw_norm and kw_norm in text_norm:
            kw_matched.append(kw)
            score += 2

    # 3. 问题关键词在 chunk 中
    for word in re.findall(r'[一-鿿]{2,}|[a-zA-Z]+', question):
        if len(word) >= 2 and word in text_norm:
            score += 1

    # 4. 答案关键词在 chunk 中
    for word in re.findall(r'[一-鿿]{2,}|[a-zA-Z]+', answer):
        if len(word) >= 2 and word in text_norm:
            score += 1

    # 5. 标题包含问题核心词
    for word in re.findall(r'[一-鿿]{2,}|[a-zA-Z]+', question):
        if len(word) >= 3 and word in normalize_key(title):
            score += 3

    if not kw_matched:
        return 0, "no keyword match"

    reason = f"source={chunk['file_title']}, keywords_match={[','.join(kw_matched[:3])]}"
    if score >= 5:
        reason += f", content_match"
    return score, reason


def find_gold_chunks(question: dict, all_chunks: list) -> list:
    """为一道题找到所有匹配的 gold chunks"""
    q = question["question"]
    a = question["answer"]
    src = question.get("source", [])
    kws = question.get("keywords", [])

    candidates = []
    for ch in all_chunks:
        score, reason = chunk_matches_question(ch, q, a, kws, src)
        if score >= 3:
            candidates.append({
                "chunk_id": str(ch["chunk_id"]),
                "source": ch["file_title"],
                "score": score,
                "reason": reason,
                "title": ch["title"][:60] if ch["title"] else "",
                "content_preview": ch["content"][:120].replace("\n", " "),
            })

    # 去重（同一 chunk 可能多次匹配）
    seen = set()
    unique = []
    for c in sorted(candidates, key=lambda x: -x["score"]):
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique.append(c)

    return unique


def find_gold_chunks_for_comparison(question: dict, all_chunks: list) -> list:
    """
    针对跨文档对比题的特殊处理：
    每个 source 文件分别匹配，取各文件得分最高的 chunk
    使用更宽松的阈值（>=1 即可），因为对比题的关键词分散在不同文档中
    """
    src = question.get("source", [])
    q = question["question"]
    a = question["answer"]
    kws = question.get("keywords", [])

    candidates = []
    for sf in src:
        best_for_source = None
        best_score = 0
        for ch in all_chunks:
            score, reason = chunk_matches_question(ch, q, a, kws, [sf])
            # 跨文档时，只要 source 匹配且有任意关键词命中就接受
            if score >= 1 and score > best_score:
                best_score = score
                best_for_source = {
                    "chunk_id": str(ch["chunk_id"]),
                    "source": ch["file_title"],
                    "score": score,
                    "reason": reason,
                    "title": ch["title"][:60] if ch["title"] else "",
                    "content_preview": ch["content"][:120].replace("\n", " "),
                }
        if best_for_source and best_score >= 1:
            candidates.append(best_for_source)

    # 去重
    seen = set()
    unique = []
    for c in sorted(candidates, key=lambda x: -x["score"]):
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique.append(c)

    return unique


def main():
    dataset_path = Path(__file__).parent / "dataset" / "rag_eval.jsonl"
    output_path = Path(__file__).parent / "dataset" / "rag_eval_chunk_level.jsonl"

    print("=" * 60)
    print("Chunk-level Ground Truth 构建")
    print("=" * 60)

    # 1. 加载测试集
    dataset = load_test_dataset(str(dataset_path))
    eval_dataset = [d for d in dataset if d.get("type") != "out_of_scope"]
    print(f"\n测试集: {len(dataset)} 条, 有效: {len(eval_dataset)} 条")

    # 2. 加载 Milvus chunks
    print("\n加载 Milvus chunks...")
    all_chunks = load_all_chunks()
    print(f"  Milvus 共 {len(all_chunks)} 条 chunk")

    # 3. 为每道题找 gold chunk
    results = []
    stats = {"total": len(eval_dataset), "matched": 0, "needs_review": 0, "unmatched": 0}

    for case in eval_dataset:
        qid = case["id"]
        print(f"\n[{qid}] {case['question'][:50]}...")
        print(f"  source: {case.get('source')}")
        print(f"  keywords: {case.get('keywords')}")

        golds = find_gold_chunks(case, all_chunks)
        # 如果单文件匹配为0且是多源问题，尝试跨文档匹配
        if len(golds) == 0 and len(case.get("source", [])) > 1:
            golds = find_gold_chunks_for_comparison(case, all_chunks)
        print(f"  匹配到 {len(golds)} 个候选 chunk:")

        gold_chunks = []
        for g in golds[:5]:  # 最多取5个
            print(f"    chunk_id={g['chunk_id']} score={g['score']} reason={g['reason'][:60]}")
            print(f"    preview: {g['content_preview'][:80]}...")
            gold_chunks.append({
                "chunk_id": g["chunk_id"],
                "source": g["source"],
                "reason": g["reason"],
            })

        if len(golds) >= 2:
            stats["matched"] += 1
        elif len(golds) == 1:
            # 只有一个匹配，需要确认
            stats["needs_review"] += 1
        else:
            stats["unmatched"] += 1
            print(f"  >>> 无法匹配，标记为 needs_manual_review")

        result = {
            "id": qid,
            "question": case["question"],
            "answer": case["answer"],
            "source": case.get("source", []),
            "keywords": case.get("keywords", []),
            "gold_chunks": gold_chunks,
            "needs_manual_review": len(golds) < 2,
        }
        results.append(result)

    # 4. 保存
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print("完成！结果已保存到:", output_path)
    print(f"  total_questions: {stats['total']}")
    print(f"  matched_questions: {stats['matched']}")
    print(f"  needs_manual_review: {stats['needs_review']}")
    print(f"  unmatched_questions: {stats['unmatched']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
