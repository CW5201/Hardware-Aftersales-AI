import sys
sys.path.insert(0, 'D:/Demo/knowledge_base')
from utils.milvus_utils import get_milvus_client
from config.milvus_config import milvus_config
import json

client = get_milvus_client()
print('Milvus连接成功')
print('集合名:', milvus_config.chunks_collection)

# 用 MilvusClient 方式查询
all_chunks = client.query(collection_name=milvus_config.chunks_collection, output_fields=['id','content','file_title','item_name'], limit=5000)
print(f'总chunk数: {len(all_chunks)}')

# 按file_title分组统计
from collections import Counter
titles = Counter(r.get('file_title','') for r in all_chunks)
print('\n=== Chunk分布 ===')
for t, c in titles.most_common():
    print(f'  {t}: {c} chunks')

# 打印各文档的前几条chunk内容
print('\n=== 各文档样本chunk ===')
for title in titles.keys():
    sample = [r for r in all_chunks if r.get('file_title','') == title][:2]
    print(f'\n[{title}]')
    for r in sample:
        cid = r['id']
        content = r.get('content','')[:150].replace('\n',' ')
        print(f'  chunk_id={cid}: {content}...')