import sys
sys.path.insert(0, 'D:/Demo/knowledge_base')
from utils.milvus_utils import get_milvus_client
from config.milvus_config import milvus_config
from pymilvus import Collection

client = get_milvus_client()
print('Milvus连接成功')
print('集合名:', milvus_config.chunks_collection)

col = Collection(milvus_config.chunks_collection)
col.load()
print('字段:', [f.name for f in col.schema.fields])

# 查询所有chunk的file_title分布
results = col.query(expr='id > 0', output_fields=['id', 'file_title', 'item_name'], limit=1000)
print('\n=== Chunk分布 ===')
from collections import Counter
titles = Counter(r.get('file_title','') for r in results)
for t, c in titles.most_common():
    print(f'  {t}: {c} chunks')