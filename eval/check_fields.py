import sys
sys.path.insert(0, 'D:/Demo/knowledge_base')
from utils.milvus_utils import get_milvus_client
from config.milvus_config import milvus_config

client = get_milvus_client()
all_chunks = client.query(collection_name=milvus_config.chunks_collection, output_fields=['id','content','file_title','item_name'], limit=5)
print('字段:', list(all_chunks[0].keys()) if all_chunks else [])
print('第一条:', json.dumps(all_chunks[0], ensure_ascii=False, default=str)[:300] if all_chunks else '空')

import json