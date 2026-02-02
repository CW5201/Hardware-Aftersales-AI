
from modelscope.hub.snapshot_download import snapshot_download

# 下载 BGE Reranker Large 模型
model_dir = snapshot_download('BAAI/bge-reranker-large', cache_dir='D:/ai_models/modelscope_cache/models')
print(f"模型已下载到：{model_dir}")
