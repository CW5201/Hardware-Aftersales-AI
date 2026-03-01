import os
# from FlagEmbedding import FlagReranker

_reranker_model = None
def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model= FlagReranker(
            model_name_or_path=os.getenv("BGE_RERANKER_LARGE"),
            device=os.getenv("BGE_RERANKER_DEVICE" ),
            use_fp16=os.getenv("BGE_RERANKER_FP16" )
        )
    return _reranker_model