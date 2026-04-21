import numpy as np
import os
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from config.embedding_config import embedding_config


_bge_m3_ef = None

def get_bge_m3_ef():
    """
    获取全局单例的BGEM3EmbeddingFunction对象
    :return:  BGEM3EmbeddingFunction对象
    """
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    model_name = embedding_config.bge_m3_path
    device = embedding_config.bge_device
    use_fp16 = embedding_config.bge_fp16

    _bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
        trust_remote_code=True
    )

    return _bge_m3_ef


def generate_embeddings(texts):
    """
    为文本生成向量嵌入
    :param texts: 要生成嵌入的文本列表
    :return: 包含dense和sparse向量的字典
    """

    model = get_bge_m3_ef()
    embeddings = model.encode_documents(texts)

    processed_sparse = []
    for i in range(len(texts)):

        sparse_obj = embeddings["sparse"]
        sparse_indices = sparse_obj.indices[
            sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]].tolist()

        sparse_data = sparse_obj.data[
            sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]].tolist()

        sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
        processed_sparse.append(sparse_dict)

    return {
        "dense": [emb.tolist() for emb in embeddings["dense"]],
        "sparse": processed_sparse
    }



def normalize_sparse_vector(sparse_vec):
    """
    对稀疏向量做 L2 归一化（仅处理非零维度，不影响零维度）
    :param sparse_vec: 原始稀疏向量（dict 格式：{维度: 数值}）
    :return: 归一化后的稀疏向量
    """
    if not sparse_vec:  # 空向量直接返回
        return sparse_vec

    # 提取非零维度的数值
    values = np.array(list(sparse_vec.values()), dtype=np.float64)
    # 计算 L2 范数（避免除以 0）
    l2_norm = np.linalg.norm(values)
    if l2_norm < 1e-9:  # 范数接近 0 时，直接返回原向量（避免除零错误）
        return sparse_vec

    # 归一化：每个数值除以 L2 范数
    normalized_values = values / l2_norm
    # normalized_values = (values / l2_norm).astype(np.float32)  # 统一转为 float32
    # 重建稀疏向量 dict
    return dict(zip(sparse_vec.keys(), normalized_values))


def validate_embedding_dim(vector, expected_dim=1024):
    """修复embedding维度检查错误"""
    if not vector or len(vector) == 0:
        raise ValueError("Embedding向量不能为空")
    if len(vector) != expected_dim:
        raise ValueError(f"Embedding维度错误: 期望{expected_dim}, 实际{len(vector)}")
    return True


def safe_generate_embeddings(texts):
    """安全的embedding生成，带异常处理"""
    try:
        return generate_embeddings(texts)
    except Exception as e:
        raise RuntimeError(f"Embedding生成失败: {str(e)}")