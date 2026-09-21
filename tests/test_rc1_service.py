"""
v1.0.0-rc1：service 路由 /health 与 DELETE /message 回归测试。

本文件独立运行（`uv run pytest tests/test_rc1_service.py`），
保证 numpy/torch/langsmith 等 C 扩展在同一进程里只初始化一次。

覆盖：
- import_service GET /health  → 200 {"ok": true}
- query_service  GET /health  → 200 {"ok": true}
- 核心回归：query_service DELETE /message/{id} 不再因 delete_message 未定义而 500
"""
import sys
import importlib
from pathlib import Path
from unittest import mock

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bson import ObjectId
from fastapi.testclient import TestClient


# ---- mock 掉基础设施模块，使 service 可无外部依赖导入（本进程仅此一次）----
_infra = mock.patch.dict(
    sys.modules,
    {
        "utils.mongo_history_utils": mock.MagicMock(),
        "utils.minio_utils": mock.MagicMock(),
    },
)
_infra.start()
try:
    _import_app = importlib.import_module("web.api.import_service").app
    _query_app = importlib.import_module("web.api.query_service").app
finally:
    _infra.stop()


def test_import_service_has_health():
    client = TestClient(_import_app, raise_server_exceptions=False)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_query_service_has_health():
    client = TestClient(_query_app, raise_server_exceptions=False)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_query_delete_message_no_longer_500():
    """核心回归：DELETE /message/{id} 不再因 delete_message 未定义而 500"""
    client = TestClient(_query_app, raise_server_exceptions=False)
    valid_id = str(ObjectId())
    resp = client.request("DELETE", f"/message/{valid_id}")
    # 关键断言：不再是 500（内部错误），而是 404（未找到）或 200（删除成功）
    assert resp.status_code in (200, 404)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
