"""
v1.0.0-rc1：delete_message 数据访问层测试。
只测 utils.mongo_history_utils.delete_message 的真实逻辑（不触发 service 导入链，
避免 numpy/torch 等 C 扩展在同一 pytest 进程里被重复初始化的问题）。

覆盖：删除存在/不存在消息、非法 message_id、Mongo 异常 四种情况。
不依赖真实 MongoDB。
"""
import sys
import importlib
from pathlib import Path
from unittest import mock

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bson import ObjectId


def _valid_id() -> str:
    return str(ObjectId())


def _load_real_module():
    """独立加载真实模块源码，拿到真实 delete_message（不 mock 该模块本身）"""
    sys.modules.pop("utils.mongo_history_utils", None)
    real_module = importlib.import_module("utils.mongo_history_utils")
    return real_module


def _run_delete(message_id, deleted_count=1, raise_error=False):
    real_module = _load_real_module()
    delete_message = real_module.delete_message
    fake_collection = mock.MagicMock()
    if raise_error:
        fake_collection.delete_one.side_effect = RuntimeError("mongo connection lost")
    else:
        fake_collection.delete_one.return_value = mock.Mock(deleted_count=deleted_count)
    fake_tool = mock.MagicMock()
    fake_tool.chat_message = fake_collection
    with mock.patch.object(real_module, "get_history_mongo_tool", return_value=fake_tool):
        return delete_message(message_id)


def test_delete_message_existing():
    """删除存在的消息：返回删除数 1"""
    assert _run_delete(_valid_id(), deleted_count=1) == 1


def test_delete_message_not_found():
    """删除不存在的消息：记录不存在，deleted_count=0，不抛异常"""
    assert _run_delete(_valid_id(), deleted_count=0) == 0


def test_delete_message_invalid_id():
    """非法 message_id（非合法 ObjectId 字符串）：捕获异常，返回 0"""
    assert _run_delete("not-a-valid-objectid", deleted_count=1) == 0


def test_delete_message_mongo_exception():
    """Mongo 连接异常：捕获异常，返回 0，不向上抛"""
    assert _run_delete(_valid_id(), raise_error=True) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
