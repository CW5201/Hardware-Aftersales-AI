"""
tests/test_vue_workbench.py

Step 12 Vue3 工作台（前端）测试（node 可用时跑 `npm run build`，否则 Not verified）：
  1. 关键源文件存在（package.json / vite.config / src/main / App / router /
     api / 8 个页面 / AgentLifecycle 组件）
  2. vite build 成功（node_modules 存在时）
  3. 8 个页面路由都在 router 里
  4. 页面覆盖 Agent 生命周期（Triage→…→Ticket 7 步在 Chat/Dashboard/Trace）
  5. 无 node 时明确 "Not verified in current environment"（不伪造构建结果）
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
WB = project_root / "web" / "workbench"

PAGES = [
    "Dashboard.vue",
    "Chat.vue",
    "Devices.vue",
    "Tickets.vue",
    "Approvals.vue",
    "Trace.vue",
    "Evaluation.vue",
    "KnowledgeBase.vue",
]

REQUIRED_FILES = [
    "package.json",
    "vite.config.ts",
    "tsconfig.json",
    "index.html",
    "src/main.ts",
    "src/App.vue",
    "src/router/index.ts",
    "src/api.ts",
    "src/types.ts",
    "src/components/AgentLifecycle.vue",
    *["src/pages/" + p for p in PAGES],
]

LIFECYCLE_STEPS = ["Triage", "Memory", "Retrieval", "Diagnosis", "Tool", "Approval", "Ticket"]


def test_all_required_files_exist():
    missing = [f for f in REQUIRED_FILES if not (WB / f).exists()]
    assert not missing, f"缺少文件: {missing}"


def test_router_covers_all_eight_pages():
    router_src = (WB / "src" / "router" / "index.ts").read_text(encoding="utf-8")
    for p in PAGES:
        assert p in router_src, f"router 未包含 {p}"
    for path in (
        "/dashboard", "/chat", "/devices", "/tickets",
        "/approvals", "/trace", "/evaluation", "/knowledge",
    ):
        assert path in router_src, f"router 缺路由 {path}"


def test_pages_show_agent_lifecycle():
    """重点不是聊天框，而是展示 Agent 生命周期 7 步。"""
    chat = (WB / "src" / "pages" / "Chat.vue").read_text(encoding="utf-8")
    dashboard = (WB / "src" / "pages" / "Dashboard.vue").read_text(encoding="utf-8")
    trace = (WB / "src" / "pages" / "Trace.vue").read_text(encoding="utf-8")
    for step in LIFECYCLE_STEPS:
        assert step in chat, f"Chat 未展示 {step}"
        assert step in dashboard, f"Dashboard 未展示 {step}"
        assert step in trace, f"Trace 未展示 {step}"


def test_chat_shows_state_evidence_tool_diagnosis_approval():
    chat = (WB / "src" / "pages" / "Chat.vue").read_text(encoding="utf-8")
    for kw in ("当前 State", "Evidence", "Diagnosis", "Triage", "Retrieval", "Trace"):
        assert kw in chat, f"Chat 缺 {kw}"


def test_trace_shows_input_output_tokens_latency_error():
    trace = (WB / "src" / "pages" / "Trace.vue").read_text(encoding="utf-8")
    for kw in ("Input", "Output", "Tokens", "Latency", "Status", "Tool Result"):
        assert kw in trace, f"Trace 缺 {kw}"


def test_vite_build_succeeds():
    """node_modules 存在且 node 可用时跑 build；否则 Not verified。"""
    node = shutil.which("node")
    if not node:
        pytest.skip("Not verified in current environment: node 不可用")
    if not (WB / "node_modules").exists():
        pytest.skip("Not verified in current environment: 未安装 npm 依赖（node_modules 缺失）")

    res = subprocess.run(
        [node, "node_modules/vite/bin/vite.js", "build"],
        cwd=str(WB),
        capture_output=True, text=True, timeout=300,
    )
    assert res.returncode == 0, f"vite build 失败:\n{res.stdout}\n{res.stderr}"
    assert (WB / "dist" / "index.html").exists(), "build 未产出 dist/index.html"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
