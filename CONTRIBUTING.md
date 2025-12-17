# 贡献指南

感谢你对 Hardware-Aftersales-AI 的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告 Bug

1. 在 [GitHub Issues](https://github.com/CW5201/Hardware-Aftersales-AI/issues) 中搜索是否已有相同问题
2. 如果没有，请创建新的 Issue，并包含以下信息：
   - 问题描述
   - 复现步骤
   - 期望行为
   - 实际行为
   - 环境信息（Python版本、操作系统等）

### 提交新功能

1. Fork 本仓库
2. 创建你的特性分支：`git checkout -b feature/AmazingFeature`
3. 提交你的更改：`git commit -m 'Add some AmazingFeature'`
4. 推送到分支：`git push origin feature/AmazingFeature`
5. 打开一个 Pull Request

### 改进文档

文档的改进同样重要！你可以：
- 修复错别字
- 添加更好的示例
- 翻译文档

## 开发环境设置

### 前置要求

- Python 3.11+
- uv 包管理器
- Milvus
- MongoDB
- MinIO

### 本地开发

```bash
# 1. 克隆仓库
git clone https://github.com/CW5201/Hardware-Aftersales-AI.git
cd Hardware-Aftersales-AI

# 2. 安装依赖
uv sync

# 3. 启动开发服务
uv run python -m web.api.import_service
uv run python -m web.api.query_service
```

## 代码规范

### Python 代码风格

- 遵循 PEP 8 规范
- 使用 type hints
- 编写 docstrings
- 使用 ruff 进行代码检查

```bash
# 代码检查
uv run ruff check .

# 格式化
uv run ruff format .
```

### 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

- `feat:` 新功能
- `fix:` Bug 修复
- `docs:` 文档更新
- `style:` 代码格式（不影响代码运行的变动）
- `refactor:` 重构
- `test:` 测试
- `chore:` 构建过程或辅助工具的变动

示例：
```
feat: 添加新的检索策略
fix: 修复向量搜索超时问题
docs: 更新 API 文档
```

## Pull Request 流程

1. 确保代码通过所有测试
2. 更新相关文档
3. 添加必要的测试用例
4. 填写 PR 描述，说明改动内容
5. 等待 Code Review

## 问题优先级

- **P0 - Critical**: 系统崩溃、数据丢失
- **P1 - High**: 核心功能无法使用
- **P2 - Medium**: 功能异常但有替代方案
- **P3 - Low**: 体验优化、文档改进

## 行为准则

- 尊重每一位贡献者
- 接受建设性的批评
- 专注于对社区最有利的事情
- 对其他社区成员表示同理心

## 获取帮助

如果你有任何问题，可以通过以下方式获取帮助：

- 在 Issue 中提问
- 在 Discussion 中讨论
- 联系维护者

感谢你的贡献！🎉
