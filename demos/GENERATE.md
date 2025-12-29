# 演示内容生成指南

本指南将帮助你生成项目的演示截图和 GIF 动图。

## 准备工作

### 1. 启动项目

```bash
# 启动导入服务
uv run python -m web.api.import_service

# 启动查询服务
uv run python -m web.api.query_service
```

### 2. 准备示例文档

准备一些示例 PDF 文档用于演示：
- 产品手册
- 技术文档
- 用户指南

## 生成截图

### 推荐工具

1. **Windows Snipping Tool** (Windows)
2. **SnagIt** (Windows/Mac) - 专业截图工具
3. **Greenshot** (Windows) - 开源免费
4. **macOS Screenshot** (Mac)

### 截图清单

#### 1. 文档导入界面

打开 http://localhost:8000/import.html，截图包含：
- 文件上传区域
- 进度显示
- 完成状态

#### 2. 对话问答界面

打开 http://localhost:8001/chat.html，截图包含：
- 聊天界面
- 产品确认流程
- 回答结果
- 参考资料

#### 3. 查询进度展示

展示完整的查询流程：
- 进度步骤显示
- 流式回答
- 图片展示

## 生成 GIF 动图

### 推荐工具

1. **LICEcap** (Windows/Mac) - 轻量级
2. **Kap** (Mac) - 开源免费
3. **Peek** (Linux) - 简单易用
4. **OBS Studio** - 专业级

### GIF 清单

#### 1. 完整演示 (full-demo.gif)

录制 10-15 秒的完整流程：
1. 打开导入界面
2. 上传 PDF 文档
3. 等待处理完成
4. 打开聊天界面
5. 输入问题
6. 查看回答

#### 2. 导入流程 (import-flow.gif)

录制 5-10 秒的导入过程：
1. 选择 PDF 文件
2. 上传进度
3. 处理步骤
4. 完成状态

#### 3. 查询流程 (query-flow.gif)

录制 8-12 秒的查询过程：
1. 输入问题
2. 产品确认（如有）
3. 进度显示
4. 流式回答
5. 参考资料

## 录制技巧

### 分辨率

- 推荐：1280x720 (HD)
- 备选：1920x1080 (Full HD)

### 帧率

- 15-30 FPS 即可
- GIF 不需要太高帧率

### 长度

- 截图：静态即可
- GIF：5-15 秒为宜
- 过长会影响加载速度

### 优化

```bash
# 使用 gifsicle 优化 GIF
gifsicle -O3 --colors 256 input.gif -o output.gif

# 使用 ffmpeg 转换
ffmpeg -i input.mp4 -vf "fps=15,scale=1280:-1" output.gif
```

## 文件命名

### 截图

- `import.png` - 导入界面
- `chat.png` - 聊天界面
- `progress.png` - 进度显示

### GIF

- `full-demo.gif` - 完整演示
- `import-flow.gif` - 导入流程
- `query-flow.gif` - 查询流程

## 提交演示内容

1. 将截图放入 `screenshots/` 目录
2. 将 GIF 放入 `demos/` 目录
3. 更新 `demos/README.md` 文件
4. 提交到 Git 仓库

## 注意事项

1. **隐私**：确保演示内容不包含敏感信息
2. **大小**：GIF 文件尽量小于 5MB
3. **质量**：确保文字清晰可读
4. **更新**：项目更新后记得更新演示内容

## 示例截图描述

### 导入界面截图

```
显示：
- 左侧：文件上传区域，显示"拖拽或点击上传 PDF 文件"
- 中间：进度条，显示"正在处理..."
- 右侧：完成状态，显示"✓ 处理完成"
```

### 聊天界面截图

```
显示：
- 左侧：历史对话列表
- 中间：聊天区域
  - 用户：这款打印机的耗材型号是什么？
  - 助手：根据知识库查询，华为B3-243H显示器的耗材型号是...
  - 参考资料：华为B3-243H用户指南
- 右侧：产品确认弹窗
```

---

如果需要帮助生成演示内容，请随时提问！
