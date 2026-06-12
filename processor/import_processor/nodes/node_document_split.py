# processor/import_processor/nodes/node_document_split.py
import json
import logging
import re
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    节点: 文档切分 (node_document_split)
    将长文档切分成小的 Chunks (切片) 以便检索。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_document_split"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点：文档切分（node_document_split）
        整体流程：加载输入→按MD标题初切→长切短合→统计输出→结果备份
        核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
        后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等

        必要参数：md_content、file_title
        更新参数：chunks

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：加载并标准化输入数据
        content, file_title = self._step_1_get_inputs(state)

        # 步骤2：按MD标题进行初次切分
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)

        # 步骤3：无标题场景兜底处理
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        # 步骤4：Chunk精细化处理（长切短合
        sections = self._step_4_refine_chunks(sections)

        # 步骤5：输出文档切分统计信息
        self._step_5_print_stats(lines_count, sections)

        # 步骤6：Chunk结果本地JSON备份
        self._step_6_backup(state, sections)

        # 写入状态字典
        state["chunks"] = sections
        return state


    def _step_6_backup(self, state: ImportGraphState, sections: List[Dict[str, str]]) -> None:
        """
        【步骤6】Chunk结果本地JSON备份（便于调试/问题排查，保留处理结果）
        :param state: 项目状态字典，需包含md_dir（备份目录）
        :param sections: 最终处理后的Chunk列表
        """

        try:
            # 拼接备份文件路径：固定文件名，便于查找
            backup_path = Path("D:/output") / state.get("file_title") / "chunks.json"
            # 写入JSON文件：保留中文/格式化缩进，便于人工查看
            with open(backup_path, "w", encoding="utf-8") as f:
                """
                sections是Python 嵌套数据结构（List[Dict[str, str]]，列表里装字典，字典里可能嵌套字符串 / 数字等），而普通文件写入
                （如f.write(sections)）仅支持写入字符串，直接写 Python 数据结构会报错。
                json.dump的核心作用就是：将 Python 原生数据结构（列表、字典、字符串、数字等）直接序列化并写入 JSON 文件，无需手动转换为字符串，
                同时保证数据格式规范、可跨语言 / 跨场景读取，完美适配「Chunk 列表备份」的需求。
                """
                json.dump(
                    sections,
                    f,
                    #开启 True："title": "\u4e00\u7ea7\u6807\u9898"（乱码，无法直接看）；
                    #开启 False："title": "一级标题"（正常中文，人工可直接阅读）。
                    ensure_ascii=False,  # 保留中文，不转义为\u编码
                    indent=2             # 格式化缩进，便于阅读
                )
            self.logger.info(f"步骤6：Chunk结果备份成功，备份文件路径：{backup_path}")
        except Exception as e:
            # 备份失败仅记录日志，不终止主流程
            self.logger.error(f"步骤6：Chunk结果备份失败，错误信息：{str(e)}", exc_info=False)

    def _step_5_print_stats(self, lines_count: int, sections: List[Dict[str, str]]) -> None:
        """
        【步骤5】输出文档切分统计信息（日志记录，便于监控/调试）
        :param lines_count: MD原始文本总行数
        :param sections: 最终处理后的Chunk列表
        """
        chunk_num = len(sections)
        # 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
        self.logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        self.logger.info(f"MD原始文本总行数：{lines_count}")
        self.logger.info(f"最终生成Chunk数量：{chunk_num}")

    def _split_long_section(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        """
        【辅助函数】超长章节二次切分（核心适配LangChain分割器）
        功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
        切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
        :param section: 原始章节字典，必须包含content键，可选title/file_title等
        :return: 切分后的子章节列表，每个子章节带父标题/序号等元信息
        """
        content = section.get("content")

        # 1. 长度未超限，无需切分，直接返回
        if len(content) <= self.config.max_content_length:
            return [section]

        # 2. 切割文本
        # 章节标题
        title = section.get("title")
        #计算标题的长度
        prefix = f"{title}\n\n" if title else ""

        # 计算正文的长度
        available_len = self.config.max_content_length - len(prefix)

        if available_len <= 0:
            self.logger.warning("章节标题过长，无法切分")
            return [section]


        # 去掉title部分，便于以后判断
        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title) :].lstrip()

        # 对正文进行切割
        # langchain扽文本切割器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len, # 切片长度（扣除标题）
            chunk_overlap=0, #重叠部分的长度
            separators=[
                "\n\n",
                "\n",
                "。",
                "？",
                "！",
                "；",
                ".",
                "?",
                "!",
                ";",
                " ",
            ],
        )

        sub_sections = []
        chunks = splitter.split_text(body)
        for idx, chunk in enumerate(chunks, start=1):
            text = chunk.strip()
            if not text:
                continue

            full_text = (prefix + text).strip()
            new_title = f"{title}-{idx}"

            sub_sections.append(
                {
                    "parent_title": title,
                    "title": new_title,
                    "content": full_text,
                    "part": idx,
                    "file_title": section.get("file_title")
                }
            )

        return sub_sections


    def _step_4_refine_chunks(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
       【步骤4】Chunk精细化处理（核心：长切短合，适配大模型/检索）
       执行流程：1.过滤图片标签 2.切分超长章节 3.合并过短章节 4.父标题兜底（适配Milvus向量库schema）
       :param sections: 步骤3处理后的章节列表
       :return: 长度适中、低碎片化的最终Chunk列表
       """
        # 阶段0：过滤图片标签
        for section in sections:
            section["content"] = self._remove_image_markdown(section["content"])

        # 阶段1：超长段落切分，控制在最大字符数内（self.config.max_content_length）
        refined_split = []
        for section in sections:
            sub_sections = self._split_long_section(section)
            refined_split.extend(sub_sections)

        # 阶段2：短段落合并，减少内容碎片化
        final_sections = self._merge_short_sections(refined_split)

        # 阶段3：父标题兜底
        # 兜底规则：无parent_title则用自身title，title也无则填空字符串
        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""

        return final_sections

    def _remove_image_markdown(self, content: str) -> str:
        """
        过滤Markdown中的图片标签
        只删除本地图片路径（./images/xxx.png），保留MinIO/远程URL的图片
        :param content: 原始内容
        :return: 过滤后的内容
        """
        import re
        def _replace_image(match):
            url = match.group(2)
            # 保留远程URL（http/https开头的），删除本地路径
            if url.startswith('http://') or url.startswith('https://'):
                return match.group(0)
            return ''
        content = re.sub(r'!\[(.*?)\]\((.*?)\)', _replace_image, content)
        content = re.sub(r'<img[^>]*>', '', content, flags=re.IGNORECASE)
        lines = [line for line in content.split('\n') if line.strip()]
        return '\n'.join(lines)

    def _merge_short_sections(self, sections: List[Dict[str,str]]) -> List[Dict[str,str]]:

        #边界处理：空列表直接范湖
        if not sections:
            return []


        #合并后的结果
        merged_sections = []
        # 保存当前要合并的chunk
        current_chunk = None

        for section in sections:
            #获取第一个section赋值给current_chunk
            if current_chunk is None:
                current_chunk = section
                continue

            #段落是否过短
            is_current_short = len(current_chunk["content"]) < self.config.min_content_length
            is_same_parent = current_chunk.get("parent_title") == section.get("parent_title")

            if is_current_short and is_same_parent:
                #合并前清理每个content前面重复的parent_title
                parent_title = section.get("parent_title")
                section_content = section.get("content")
                if parent_title and section_content.startswith(parent_title):
                    section_content = section_content[len(parent_title):].lstrip()

                #合并段落
                current_chunk["content"] += "\n\n" + section_content

                #处理part
                if "part" in section:
                    current_chunk["part"] = section["part"]

            else:
                #保存当前段落
                merged_sections.append(current_chunk)
                current_chunk = section

        if current_chunk is not None:
            merged_sections.append(current_chunk)

        return merged_sections


    def _step_3_handle_no_title(self, content: str, sections: List[Dict[str, str]], title_count: int,
                                file_title: str) -> List[Dict[str, str]]:
        """
        【步骤3】无标题兜底处理
        功能：若MD中未识别到任何标题，将全文作为一个整体处理，避免后续逻辑异常
        :param content: 标准化后的MD完整内容
        :param sections: 步骤2切分后的章节列表
        :param title_count: 步骤2识别的有效标题数量
        :param file_title: 所属文件标题
        :return: 兜底后的章节列表
        """
        if title_count == 0:

            # 将content交给LLM，总结一个标题出来
            # 提示词：根据content和file_title总结一个段落标题，不要有空格，不要换行，控制在20个字符以内

            # 无标题处理
            return [{
                "title": "无标题",
                "content": content,
                "file_title": file_title
            }]

        return sections

    def _step_2_split_by_titles(self, content: str, file_title: str) -> Tuple[List[Dict[str, str]], int, int]:
        """
        【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
        LangChain前置预处理：将整份MD按标题拆分为独立章节，为后续精细化切分做基础
        :param content: 标准化后的MD完整内容（字符串）
        :param file_title: 所属文件标题，用于标记章节归属
        :return: 切分后的章节列表/有效标题数量/原始文本总行数
        """

        # 1、定义标题正则
        # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
        # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
        # #{1,6}：匹配1-6个#（对应MD1-6级标题）
        # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
        # .+：标题文字至少1个字符（避免空标题）
        title_pattern = r'\s*#{1,6}\s+.+'

        # 2、初始化需要的数据
        lines = content.split("\n")
        sections = []  # 章节列表
        title_count = 0  # 标题数量
        current_title = ""  # 当前章节的标题
        current_lines = []  # 当前标题和下一个标题之间的文本内容
        in_code_block = False  # 代码块标记：False当前没在代码块中，True当前在代码块中

        # 3、定义内部函数组装sections列表
        def _flush_section():
            """内部辅助函数：将当前缓存的章节写入sections，空缓存则跳过"""
            if not current_lines:
                return
            sections.append({
                "title": current_title,
                # 每段使用 \n换行区分
                "content": "\n".join(current_lines),
                "file_title": file_title,
            })

        # 4、逐行遍历，识别标题和普通行以及代码快
        for line in lines:
            # 去空格
            striped_line = line.strip()

            # # 4.1 识别代码快边界 ``` 或 ~~~
            # if striped_line.startswith("```") or striped_line.startswith("~~~"):
            #     in_code_block = not in_code_block
            #     current_lines.append(striped_line)
            #     continue
            # 4.1 识别代码块边界 ```、~~~、````、~~~~ 等（至少 3 个连续字符）
            # 使用正则匹配：行首到行尾只有 ` 或 ~ 字符，且数量>=3
            code_block_marker_match = re.match(r'^(`{3,}|~{3,})$', striped_line)
            if code_block_marker_match:
                marker = code_block_marker_match.group(1)

                if not in_code_block:
                    # 进入代码块，记录开始的标记特征
                    in_code_block = True
                    code_block_start_marker = marker
                elif in_code_block and striped_line == code_block_start_marker:
                    # 遇到匹配的结束标记（相同字符且相同长度）
                    in_code_block = False
                    code_block_start_marker = None

                current_lines.append(line)
                continue

            # 4.2 表示标题
            is_valid_title = (not in_code_block) and re.match(title_pattern, line)
            if is_valid_title:
                #遇到标题行则先将上一个片段写入section
                _flush_section()
                current_title = striped_line
                current_lines = [current_title]
                title_count += 1
            else:
                current_lines.append(striped_line)

        _flush_section()

        return sections, title_count, len(lines)


    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        """
        【步骤1】获取并预处理输入数据
        功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
        :param state: 项目状态字典（ImportGraphState），包含md_content等核心键
        :return: 标准化后的MD内容/文件标题（无内容则返回None,None）
        """

        # 中间节点也可以只做轻量级的防御性判断，不做完整的磁盘级别的检查 => 前置校验 + 信任状态

        # 1、非空校验
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(field_name="md_content", message="文件内容不能为空", expected_type=str)

        # 2、基础标准化：统一换行符
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        return md_content, file_title

if __name__ == "__main__":
    setup_logging()

    md_path = r"D:\output\hak180产品安全手册\hak180产品安全手册_new.md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "md_path": md_path,
        "md_content": md_content,
        "file_title": "hak180产品安全手册"
    }

    # 执行核心处理流程
    node_document_split = NodeDocumentSplit()
    result = node_document_split(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))


def safe_split_document(text, chunk_size=500, overlap=50):
    """安全的文档分块，带边界检查"""
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    if overlap >= chunk_size:
        raise ValueError("overlap必须小于chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return chunks


def optimized_split_with_dedup(text, chunk_size=500, overlap=50):
    """优化文档分块算法"""
    if not text:
        return []
    chunks = safe_split_document(text, chunk_size, overlap)
    seen = set()
    unique_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_key = chunk[:100].strip()
        if chunk_key not in seen:
            seen.add(chunk_key)
            if i > 0 and overlap > 0:
                prev_chunk = chunks[i-1]
                if chunk.startswith(prev_chunk[-overlap:]):
                    chunk = chunk[overlap:]
            unique_chunks.append(chunk)
    return unique_chunks