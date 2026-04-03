# processor/import_processor/nodes/node_pdf_to_md.py
import json
import logging
import shutil
import time
import zipfile
from pathlib import Path

import requests

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ConfigurationError, \
    PdfConversionError
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def fix_chinese_encoding(text):
        """修复中文编码问题"""
        if text is None:
            return ""
        try:
            return text.encode('utf-8').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            return text.encode('latin-1').decode('utf-8', errors='ignore')

    def process(self, state: ImportGraphState):
        """

        :param state:
        :return:
        """

        # 1. 参数校验并返回Path结果
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 2. 将PDF上传到MinerU并轮询结果最后得到解压文件路径
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)
        self.logger.info(zip_url)

        # 3. 下载ZIP包并解压并且得到md的绝对路径
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        # 4. 将MD文件的内容都取出来
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 5. 返回结果
        state["md_content"] = md_content
        state["md_path"] = md_path
        return state
        # return {
        #     "md_content": md_content,
        #     "md_path": md_path
        # }

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        """
        步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件
        核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
        参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
        返回：最终MD文件的字符串格式绝对路径
        异常：RuntimeError(下载失败)
        """

        # 1.下载zip
        response = requests.get(zip_url)

        if response.status_code != 200:
            raise FileProcessingError("下载失败")

        #定义做包的保存路径
        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        with open(zip_save_path, "wb") as f:
            f.write(response.content)

        self.log_step("ZIP下载", "下载成功")

        # 2. 解压ZIP
        # 解压目录
        extract_target_dir = output_dir_obj / pdf_stem

        # 删除已有目录
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)

        # 创建新目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 解压ZIP
        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)

        self.log_step("ZIP下载", "解压完成")

        target_md_file = extract_target_dir / "full.md"
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        target_md_file.rename(new_md_path)

        return str(new_md_path.absolute())


    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """
        步骤2：上传PDF至MinerU并轮询解析任务状态
        核心流程：配置校验 → 获取上传链接 → 文件上传 → 任务轮询（直至完成/失败/超时）
        参数：pdf_path_obj-已校验的PDF Path对象
        返回：解析结果ZIP包下载链接full_zip_url
        异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
        """
        # 1、配置文件校验
        if not self.config.mineru_base_url:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_BASE_URL 参数")
        if not self.config.mineru_api_token:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_API_TOKEN 参数")

        # 2. 调用MinerU的远程API接口，获取上传链接
        # 2.1 组织上传需要的数据
        token = self.config.mineru_api_token
        url = f"{self.config.mineru_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }

        # 2.2 发送请求
        response = requests.post(url, headers=header, json=data)

        # 2.3 获取响应结果
        if response.status_code != 200:
            raise PdfConversionError(
                '获取上传链接失败. 响应状态:{} ,响应结果:{}'.format(response.status_code, response))

        result = response.json()
        if result["code"] != 0:
            raise PdfConversionError('获取上传链接失败. 返回数据：{}'.format(result["msg"]))

        # 2.4 获取到上传链接
        # 批量提取任务 id，可用于批量查询解析结果
        batch_id = result["data"]["batch_id"]
        # 文件上传链接
        urls = result["data"]["file_urls"]

        # 3. 文件上传
        with open(pdf_path_obj, 'rb') as f:
            res_upload = requests.put(urls[0], data=f)
            if res_upload.status_code != 200:
                raise PdfConversionError(f'{urls[0]}文件上传失败')

            self.logger.info('上传文件成功!')

        # 4. 获取解析结果
        poll_url = f"{self.config.mineru_base_url}/extract-results/batch/{batch_id}"

        # 轮询获取
        start_time = time.time()  # 记录当前时间
        timeout_seconds = 600  # 最大超时时间
        poll_interval = 3  # 轮询间隔
        self.log_step(
            step_name="轮询开始",
            message=f"轮询间隔: {poll_interval}s, 超时时间: {timeout_seconds}s，batch_id:{batch_id}"
        )
        while True:
            #已消耗时间
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
              raise TimeoutError(f"[任务轮询]超时，已消耗时间: {int(elapsed_time)}s，batch_id:{batch_id}")

            # 获取任务结果
            try:
                poll_res = requests.get(poll_url, headers=header, timeout=10)
            except Exception as e:
                self.logger.warning(f"网络请求异常，{poll_interval}s后重试，batch_id:{batch_id}")
                self.logger.warning(f"异常信息：{str(e)}")
                time.sleep(poll_interval)
                continue

            # print(poll_res.status_code)
            # print(poll_res.json())
            # print(poll_res.json()["data"])
            if poll_res.status_code != 200:
                raise PdfConversionError(f'[任务轮询]失败，状态码：{poll_res.status_code}')

            poll_res_json = poll_res.json()
            if poll_res_json["code"] != 0:
                raise PdfConversionError(f'[任务轮询]失败，错误码：{poll_res_json["code"]}')

            extract_results = poll_res_json["data"]["extract_result"]
            extract_result = extract_results[0]
            # 获取任务的状态值
            data_state = extract_result["state"]
            if data_state == "done":
                self.log_step(
                    step_name="任务轮询",
                    message=f"解析完成s, 总耗时: {int(elapsed_time)}s，batch_id:{batch_id}"
                )

                full_zip_url = extract_result["full_zip_url"]
                self.log_step(
                    step_name="任务轮询",
                    message=f"获取全量zip地址成功: {full_zip_url}, 总耗时: {int(elapsed_time)}s，batch_id:{batch_id}"
                )

                return full_zip_url

            elif data_state == "failed":
                raise PdfConversionError(f'[任务轮询]失败，错误信息: {extract_result["err_msg"]}')

            else:
                self.log_step(
                    step_name="任务轮询",
                    message=f"处理中...... 已耗时: {int(elapsed_time)}s，状态：{data_state}，batch_id:{batch_id}"
                )
                time.sleep(poll_interval)


    def _step_1_validate_paths(self, state: ImportGraphState):
        """
        步骤1：校验PDF文件路径和输出目录
        核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
        返回：合法的PDF文件Path对象、输出目录Path对象
        异常：StateFieldError(参数缺失)、FileNotFoundError(文件无效)
        """

        # 1.参数的非空校验
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path", message="PDF路径不能为空", expected_type=str)

        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir", message="输出路径不能为空", expected_type=str)

        # 2. 转换为Path对象
        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        # 3.pdf是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(f"文件{pdf_path_obj.name}不存在")

        # 4.输出目录不存在则创建
        if not file_dir_obj.exists():
            self.logger.info(f"输出目录{file_dir_obj.absolute()}不存在，开始创建...")
            file_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, file_dir_obj


if __name__ == "__main__":
    # 激活日志
    setup_logging()

    init_state = {
        "pdf_path": r"D:\doc\联想海豚用户手册.pdf",
        "file_dir": r"D:\output"
    }
    node_pdf_to_md = NodePDFToMD()
    # 使用 对象() 的方式相当于调用了 对象的__call__()
    result = node_pdf_to_md(init_state)

    # 将返回的图状态进行json序列化
    json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logging.getLogger().info(json_state)
