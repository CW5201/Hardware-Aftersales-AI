import logging
from typing import Dict

from processor.import_processor.base import BaseNode, setup_logging


class NodeTest(BaseNode):

    name = "node_test"

    def process(self, state: Dict) -> Dict:


        self.logger.debug(f"{self.name} 正在debug")
        self.logger.info(f"{self.name} 正在info")
        self.logger.warning(f"{self.name} warning")
        self.logger.error(f"{self.name} error")
        self.logger.critical(f"{self.name} critical")

        self.log_step("PDF解析","文件上传")
        self.log_step("PDF解析","文件解析")
        self.log_step("PDF解析","文件下载")
        self.log_step("PDF解析","文件解压")

        return {}



if __name__ == "__main__":

    # 激活日志
    setup_logging(logging.DEBUG)

    test_node = NodeTest()
    # 使用 对象() 的方式相当于调用了 对象的__call__()
    test_node({"abc": 456})