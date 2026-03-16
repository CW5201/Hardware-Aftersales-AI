# processor/query_processor/base.py

from abc import ABC, abstractmethod
from typing import TypeVar

from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.task_utils import add_running_task, add_done_task

# 定义泛型
T = TypeVar("T")

class NodeBase(ABC):

    # 节点名称：占位符
    # 子类中要覆盖这个名字
    name: str = "base_node"

    def __call__(self, state: QueryGraphState) -> QueryGraphState:

        try:
            logger.info(f"{self.name} 开始执行")

            # 开始：记录节点运行状态
            add_running_task(state['session_id'], self.name, state.get("is_stream"))

            state = self.process(state)

            # 此处加任务追踪，多路搜索节点无法获取session_id，因此需要在process内部处理
            # add_done_task(state.get("session_id"), self.name, state.get("is_stream"))

            logger.info(f"{self.name} 结束执行")

        except Exception as e:
            logger.exception(f"{self.name} 执行异常: {e}")
            raise

        return state

    @abstractmethod
    def process(self, state: T) -> T:
        pass