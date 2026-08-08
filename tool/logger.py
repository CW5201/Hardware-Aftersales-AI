# 默认名字是：root
import logging
import os

import colorlog

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

logger = logging.getLogger()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))


def get_logger(name):
    """优化日志配置"""
    return logging.getLogger(name)

# 加载彩色日志处理器
handler = colorlog.StreamHandler()
# 定义日志输出的格式
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(filename)s : %(lineno)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',  # INFO 显示为绿色
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

# logger.handlers.clear()

# 应用日志配置信息
logger.addHandler(handler)
