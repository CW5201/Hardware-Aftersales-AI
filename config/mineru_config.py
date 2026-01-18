from dataclasses import dataclass
import os
from dotenv import load_dotenv

#1.加载配置文件
load_dotenv()

@dataclass
class MinerUConfig:

    base_url_prod: str
    api_token_prod: str
    base_url_test: str
    api_token_test: str

mineru_config = MinerUConfig(
    base_url_prod=os.getenv("MINERU_BASE_URL", ""),
    api_token_prod=os.getenv("MINERU_API_TOKEN", ""),
    base_url_test = "test",
    api_token_test = "test"
)




