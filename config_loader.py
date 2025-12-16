#!/usr/bin/env python3
"""
简单的配置加载器，用于读取 newsletter.yml 配置文件
"""

import yaml
from pathlib import Path


class Config:
    """配置管理器"""

    def __init__(self, config_file="newsletter.yml"):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self):
        """从 YAML 文件加载配置"""
        config_path = Path(self.config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件 {self.config_file} 不存在")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"解析 YAML 配置失败: {e}")

    def get_qwen_config(self):
        """获取 Qwen API 配置"""
        return self.config.get('qwen', {})

    def reload(self):
        """重新加载配置文件"""
        self.config = self._load_config()


# 全局配置实例
try:
    config = Config()
except Exception as e:
    print(f"❌ 加载配置失败: {e}")
    config = None
