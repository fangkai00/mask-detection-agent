# -*- coding: utf-8 -*-
"""
口罩检测智能体全局配置(从 config/config.yaml 加载)

设计:
- 配置数据存放在 config/config.yaml(human-editable,API key 填在顶部)。
- 本文件为加载器:读取 config.yaml → 注入为模块级大写属性,业务代码用
  `import config; config.XXX` 访问。
- 路径类配置(以 _DIR / _FILE 结尾)在 YAML 中用相对项目根的相对路径,
  加载时自动拼接 PROJECT_ROOT 解析为绝对路径。
"""
import os

import yaml

# 项目根目录(config/ 的父目录)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 读取并解析 YAML(位于 config/ 文件夹内)
_YAML_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
with open(_YAML_PATH, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

# 注入为模块级属性(config.DASHSCOPE_API_KEY 等可直接访问)
globals().update(_cfg)

# 路径解析:以 _DIR / _FILE 结尾的相对路径,拼接 PROJECT_ROOT 解析为绝对路径
for _k, _v in list(_cfg.items()):
    if isinstance(_v, str) and _k.endswith(("_DIR", "_FILE")) and not os.path.isabs(_v):
        globals()[_k] = os.path.join(PROJECT_ROOT, _v)

# 供 `from config import *` 使用
__all__ = list(_cfg.keys()) + ["PROJECT_ROOT"]


def reload_config():
    """重新读取 config.yaml 并刷新模块属性(运行期改了 YAML 后调用)。"""
    with open(_YAML_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    globals().update(cfg)
    for k, v in list(cfg.items()):
        if isinstance(v, str) and k.endswith(("_DIR", "_FILE")) and not os.path.isabs(v):
            globals()[k] = os.path.join(PROJECT_ROOT, v)
    globals()["_cfg"] = cfg
