"""
配置管理模块
- 持久化 API 设置到 settings.json
- API Key 掩码处理
- 在线/离线模式管理
"""
import os
import json
import base64

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_SETTINGS = {
    "api_key": "",
    "base_url": "https://api.deepseek.com",
    "model_name": "deepseek-chat",
    "api_enabled": False,
    "top_k": 5,
}


def _simple_crypt(text: str, encrypt: bool = True) -> str:
    """
    对 API Key 做简单的 base64 编码/解码存储。
    注意：这不是真正的加密，仅防止明文泄露。
    更安全的方式应使用 keyring 库，但这里保持轻量。
    """
    if not text:
        return ""
    if encrypt:
        return base64.b64encode(text.encode("utf-8")).decode("utf-8")
    else:
        try:
            return base64.b64decode(text.encode("utf-8")).decode("utf-8")
        except Exception:
            return text  # 兼容旧版本明文数据


class Settings:
    """应用配置单例"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = None
            cls._instance._load()
        return cls._instance

    def _load(self):
        """从文件加载配置"""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                # 解码 api_key
                if self._data.get("api_key"):
                    self._data["api_key"] = _simple_crypt(self._data["api_key"], encrypt=False)
            except Exception:
                self._data = dict(DEFAULT_SETTINGS)
        else:
            self._data = dict(DEFAULT_SETTINGS)

    def _save(self):
        """保存配置到文件（加密 api_key）"""
        data_to_save = dict(self._data)
        if data_to_save.get("api_key"):
            data_to_save["api_key"] = _simple_crypt(data_to_save["api_key"], encrypt=True)
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # --- 属性访问 ---

    @property
    def api_key(self) -> str:
        return self._data.get("api_key", "")

    @api_key.setter
    def api_key(self, value: str):
        self._data["api_key"] = value
        self._save()

    @property
    def base_url(self) -> str:
        return self._data.get("base_url", DEFAULT_SETTINGS["base_url"])

    @base_url.setter
    def base_url(self, value: str):
        self._data["base_url"] = value
        self._save()

    @property
    def model_name(self) -> str:
        return self._data.get("model_name", DEFAULT_SETTINGS["model_name"])

    @model_name.setter
    def model_name(self, value: str):
        self._data["model_name"] = value
        self._save()

    @property
    def api_enabled(self) -> bool:
        return self._data.get("api_enabled", DEFAULT_SETTINGS["api_enabled"])

    @api_enabled.setter
    def api_enabled(self, value: bool):
        self._data["api_enabled"] = value
        self._save()

    @property
    def top_k(self) -> int:
        return self._data.get("top_k", DEFAULT_SETTINGS["top_k"])

    @top_k.setter
    def top_k(self, value: int):
        self._data["top_k"] = value
        self._save()

    def mask_key(self) -> str:
        """返回掩码后的 API Key，如 sk-****abcd"""
        key = self.api_key
        if not key:
            return ""
        if len(key) <= 8:
            return key[:2] + "****" + key[-2:]
        return key[:3] + "****" + key[-4:]

    def to_dict(self) -> dict:
        """返回展示用配置（key 已掩码）"""
        return {
            "api_key": self.mask_key(),
            "base_url": self.base_url,
            "model_name": self.model_name,
            "api_enabled": self.api_enabled,
            "top_k": self.top_k,
        }
