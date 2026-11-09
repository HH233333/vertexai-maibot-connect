from google import genai
from google.genai.types import HttpOptions
from typing import Any, Dict, Optional, Tuple

import logging

from .auth import VertexAuthInfo, resolve_vertex_auth

logger = logging.getLogger("VertexAI-ClientManager")


def _build_http_options(provider_data: Dict[str, Any]) -> HttpOptions:
    """根据 Provider 数据构建 Gemini SDK 的 HttpOptions。"""
    http_options_payload: Dict[str, Any] = {}

    timeout = provider_data.get("timeout")
    if timeout is not None:
        try:
            http_options_payload["timeout"] = int(float(timeout) * 1000)
        except (TypeError, ValueError):
            pass

    base_url = str(provider_data.get("base_url") or "").strip()
    if base_url:
        normalized_base_url = base_url.rstrip("/")
        version_candidate = normalized_base_url.rsplit("/", 1)
        if len(version_candidate) == 2 and version_candidate[1].startswith("v"):
            http_options_payload["base_url"] = f"{version_candidate[0]}/"
            http_options_payload["api_version"] = version_candidate[1]
        else:
            http_options_payload["base_url"] = f"{normalized_base_url}/"

    return HttpOptions(**http_options_payload)


class VertexClientManager:
    """Vertex AI 客户端实例管理与缓存。"""

    def __init__(self, default_project: str = "", default_location: str = "global") -> None:
        self.default_project = default_project
        self.default_location = default_location
        self._clients: Dict[str, genai.Client] = {}

    def _get_cache_key(self, auth_info: VertexAuthInfo, base_url: str) -> str:
        cred_sig = ""
        if auth_info.api_key:
            cred_sig = f"key:{auth_info.api_key[:6]}"
        elif auth_info.credentials:
            cred_sig = f"cred:{id(auth_info.credentials)}"
        return f"{auth_info.project}:{auth_info.location}:{cred_sig}:{base_url}"

    def get_client(self, provider_data: Optional[Dict[str, Any]] = None) -> Tuple[genai.Client, VertexAuthInfo]:
        """获取或创建与当前请求匹配的 genai.Client 实例。"""
        provider = provider_data or {}
        auth_info = resolve_vertex_auth(
            provider_data=provider,
            default_project=self.default_project,
            default_location=self.default_location,
        )

        base_url = str(provider.get("base_url") or "").strip()
        cache_key = self._get_cache_key(auth_info, base_url)

        if cache_key not in self._clients:
            logger.info(
                "初始化 Vertex AI 客户端: project=%s, location=%s, auth_mode=%s",
                auth_info.project,
                auth_info.location,
                "api_key" if auth_info.api_key else "oauth2_credentials",
            )
            client = genai.Client(
                vertexai=True,
                project=auth_info.project,
                location=auth_info.location,
                credentials=auth_info.credentials,
                api_key=auth_info.api_key,
                http_options=_build_http_options(provider),
            )
            self._clients[cache_key] = client

        return self._clients[cache_key], auth_info

    def clear(self) -> None:
        """清空客户端缓存。"""
        self._clients.clear()
