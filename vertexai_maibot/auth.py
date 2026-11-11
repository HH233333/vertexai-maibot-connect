from dataclasses import dataclass
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from typing import Any, Dict, Optional, Tuple

import google.auth
import json
import logging
import os
import re
import shutil
import subprocess

logger = logging.getLogger("VertexAI-Auth")


@dataclass(frozen=True, slots=True)
class VertexAuthInfo:
    """Vertex AI 认证与端点调度信息。"""

    project: str
    """Google Cloud 项目 ID。"""

    location: str
    """Vertex AI 区域，如 global、us-central1 等。"""

    credentials: Optional[Credentials | service_account.Credentials] = None
    """OAuth2 凭据对象（支持 Service Account、ADC 或 Bearer Token）。"""

    api_key: Optional[str] = None
    """Vertex Express Mode 或 Google Cloud API Key。"""


def _detect_gcloud_project() -> Optional[str]:
    """尝试通过本地 gcloud CLI 检测当前默认的 GCP Project ID。"""
    gcloud_bin = shutil.which("gcloud.cmd") if os.name == "nt" else shutil.which("gcloud")
    if not gcloud_bin:
        return None
    try:
        cmd = [gcloud_bin, "config", "get-value", "project"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
        if res.returncode == 0:
            project = res.stdout.strip()
            if project and "(unset)" not in project and not project.startswith("ERROR"):
                return project
    except Exception:
        pass
    return None


def _get_token_from_gcloud() -> Tuple[Optional[str], Optional[str]]:
    """尝试通过本地 gcloud CLI 获取 Access Token 及当前 Project ID。"""
    gcloud_bin = shutil.which("gcloud.cmd") if os.name == "nt" else shutil.which("gcloud")
    if not gcloud_bin:
        return None, None
    try:
        cmd = [gcloud_bin, "auth", "print-access-token"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        if res.returncode == 0:
            token = res.stdout.strip()
            if token and not token.startswith("ERROR"):
                return token, _detect_gcloud_project()
    except Exception:
        pass
    return None, None


def _extract_case_insensitive(data: Optional[Dict[str, Any]], *keys: str) -> Optional[str]:
    """从字典中不区分大小写、忽略下划线/中划线提取键值。"""
    if not data or not isinstance(data, dict):
        return None
    normalized_keys = {k.lower().replace("-", "").replace("_", "") for k in keys}
    for k, v in data.items():
        if str(k).lower().replace("-", "").replace("_", "") in normalized_keys and v is not None:
            val = str(v).strip()
            if val:
                return val
    return None


def resolve_vertex_auth(
    provider_data: Optional[Dict[str, Any]] = None,
    default_project: str = "",
    default_location: str = "global",
) -> VertexAuthInfo:
    """解析 Vertex AI 的认证凭据、Project ID 与 Location。

    支持以下认证模式自动解析与降级：
    1. Direct Bearer Access Token (以 ya29. 开头)
    2. Vertex Express Mode / GCP API Key (以 AIzaSy 或 AQ 开头)
    3. Service Account JSON 密钥文件路径 (在 api_key 或 default_headers['credentials_file'] 中指定)
    4. Service Account JSON 内容字符串 (直接填入 api_key)
    5. 系统环境变量 GOOGLE_APPLICATION_CREDENTIALS 指定的凭据文件
    6. Application Default Credentials (ADC) 本地默认凭据
    7. 本地 gcloud CLI 自动登录凭据与项目探测

    Args:
        provider_data: 来自 MaiBot 请求上下文或配置的 API Provider 字典。
        default_project: 插件默认预设的 GCP Project ID。
        default_location: 插件默认预设的 Vertex Location。

    Returns:
        VertexAuthInfo: 解析完成的认证信息。

    Raises:
        ValueError: 当未能获取有效凭据或未能识别 Project ID 时抛出。
    """
    provider = provider_data or {}
    raw_key = str(provider.get("api_key") or "").strip()
    headers = provider.get("default_headers") or {}
    query = provider.get("default_query") or {}

    # 1. 解析 project：支持直接配置、默认请求头、默认查询参数、插件默认值或环境变量
    configured_project = str(provider.get("project") or "").strip()
    if not configured_project:
        configured_project = _extract_case_insensitive(
            headers,
            "project", "project_id", "projectid", "x-goog-user-project", "x-goog-project-id",
        ) or ""
    if not configured_project:
        configured_project = _extract_case_insensitive(
            query,
            "project", "project_id", "projectid",
        ) or ""
    if not configured_project and default_project:
        configured_project = default_project.strip()

    project: Optional[str] = configured_project if configured_project else None

    # 2. 解析 location：优先直接配置（provider、headers、query），其次从 base_url 推导，最后降级到 default_location / global
    location = str(provider.get("location") or "").strip()
    if not location:
        location = _extract_case_insensitive(headers, "location", "region") or ""
    if not location:
        location = _extract_case_insensitive(query, "location", "region") or ""
    if not location:
        base_url = str(provider.get("base_url") or "").strip().lower()
        url_match = re.search(r"https?://(?:([a-z0-9-]+)-aiplatform|aiplatform\.([a-z0-9-]+))\.", base_url)
        if url_match:
            candidate_loc = url_match.group(1) or url_match.group(2)
            if candidate_loc and candidate_loc not in {"googleapis", "rep"}:
                location = candidate_loc
    if not location:
        location = default_location.strip() or os.getenv("VERTEX_LOCATION", "global").strip()
    location = location or "global"

    # 1. 直接提供 OAuth2 Bearer Access Token (通常以 ya29. 开头)
    if raw_key.startswith("ya29."):
        resolved_proj = (
            project
            or _detect_gcloud_project()
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCP_PROJECT")
        )
        if not resolved_proj:
            raise ValueError(
                "检测到 Bearer Token，但未能识别 Google Cloud Project ID。请在 Provider 配置中填入 GCP 项目 ID。"
            )
        return VertexAuthInfo(
            project=resolved_proj,
            location=location,
            credentials=Credentials(token=raw_key),
        )

    # 2. 传入了 Vertex Express Mode 或 GCP API Key (通常以 AIzaSy 或 AQ 开头)
    if raw_key.startswith("AIzaSy") or raw_key.startswith("AQ"):
        resolved_proj = (
            project
            or _detect_gcloud_project()
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCP_PROJECT")
        )
        if not resolved_proj:
            raise ValueError(
                "检测到 Vertex API Key，但未能识别 Google Cloud Project ID。请在 Provider 配置中填入 GCP 项目 ID。"
            )
        return VertexAuthInfo(
            project=resolved_proj,
            location=location,
            api_key=raw_key,
        )

    # 3. 指定了服务账号 JSON 密钥文件路径
    file_candidate = raw_key
    if not file_candidate or not os.path.isfile(file_candidate):
        file_candidate = _extract_case_insensitive(
            headers,
            "credentials_file", "service_account_file", "credentials", "sa_file",
        ) or ""
    if not file_candidate or not os.path.isfile(file_candidate):
        file_candidate = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

    if file_candidate and os.path.isfile(file_candidate):
        try:
            with open(file_candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            sa_proj = data.get("project_id")
            creds = service_account.Credentials.from_service_account_file(
                file_candidate,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            resolved_proj = project or sa_proj or _detect_gcloud_project()
            if not resolved_proj:
                raise ValueError(
                    f"服务账号文件 {file_candidate} 中未包含 project_id，且未在配置中指定 project。"
                )
            return VertexAuthInfo(
                project=resolved_proj,
                location=location,
                credentials=creds,
            )
        except Exception as e:
            logger.error("解析 Vertex AI 服务账号凭据文件 %s 失败: %s", file_candidate, e)
            raise ValueError(f"解析 Vertex AI 服务账号文件失败: {e}") from e

    # 4. 直接传入了服务账号 JSON 字符串
    if raw_key.startswith("{") and ("client_email" in raw_key or "private_key" in raw_key):
        try:
            data = json.loads(raw_key)
            if "token_uri" not in data:
                data["token_uri"] = "https://oauth2.googleapis.com/token"
            sa_proj = data.get("project_id")
            creds = service_account.Credentials.from_service_account_info(
                data,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            resolved_proj = project or sa_proj or _detect_gcloud_project()
            if not resolved_proj:
                raise ValueError("服务账号 JSON 中未包含 project_id，且未在配置中指定 project。")
            return VertexAuthInfo(
                project=resolved_proj,
                location=location,
                credentials=creds,
            )
        except Exception as e:
            logger.error("解析 Vertex AI 服务账号 JSON 内容失败: %s", e)
            raise ValueError(f"解析 Vertex AI 服务账号 JSON 内容失败: {e}") from e

    # 5. 尝试通过 google.auth 读取本地 ADC (Application Default Credentials)
    try:
        adc_creds, adc_proj = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        resolved_proj = project or adc_proj or _detect_gcloud_project()
        if resolved_proj:
            return VertexAuthInfo(
                project=resolved_proj,
                location=location,
                credentials=adc_creds,
            )
    except (DefaultCredentialsError, Exception):
        pass

    # 6. 尝试通过本地 gcloud CLI 自动获取 Access Token
    gc_token, gc_proj = _get_token_from_gcloud()
    if gc_token:
        resolved_proj = project or gc_proj or _detect_gcloud_project()
        if resolved_proj:
            return VertexAuthInfo(
                project=resolved_proj,
                location=location,
                credentials=Credentials(token=gc_token),
            )

    raise ValueError(
        "未能解析到有效的 Vertex AI 认证凭据。请在 API Key 中配置 Service Account JSON 密钥文件路径或内容、"
        "输入 Bearer Access Token (ya29...)、输入 Vertex Express API Key (AIzaSy...)，"
        "或在本地终端通过 'gcloud auth login' 登录 / 配置系统 ADC。"
    )
