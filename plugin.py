"""Vertex AI MaiBot Connect Plugin.

通过 Google Cloud Vertex AI 官方 SDK 接入通道，为 MaiBot 提供对话、视觉理解、音频转写及嵌入的插件化 Provider。
"""

from __future__ import annotations

from typing import Any

from maibot_sdk import Command, Field, LLMProvider, MaiBotPlugin, PluginConfigBase

if __package__:
    from .vertexai_maibot import __version__
    from .vertexai_maibot.auth import resolve_vertex_auth
    from .vertexai_maibot.client_manager import VertexClientManager
    from .vertexai_maibot.provider import VertexAIProvider
else:
    from vertexai_maibot import __version__
    from vertexai_maibot.auth import resolve_vertex_auth
    from vertexai_maibot.client_manager import VertexClientManager
    from vertexai_maibot.provider import VertexAIProvider

CLIENT_TYPE = "vertex_connect"


class PluginSectionConfig(PluginConfigBase):
    """Vertex AI 插件基础设置。"""

    __ui_label__ = "基础设置"
    __ui_icon__ = "settings"
    __ui_order__ = 0

    config_version: str = Field(default="0.1.0", description="配置结构版本")
    enabled: bool = Field(default=True, description="启用 Vertex AI Provider")
    default_project: str = Field(
        default="",
        description="默认 Google Cloud 项目 ID；若留空则优先从 model_config 的 api_provider 或本地环境/ADC/gcloud 读取",
    )
    default_location: str = Field(
        default="global",
        description="默认 Vertex AI 区域（如 global, us-central1 等）",
    )


class VertexPluginConfig(PluginConfigBase):
    """Vertex AI 插件根配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)


class VertexAIConnectPlugin(MaiBotPlugin):
    """Vertex AI 连接器插件。"""

    config_model = VertexPluginConfig

    async def on_load(self) -> None:
        self.client_manager = VertexClientManager(
            default_project=self.config.plugin.default_project,
            default_location=self.config.plugin.default_location,
        )
        self.provider = VertexAIProvider(self.client_manager)
        self.ctx.logger.info("Vertex AI Connect Plugin v%s 已成功加载", __version__)

    async def on_unload(self) -> None:
        if hasattr(self, "client_manager"):
            self.client_manager.clear()

    async def on_config_update(
        self, scope: str, config_data: dict[str, object], version: str
    ) -> None:
        del config_data, version
        if scope == "self" and hasattr(self, "client_manager"):
            self.client_manager.default_project = self.config.plugin.default_project
            self.client_manager.default_location = self.config.plugin.default_location
            self.client_manager.clear()
            self.ctx.logger.info("Vertex AI 插件配置已更新并刷新客户端缓存")

    @LLMProvider(
        CLIENT_TYPE,
        name="Vertex AI Connect",
        description="Google Cloud Vertex AI 官方 SDK 接入通道",
        version=__version__,
    )
    async def handle_llm_provider(self, operation: str, request: dict[str, Any]) -> dict[str, Any]:
        """处理来自 MaiBot Host 的统一 LLM Provider 调用。"""
        if not self.config.plugin.enabled:
            raise RuntimeError("Vertex AI Connect Provider 已在插件配置中禁用")

        if operation == "response":
            return await self.provider.handle_response(request)
        if operation == "audio_transcription":
            return await self.provider.handle_audio_transcription(request)
        if operation == "embedding":
            return await self.provider.handle_embedding(request)

        raise NotImplementedError(f"Vertex AI Provider 暂不支持操作: {operation}")

    @Command("vertex_status", description="查看当前 Vertex AI 凭据状态与探测结果", pattern=r"^/vertex_status$")
    async def status_command(self, **kwargs: Any) -> tuple[bool, str, int]:
        stream = kwargs.get("stream_id")
        try:
            auth_info = resolve_vertex_auth(
                provider_data={},
                default_project=self.config.plugin.default_project,
                default_location=self.config.plugin.default_location,
            )
            mode = "Vertex API Key" if auth_info.api_key else "OAuth2 / ADC / Service Account"
            text = (
                f"【Vertex AI 连接状态】\n"
                f"- Project ID: {auth_info.project}\n"
                f"- Location: {auth_info.location}\n"
                f"- 凭据模式: {mode}\n"
                f"- 插件启用状态: {'开启' if self.config.plugin.enabled else '关闭'}"
            )
        except Exception as e:
            text = f"【Vertex AI 连接状态】\n- 凭据解析失败: {e}"

        if stream:
            await self.ctx.send.text(text, stream)
        return True, "已返回状态", 2

    @Command("vertex_models", description="查看常用推荐的 Vertex AI 模型标识", pattern=r"^/vertex_models$")
    async def models_command(self, **kwargs: Any) -> tuple[bool, str, int]:
        stream = kwargs.get("stream_id")
        text = (
            "【常用 Vertex AI 模型标识】\n"
            "• 对话/规划/VLM: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash\n"
            "• 语音转写 (voice): gemini-2.5-flash, gemini-2.0-flash\n"
            "• 文本嵌入 (embedding): text-embedding-005, text-multilingual-embedding-002\n\n"
            "在 config/model_config.toml 中配置 api_providers.client_type = 'vertex_connect' 即可使用。"
        )
        if stream:
            await self.ctx.send.text(text, stream)
        return True, "已返回模型列表", 2


def create_plugin() -> VertexAIConnectPlugin:
    return VertexAIConnectPlugin()
