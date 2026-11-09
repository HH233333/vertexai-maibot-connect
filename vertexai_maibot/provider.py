from google.genai import types
from typing import Any, Dict, List, Optional

import base64
import logging

from .client_manager import VertexClientManager
from .converter import (
    build_generate_content_config,
    convert_context_items_to_contents,
    parse_gemini_response_to_maibot,
)

logger = logging.getLogger("VertexAI-Provider")


class VertexAIProvider:
    """Vertex AI 操作分发器，处理 response、audio_transcription、embedding。"""

    def __init__(self, client_manager: VertexClientManager) -> None:
        self.client_manager = client_manager

    def _resolve_model_name(self, model_info: Optional[Dict[str, Any]]) -> str:
        if not model_info:
            return "gemini-2.5-flash"
        identifier = str(model_info.get("model_identifier") or model_info.get("name") or "").strip()
        if not identifier:
            identifier = "gemini-2.5-flash"
        # 剥离前缀 publishers/google/models/
        if "/" in identifier:
            identifier = identifier.split("/")[-1]
        return identifier

    async def handle_response(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理文本/多模态对话与规划请求。"""
        provider_data = request.get("api_provider") or {}
        client, _ = self.client_manager.get_client(provider_data)

        model_name = self._resolve_model_name(request.get("model_info"))
        context_items = request.get("context_items") or []
        contents, system_instruction = convert_context_items_to_contents(context_items)
        config = build_generate_content_config(request, system_instruction=system_instruction)

        logger.debug(
            "调用 Vertex AI 响应: model=%s, contents_len=%d, has_system=%s",
            model_name,
            len(contents),
            bool(system_instruction),
        )

        response: types.GenerateContentResponse = await client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        logical_turn_id = request.get("logical_turn_id")
        return parse_gemini_response_to_maibot(response, logical_turn_id=logical_turn_id)

    async def handle_audio_transcription(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理音频转写（语音转文字）请求。"""
        provider_data = request.get("api_provider") or {}
        client, _ = self.client_manager.get_client(provider_data)

        model_name = self._resolve_model_name(request.get("model_info"))
        audio_base64 = str(request.get("audio_base64") or "").strip()
        if not audio_base64:
            return {"content": "", "response": ""}

        audio_bytes = base64.b64decode(audio_base64)

        # 构造音频识别的多模态内容
        contents = [
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
            types.Part.from_text(text="请将这段音频中的语音内容准确无误地转录为文字，仅直接输出识别结果，不要包含任何多余的开场白、说明或问答。"),
        ]

        logger.info("调用 Vertex AI 进行音频转写: model=%s, audio_size=%d bytes", model_name, len(audio_bytes))
        response: types.GenerateContentResponse = await client.aio.models.generate_content(
            model=model_name,
            contents=contents,
        )

        text = (response.text or "").strip()
        return {
            "content": text,
            "response": text,
            "status": "completed",
        }

    async def handle_embedding(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理文本嵌入请求。"""
        provider_data = request.get("api_provider") or {}
        client, _ = self.client_manager.get_client(provider_data)

        model_name = self._resolve_model_name(request.get("model_info"))
        embedding_input = request.get("embedding_input")

        contents: List[str]
        if isinstance(embedding_input, list):
            contents = [str(item) for item in embedding_input]
        else:
            contents = [str(embedding_input or "")]

        logger.debug("调用 Vertex AI 文本嵌入: model=%s, count=%d", model_name, len(contents))
        response = await client.aio.models.embed_content(
            model=model_name,
            contents=contents if len(contents) > 1 else contents[0],
        )

        raw_embeddings = getattr(response, "embeddings", None)
        values: List[float] = []
        if raw_embeddings:
            first = raw_embeddings[0]
            values = getattr(first, "values", []) or []
        elif hasattr(response, "embedding"):
            values = getattr(response.embedding, "values", []) or []

        return {
            "embedding": values,
            "status": "completed",
        }
