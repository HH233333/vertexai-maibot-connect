import base64
import os
import sys

# 将插件目录加入 sys.path
plugin_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if plugin_root not in sys.path:
    sys.path.insert(0, plugin_root)

from vertexai_maibot.converter import (
    build_embed_content_config,
    build_generate_content_config,
    build_thinking_config,
    convert_context_items_to_contents,
    convert_tools_to_gemini,
    parse_embedding_response,
    parse_gemini_response_to_maibot,
)
from vertexai_maibot.provider import VertexAIProvider
import logging
from google.genai import types


def test_convert_context_items_text_and_image():
    sample_items = [
        {
            "item_type": "SystemMessageItem",
            "parts": [{"type": "text", "text": "你是一个助手"}],
        },
        {
            "item_type": "UserMessageItem",
            "parts": [
                {"type": "text", "text": "看图"},
                {
                    "type": "image",
                    "image_format": "png",
                    "image_base64": base64.b64encode(b"fake_image_bytes").decode("ascii"),
                },
            ],
        },
        {
            "item_type": "AssistantMessageItem",
            "parts": [{"type": "text", "text": "已收到图片"}],
        },
    ]

    contents, sys_inst = convert_context_items_to_contents(sample_items)
    assert sys_inst == "你是一个助手"
    assert len(contents) == 2
    assert contents[0].role == "user"
    assert len(contents[0].parts) == 2
    assert contents[0].parts[0].text == "看图"
    assert contents[0].parts[1].inline_data is not None
    assert contents[0].parts[1].inline_data.mime_type == "image/png"
    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "已收到图片"


def test_convert_tools():
    tool_options = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询天气",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    tools = convert_tools_to_gemini(tool_options)
    assert len(tools) == 1
    assert len(tools[0].function_declarations) == 1
    decl = tools[0].function_declarations[0]
    assert decl.name == "get_weather"
    assert decl.description == "查询天气"
    assert decl.parameters_json_schema["required"] == ["city"]


def test_build_config():
    request = {
        "temperature": 0.5,
        "max_tokens": 2048,
        "extra_params": {"thinking_budget": 1024},
    }
    config = build_generate_content_config(request, system_instruction="系统提示")
    assert config.system_instruction == "系统提示"
    assert config.temperature == 0.5
    assert config.max_output_tokens == 2048
    assert config.thinking_config.include_thoughts is True
    assert config.thinking_config.thinking_budget == 1024


def test_parse_gemini_response():
    # 模拟一个 Gemini 响应
    candidate = types.Candidate(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="思考过程...", thought=True),
                types.Part(text="你好，我是麦麦。"),
                types.Part(
                    function_call=types.FunctionCall(
                        name="query_db", args={"sql": "SELECT 1"}
                    )
                ),
            ],
        ),
    )
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10,
        candidates_token_count=20,
        total_token_count=30,
        cached_content_token_count=5,
    )
    resp = types.GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=usage,
    )

    parsed = parse_gemini_response_to_maibot(resp, logical_turn_id="turn_123")
    assert parsed["content"] == "你好，我是麦麦。"
    assert parsed["reasoning_content"] == "思考过程..."
    assert parsed["usage"]["prompt_tokens"] == 10
    assert parsed["usage"]["completion_tokens"] == 20
    assert parsed["usage"]["total_tokens"] == 30
    assert parsed["usage"]["prompt_cache_hit_tokens"] == 5
    assert len(parsed["output_items"]) == 3
    assert parsed["output_items"][0]["item_type"] == "ReasoningItem"
    assert parsed["output_items"][1]["item_type"] == "AssistantMessageItem"
    assert parsed["output_items"][2]["item_type"] == "FunctionCallItem"
    assert parsed["output_items"][2]["tool_call"]["func_name"] == "query_db"


def test_build_config_thinking_level():
    request = {
        "extra_params": {"thinking_level": "low"},
    }
    config = build_generate_content_config(request)
    assert config.thinking_config.include_thoughts is True
    assert str(config.thinking_config.thinking_level).upper().endswith("LOW")


def test_parse_gemini_empty_response():
    import pytest

    # 1. 正常 STOP 静默结束
    stop_resp = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[]), finish_reason="STOP")]
    )
    parsed_stop = parse_gemini_response_to_maibot(stop_resp)
    assert parsed_stop["content"] == ""
    assert parsed_stop["output_items"] == []

    # 2. 异常拦截（如 SAFETY）应抛出包含 finish_reason 的异常
    safety_resp = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[]), finish_reason="SAFETY")]
    )
    with pytest.raises(RuntimeError, match="SAFETY"):
        parse_gemini_response_to_maibot(safety_resp)


def test_convert_parallel_function_calls_and_responses():
    """测试多工具（并行工具）调用的模型轮次和响应轮次均正确聚合。"""
    sample_items = [
        {
            "item_type": "UserMessageItem",
            "parts": [{"type": "text", "text": "查下小明和小红的信息"}],
        },
        {
            "item_type": "FunctionCallItem",
            "tool_call": {
                "call_id": "call_1",
                "func_name": "query_person",
                "args": {"name": "小明"},
            },
        },
        {
            "item_type": "FunctionCallItem",
            "tool_call": {
                "call_id": "call_2",
                "func_name": "query_person",
                "args": {"name": "小红"},
            },
        },
        {
            "item_type": "FunctionCallOutputItem",
            "call_id": "call_1",
            "tool_name": "query_person",
            "output": "小明资料",
            "success": True,
        },
        {
            "item_type": "FunctionCallOutputItem",
            "call_id": "call_2",
            "tool_name": "",  # 测试通过 call_id 回退查找 tool_name
            "output": "小红资料",
            "success": True,
        },
        {
            "item_type": "UserMessageItem",
            "parts": [{"type": "text", "text": "请总结"}],
        },
    ]

    contents, _ = convert_context_items_to_contents(sample_items)
    # 期望轮次结构：
    # 1. user: "查下小明和小红的信息"
    # 2. model: 包含 2 个 function_call (call_1, call_2)
    # 3. user: 包含 2 个 function_response (对应 call_1, call_2)
    # 4. user: "请总结"
    assert len(contents) == 4

    # 验证 model 轮次聚合
    model_turn = contents[1]
    assert model_turn.role == "model"
    assert len(model_turn.parts) == 2
    assert model_turn.parts[0].function_call.name == "query_person"
    assert model_turn.parts[1].function_call.name == "query_person"

    # 验证 response 轮次聚合
    resp_turn = contents[2]
    assert resp_turn.role == "user"
    assert len(resp_turn.parts) == 2
    assert resp_turn.parts[0].function_response.name == "query_person"
    assert resp_turn.parts[0].function_response.id == "call_1"
    assert resp_turn.parts[0].function_response.response == {"output": "小明资料"}
    assert resp_turn.parts[1].function_response.name == "query_person"
    assert resp_turn.parts[1].function_response.id == "call_2"
    assert resp_turn.parts[1].function_response.response == {"output": "小红资料"}

    # 验证后续 user 轮次
    next_user_turn = contents[3]
    assert next_user_turn.role == "user"
    assert len(next_user_turn.parts) == 1
    assert next_user_turn.parts[0].text == "请总结"


def test_thinking_enable_lowest_per_generation():
    """测试 enable_thinking=True 时，不同世代自动选用各自可用的最低思考配置。"""
    # 1. Gemini 4 -> LOW（不可用 MINIMAL）
    cfg4 = build_thinking_config("gemini-4-argon", {"enable_thinking": True})
    assert str(cfg4.thinking_level).upper().endswith("LOW")
    assert cfg4.thinking_budget is None
    assert cfg4.include_thoughts is True

    # 2. Gemini 3 Pro -> LOW
    cfg3_pro = build_thinking_config("gemini-3.1-pro", {"enable_thinking": True})
    assert str(cfg3_pro.thinking_level).upper().endswith("LOW")
    assert cfg3_pro.thinking_budget is None

    # 3. Gemini 3 Flash -> MINIMAL
    cfg3_flash = build_thinking_config("gemini-3.8-flash", {"enable_thinking": True})
    assert str(cfg3_flash.thinking_level).upper().endswith("MINIMAL")
    assert cfg3_flash.thinking_budget is None

    # 4. Gemini 2.5 Flash / Pro -> 128（最低正数预算）
    cfg2_flash = build_thinking_config("gemini-2.5-flash", {"enable_thinking": True})
    assert cfg2_flash.thinking_budget == 128
    assert cfg2_flash.thinking_level is None

    cfg2_pro = build_thinking_config("gemini-2.5-pro", {"enable_thinking": True})
    assert cfg2_pro.thinking_budget == 128
    assert cfg2_pro.thinking_level is None


def test_thinking_warnings_and_safety_fallbacks(caplog):
    """测试配置冲突、跨代错配及非法级别时输出 WARNING 日志并自动安全兜底。"""
    caplog.set_level(logging.WARNING, logger="VertexAI-Converter")

    # 1. Gemini 4 传入 MINIMAL -> 告警并兜底为 LOW
    caplog.clear()
    cfg_g4_min = build_thinking_config("gemini-4-hydrogen", {"thinking_level": "MINIMAL"})
    assert str(cfg_g4_min.thinking_level).upper().endswith("LOW")
    assert any(r.levelno >= logging.WARNING and "MINIMAL" in r.message for r in caplog.records)

    # 2. 同时传 level 与 budget -> 告警并按世代互斥保留
    caplog.clear()
    cfg_both_g3 = build_thinking_config(
        "gemini-3.8-flash",
        {"thinking_level": "HIGH", "thinking_budget": 1024},
    )
    assert str(cfg_both_g3.thinking_level).upper().endswith("HIGH")
    assert cfg_both_g3.thinking_budget is None
    assert any(r.levelno >= logging.WARNING and "400" in r.message for r in caplog.records)

    caplog.clear()
    cfg_both_g2 = build_thinking_config(
        "gemini-2.5-flash",
        {"thinking_level": "HIGH", "thinking_budget": 2048},
    )
    assert cfg_both_g2.thinking_budget == 2048
    assert cfg_both_g2.thinking_level is None
    assert any(r.levelno >= logging.WARNING and "400" in r.message for r in caplog.records)

    # 3. Gemini 2.5 Pro 尝试关闭或设为 0 -> 告警并兜底为 128
    caplog.clear()
    cfg_25pro_zero = build_thinking_config("gemini-2.5-pro", {"thinking_budget": 0})
    assert cfg_25pro_zero.thinking_budget == 128
    assert any(r.levelno >= logging.WARNING and "128" in r.message for r in caplog.records)

    caplog.clear()
    cfg_25pro_off = build_thinking_config("gemini-2.5-pro", {"enable_thinking": False})
    assert cfg_25pro_off.thinking_budget == 128
    assert cfg_25pro_off.include_thoughts is False
    assert any(r.levelno >= logging.WARNING for r in caplog.records)

    # 4. Gemini 2.5 Flash 关闭思考 -> 正常设为 0
    caplog.clear()
    cfg_25flash_off = build_thinking_config("gemini-2.5-flash", {"enable_thinking": False})
    assert cfg_25flash_off.thinking_budget == 0
    assert cfg_25flash_off.include_thoughts is False

    # 5. 跨代参数映射并告警
    caplog.clear()
    cfg_cross_3 = build_thinking_config("gemini-3.8-flash", {"thinking_budget": 4096})
    assert str(cfg_cross_3.thinking_level).upper().endswith("MEDIUM")
    assert cfg_cross_3.thinking_budget is None
    assert any(r.levelno >= logging.WARNING for r in caplog.records)

    caplog.clear()
    cfg_cross_2 = build_thinking_config("gemini-2.5-flash", {"thinking_level": "LOW"})
    assert cfg_cross_2.thinking_budget == 1024
    assert cfg_cross_2.thinking_level is None
    assert any(r.levelno >= logging.WARNING for r in caplog.records)

    # 6. 已下线的老一代模型（Gemini 1.5 / 2.0）直接拒绝并提示升级
    import pytest

    with pytest.raises(ValueError, match="老一代已下线模型"):
        build_thinking_config("gemini-1.5-pro", {"thinking_level": "LOW"})
    with pytest.raises(ValueError, match="老一代已下线模型"):
        build_thinking_config("gemini-2.0-flash", {})


def test_embedding_config_gemini_embedding_2_only(caplog):
    """测试仅支持 Gemini Embedding 2，拒绝老模型，告警忽略 task_type，映射 dimensions。"""
    import pytest

    caplog.set_level(logging.WARNING, logger="VertexAI-Converter")

    # 1. 拒绝老一代嵌入模型
    for legacy_model in (
        "gemini-embedding-001",
        "text-embedding-004",
        "text-embedding-005",
        "text-multilingual-embedding-002",
        "multimodalembedding@001",
    ):
        with pytest.raises(ValueError, match="老一代嵌入模型"):
            build_embed_content_config(legacy_model, {})

    # 2. Gemini Embedding 2：忽略 task_type / task / title 并打 WARNING，正确映射 dimensions 与 auto_truncate
    caplog.clear()
    cfg = build_embed_content_config(
        "gemini-embedding-2-preview",
        {
            "dimensions": 1536,
            "task_type": "RETRIEVAL_DOCUMENT",
            "title": "legacy_title",
            "auto_truncate": True,
        },
    )
    assert cfg is not None
    assert cfg.output_dimensionality == 1536
    assert cfg.auto_truncate is True
    assert cfg.task_type is None
    assert cfg.title is None
    assert any(r.levelno >= logging.WARNING and "task_type" in r.message for r in caplog.records)

    # 3. 无额外参数时返回 None
    assert build_embed_content_config("gemini-embedding-2", {}) is None

    # 4. dimensions 与 output_dimensionality 冲突应报错
    with pytest.raises(ValueError, match="冲突"):
        build_embed_content_config(
            "gemini-embedding-2",
            {"dimensions": 768, "output_dimensionality": 1536},
        )


def test_parse_embedding_response_and_image_embedding_provider():
    """测试 parse_embedding_response 与 VertexAIProvider.handle_image_embedding 全流程。"""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    fake_resp = types.EmbedContentResponse(
        embeddings=[
            types.ContentEmbedding(
                values=[0.1, 0.2, 0.3],
                statistics=types.ContentEmbeddingStatistics(token_count=42.0, truncated=False),
            )
        ]
    )
    parsed = parse_embedding_response(fake_resp)
    assert parsed["embedding"] == [0.1, 0.2, 0.3]
    assert parsed["status"] == "completed"
    assert parsed["usage"]["prompt_tokens"] == 42
    assert parsed["usage"]["total_tokens"] == 42

    # 模拟异步调用 handle_image_embedding
    mock_client = MagicMock()
    mock_embed_content = AsyncMock(return_value=fake_resp)
    mock_client.aio.models.embed_content = mock_embed_content

    mock_manager = MagicMock()
    mock_manager.get_client.return_value = (mock_client, None)

    provider = VertexAIProvider(mock_manager)
    img_b64 = base64.b64encode(b"fake_png_data").decode("ascii")
    result = asyncio.run(
        provider.handle_image_embedding(
            {
                "image_base64": img_b64,
                "mime_type": "image/png",
                "model_info": {
                    "model_identifier": "gemini-embedding-2-preview",
                    "extra_params": {"dimensions": 1536},
                },
                "extra_params": {"dimensions": 1536, "task_type": "SEMANTIC_SIMILARITY"},
            }
        )
    )
    assert result["embedding"] == [0.1, 0.2, 0.3]
    assert result["usage"]["prompt_tokens"] == 42

    call_kwargs = mock_embed_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-embedding-2-preview"
    assert len(call_kwargs["contents"]) == 1
    part = call_kwargs["contents"][0]
    assert part.inline_data.data == b"fake_png_data"
    assert part.inline_data.mime_type == "image/png"
    assert call_kwargs["config"].output_dimensionality == 1536
    assert call_kwargs["config"].task_type is None

