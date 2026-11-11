import base64
import os
import sys

# 将插件目录加入 sys.path
plugin_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if plugin_root not in sys.path:
    sys.path.insert(0, plugin_root)

from vertexai_maibot.converter import (
    build_generate_content_config,
    convert_context_items_to_contents,
    convert_tools_to_gemini,
    parse_gemini_response_to_maibot,
)
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

