from google.genai import types
from typing import Any, Dict, List, Optional, Tuple

import base64
import json
import logging
import uuid

logger = logging.getLogger("VertexAI-Converter")

ITEM_SCHEMA_VERSION = 1


def _normalize_image_mime(image_format: str) -> str:
    fmt = (image_format or "jpeg").lower().strip()
    if fmt == "jpg":
        fmt = "jpeg"
    return f"image/{fmt}"


def convert_context_items_to_contents(
    context_items: List[Dict[str, Any]],
) -> Tuple[List[types.Content], Optional[str]]:
    """将 MaiBot 序列化的 context_items 转换为 Gemini 内容列表与 system_instruction。"""
    contents: List[types.Content] = []
    system_instructions: List[str] = []
    tool_name_by_call_id: Dict[str, str] = {}

    index = 0
    total = len(context_items)

    while index < total:
        item = context_items[index]
        item_type = str(item.get("item_type") or "").strip()

        # 1. 系统指令
        if item_type == "SystemMessageItem":
            parts = item.get("parts") or []
            for part in parts:
                text = str(part.get("text") or "").strip()
                if text:
                    system_instructions.append(text)
            index += 1
            continue

        # 2. 用户消息（支持文本与图片多模态）
        if item_type == "UserMessageItem":
            user_parts: List[types.Part] = []
            for part in item.get("parts") or []:
                part_type = str(part.get("type") or "").strip().lower()
                if part_type == "text":
                    text = str(part.get("text") or "")
                    if text:
                        user_parts.append(types.Part.from_text(text=text))
                elif part_type == "image":
                    img_base64 = str(part.get("image_base64") or "").strip()
                    img_format = str(part.get("image_format") or "jpeg").strip()
                    if img_base64:
                        user_parts.append(
                            types.Part.from_bytes(
                                data=base64.b64decode(img_base64),
                                mime_type=_normalize_image_mime(img_format),
                            )
                        )
            if user_parts:
                contents.append(types.Content(role="user", parts=user_parts))
            index += 1
            continue

        # 3. 助手消息或工具调用聚合（Gemini 要求 model 轮次整合在一起）
        if item_type in ("AssistantMessageItem", "FunctionCallItem", "ReasoningItem"):
            group_items = [item]
            index += 1
            while index < total:
                next_item = context_items[index]
                next_type = str(next_item.get("item_type") or "").strip()
                if next_type in ("AssistantMessageItem", "FunctionCallItem", "ReasoningItem"):
                    group_items.append(next_item)
                    index += 1
                else:
                    break

            model_parts: List[types.Part] = []
            for g_item in group_items:
                g_type = str(g_item.get("item_type") or "").strip()
                if g_type == "AssistantMessageItem":
                    for part in g_item.get("parts") or []:
                        text = str(part.get("text") or "")
                        if text:
                            model_parts.append(types.Part.from_text(text=text))
                elif g_type == "FunctionCallItem":
                    tool_call = g_item.get("tool_call") or {}
                    func_name = str(tool_call.get("func_name") or "").strip()
                    call_id = str(tool_call.get("call_id") or "").strip()
                    if call_id and func_name:
                        tool_name_by_call_id[call_id] = func_name
                    args = tool_call.get("args") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    if func_name:
                        model_parts.append(types.Part.from_function_call(name=func_name, args=args))
            if model_parts:
                contents.append(types.Content(role="model", parts=model_parts))
            continue

        # 4. 工具输出响应聚合（Gemini 强制要求并行工具调用的响应放在同一个轮次中）
        if item_type == "FunctionCallOutputItem":
            group_items = [item]
            index += 1
            while index < total:
                next_item = context_items[index]
                next_type = str(next_item.get("item_type") or "").strip()
                if next_type == "FunctionCallOutputItem":
                    group_items.append(next_item)
                    index += 1
                else:
                    break

            resp_parts: List[types.Part] = []
            for g_item in group_items:
                call_id = str(g_item.get("call_id") or "").strip()
                func_name = str(g_item.get("tool_name") or tool_name_by_call_id.get(call_id, "")).strip()
                output = g_item.get("output")
                if not isinstance(output, (dict, list)):
                    output_payload = {"output": str(output or "")}
                else:
                    output_payload = {"output": output}
                resp_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name or "tool",
                            response=output_payload,
                            id=call_id or None,
                        )
                    )
                )
            if resp_parts:
                contents.append(types.Content(role="user", parts=resp_parts))
            continue

        # 其他未识别项直接跳过
        index += 1

    sys_text = "\n\n".join(system_instructions).strip() or None
    return contents, sys_text


def convert_tools_to_gemini(tool_options: Optional[List[Dict[str, Any]]]) -> List[types.Tool]:
    """将 MaiBot 的工具定义转换为 Gemini Tool 声明。"""
    if not tool_options:
        return []

    declarations: List[types.FunctionDeclaration] = []
    for tool in tool_options:
        # 支持 OpenAI 格式或平铺格式
        fn_data = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(fn_data.get("name") or "").strip()
        if not name:
            continue
        description = str(fn_data.get("description") or "").strip()
        params = fn_data.get("parameters") or fn_data.get("parameters_schema") or {}

        decl_payload: Dict[str, Any] = {
            "name": name,
            "description": description or None,
        }
        if params:
            decl_payload["parameters_json_schema"] = params

        declarations.append(types.FunctionDeclaration(**decl_payload))

    if declarations:
        return [types.Tool(function_declarations=declarations)]
    return []


def build_generate_content_config(
    request: Dict[str, Any],
    system_instruction: Optional[str] = None,
) -> types.GenerateContentConfig:
    """根据 MaiBot 请求快照构建 GenerateContentConfig。"""
    config_args: Dict[str, Any] = {}

    if system_instruction:
        config_args["system_instruction"] = system_instruction

    # 温度与 token
    temp = request.get("temperature")
    if temp is not None:
        try:
            config_args["temperature"] = float(temp)
        except (TypeError, ValueError):
            pass

    max_tokens = request.get("max_tokens")
    if max_tokens is not None:
        try:
            config_args["max_output_tokens"] = int(max_tokens)
        except (TypeError, ValueError):
            pass

    # 工具
    tools = convert_tools_to_gemini(request.get("tool_options"))
    if tools:
        config_args["tools"] = tools

    tool_choice = str(request.get("tool_choice") or "").strip().lower()
    if tool_choice == "none":
        config_args["tool_config"] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE")
        )

    # 思考链（Thinking）
    extra_params = request.get("extra_params") or {}
    explicit_thinking = extra_params.get("thinking_config") or extra_params.get("think_config")
    if explicit_thinking is not None:
        if isinstance(explicit_thinking, dict):
            config_args["thinking_config"] = types.ThinkingConfig(**explicit_thinking)
        else:
            config_args["thinking_config"] = explicit_thinking
    else:
        thinking_level = extra_params.get("thinking_level") or extra_params.get("reasoning_effort")
        thinking_budget = None
        if "thinking_budget" in extra_params:
            try:
                thinking_budget = int(extra_params["thinking_budget"])
            except (TypeError, ValueError):
                pass

        thinking_args: Dict[str, Any] = {
            "include_thoughts": bool(extra_params.get("include_thoughts", True))
        }
        if thinking_level is not None:
            thinking_args["thinking_level"] = str(thinking_level).strip()
        if thinking_budget is not None:
            thinking_args["thinking_budget"] = thinking_budget
        config_args["thinking_config"] = types.ThinkingConfig(**thinking_args)

    return types.GenerateContentConfig(**config_args)


def parse_gemini_response_to_maibot(
    response: types.GenerateContentResponse,
    logical_turn_id: Optional[str] = None,
) -> Dict[str, Any]:
    """解析 Gemini 响应，生成符合 MaiBot 协议的完整字典。"""
    active_turn_id = logical_turn_id or uuid.uuid4().hex
    text_pieces: List[str] = []
    reasoning_pieces: List[str] = []
    tool_calls_payload: List[Dict[str, Any]] = []
    output_items: List[Dict[str, Any]] = []
    finish_reason = None

    if response.candidates:
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        content = candidate.content
        if content and content.parts:
            for part in content.parts:
                # 1. 思考链内容
                if getattr(part, "thought", False):
                    thought_text = str(getattr(part, "text", "") or "")
                    if thought_text:
                        reasoning_pieces.append(thought_text)
                    continue

                # 2. 函数调用
                fn_call = getattr(part, "function_call", None)
                if fn_call is not None:
                    call_name = str(getattr(fn_call, "name", "") or "")
                    raw_args = getattr(fn_call, "args", {})
                    call_args = raw_args if isinstance(raw_args, dict) else {}
                    call_id = f"call_{uuid.uuid4().hex[:8]}"

                    tool_calls_payload.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": call_name,
                                "arguments": call_args,
                            },
                        }
                    )
                    output_items.append(
                        {
                            "item_type": "FunctionCallItem",
                            "tool_call": {
                                "call_id": call_id,
                                "func_name": call_name,
                                "args": call_args,
                                "extra_content": None,
                            },
                        }
                    )
                    continue

                # 3. 普通文本
                part_text = str(getattr(part, "text", "") or "")
                if part_text:
                    text_pieces.append(part_text)

    full_text = "".join(text_pieces)
    full_reasoning = "\n".join(reasoning_pieces)

    if not full_text and not tool_calls_payload and not full_reasoning:
        finish_reason_str = (str(finish_reason) if finish_reason is not None else "").upper()
        if "STOP" in finish_reason_str:
            logger.debug("Vertex AI 正常响应静默结束（无正文且无工具调用，finish_reason=%s）", finish_reason)
        else:
            prompt_feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(prompt_feedback, "block_reason", None) if prompt_feedback else None
            detail = f"block_reason={block_reason}" if block_reason else f"finish_reason={finish_reason}"
            raise RuntimeError(f"Vertex AI 响应中既无文本内容也无工具调用（{detail}）")

    # 组装 AssistantMessageItem
    if full_text:
        output_items.insert(
            0,
            {
                "item_type": "AssistantMessageItem",
                "parts": [{"type": "text", "text": full_text}],
            },
        )

    # 组装 ReasoningItem
    if full_reasoning:
        output_items.insert(
            0,
            {
                "item_type": "ReasoningItem",
                "summary_parts": [full_reasoning],
                "text_parts": [full_reasoning],
                "representation": "summary",
            },
        )

    # 提取使用量
    usage_dict: Dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
    }
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta is not None:
        p_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
        c_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
        t_tokens = int(getattr(usage_meta, "total_token_count", 0) or 0)
        hit_tokens = int(getattr(usage_meta, "cached_content_token_count", 0) or 0)

        usage_dict["prompt_tokens"] = p_tokens
        usage_dict["completion_tokens"] = c_tokens
        usage_dict["total_tokens"] = t_tokens or (p_tokens + c_tokens)
        usage_dict["prompt_cache_hit_tokens"] = hit_tokens
        usage_dict["prompt_cache_miss_tokens"] = max(0, p_tokens - hit_tokens)

    result_payload: Dict[str, Any] = {
        "content": full_text,
        "reasoning_content": full_reasoning or None,
        "output_items": output_items,
        "item_schema_version": ITEM_SCHEMA_VERSION,
        "usage": usage_dict,
        "status": "completed",
    }
    if tool_calls_payload:
        result_payload["tool_calls"] = tool_calls_payload

    return result_payload
