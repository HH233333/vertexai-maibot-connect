# VertexAI-MaiBot-connect-plugin

通过 Google Cloud **Vertex AI 官方 SDK (`google-genai`)** 将 Vertex AI 服务接入 **MaiBot**。

本插件采用插件化 LLM Provider 架构（`client_type = "vertex_connect"`），主程序无需侵入改动，完美支持对话、规划、多模态识图（VLM）、语音消息识别转写（Voice）与向量嵌入（Embedding）。

---

## 功能特性

1. **延续原版 7 种凭据自动解析与降级**：
   - **OAuth2 Bearer Token**（`ya29.` 开头）
   - **Vertex Express Mode / API Key**（`AIzaSy...` 或 `AQ...` 开头）
   - **Service Account JSON 文件路径**（在 `api_key`、请求头 `credentials_file` 或环境变量 `GOOGLE_APPLICATION_CREDENTIALS` 中指定）
   - **Service Account JSON 内容字符串**（直接将 JSON 填入 `api_key`）
   - **环境变量 `GOOGLE_APPLICATION_CREDENTIALS`**
   - **本地 ADC 默认凭据**（`google.auth.default()`）
   - **本地 `gcloud` CLI 探测与 Access Token 自动提取**（`gcloud auth print-access-token`）
2. **全角色多模态覆盖**：
   - **文本对话与规划**：`replyer`, `planner`, `utils`, `memory` 等
   - **视觉多模态**：`vlm`（识图、表情包识别、聊天带图）
   - **语音消息识别**：`voice`（音频 Base64 直传 Gemini 进行精准 ASR 转写）
   - **文本嵌入**：`embedding`
3. **函数调用与思考链**：
   - 自动映射 MaiBot 工具定义（Tool Call / Function Calling）
   - 支持捕捉 Gemini 思考链（Thinking / Thought），组装为统一的 `ReasoningItem`
4. **管理指令**：
   - `/vertex_status`：查看当前凭据解析状态、Project ID 与 Location
   - `/vertex_models`：查看推荐模型标识

---

## 安装与使用

### 1. 安装依赖
确保在 MaiBot 的 Python 虚拟环境中安装依赖：
```bash
.venv/bin/pip install -r plugins/vertexai-maibot-connect/requirements.txt
```

### 2. 配置 MaiBot
将 [配置片段示例](examples/model_config.fragment.toml) 合并到你的 `config/model_config.toml` 中：

```toml
[[api_providers]]
name = "Google-Vertex-Connect"
base_url = "https://us-central1-aiplatform.googleapis.com"
api_key = "/path/to/service-account.json"  # 或填入 Bearer Token / Express API Key，或留空走本地 ADC/gcloud
auth_type = "none"
client_type = "vertex_connect"             # 必须为 vertex_connect
timeout = 120

[[models]]
model_identifier = "gemini-2.5-flash"
name = "vertex-gemini-2.5-flash"
api_provider = "Google-Vertex-Connect"
model_type = "llm"
visual = true

[model_task_config.replyer]
model_list = ["vertex-gemini-2.5-flash"]

[model_task_config.vlm]
model_list = ["vertex-gemini-2.5-flash"]

[model_task_config.voice]
model_list = ["vertex-gemini-2.5-flash"]
```

### 3. 插件配置（可选）
在 WebUI「插件配置」或插件生成的 `config.toml` 中，可配置默认 Project ID 和区域：
```toml
[plugin]
enabled = true
default_project = "your-gcp-project-id"
default_location = "us-central1"
```

---

## 许可证

本项目采用 [MIT](LICENSE) 许可证。
