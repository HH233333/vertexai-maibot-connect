# VertexAI-MaiBot-connect-plugin

通过 Google Cloud **Vertex AI 官方 SDK (`google-genai`)** 将 Vertex AI 服务接入 **MaiBot**。

本插件采用插件化 LLM Provider 架构（`client_type = "vertex_connect"`），主程序无需侵入改动，完美支持对话、规划、多模态识图（VLM）、语音消息识别转写（Voice）与向量嵌入（Embedding）。

---

## 功能特性

1. **支持原版 7 种凭据自动解析与降级**：
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

### 2. 配置模型提供商

#### 方式一：在 WebUI 中配置（推荐）

1. 进入 MaiBot WebUI -> **「模型设置」** -> **「提供商管理」** -> 点击**「添加提供商」**：
   - **提供商名称**：自定义，例如 `vertex` 或 `Google-Vertex-Connect`
   - **客户端类型（Client Type）**：选择或输入 **`vertex_connect`**（核心必填项，关联本插件通道）
   - **Base URL**：例如 `https://aiplatform.googleapis.com` 或 `https://us-central1-aiplatform.googleapis.com`（根据你想使用的区域填写）

2. **根据你使用的鉴权方式，在 WebUI 进行相应填写**：

| 鉴权方式 | 「API Key」输入框填写 | 「高级设置 -> 请求头（Headers）」填写 | 说明与特点 |
| :--- | :--- | :--- | :--- |
| **1. Service Account JSON 文件路径（推荐）** | 填写服务器上的绝对路径，例如 `/home/user/maibot-sa.json` | 留空（无需手动填 project） | 插件会自动读取 JSON 内的 `project_id` 与私钥，最简单稳妥 |
| **2. Service Account JSON 内容字符串** | 直接粘贴完整的 JSON 文本（以 `{` 开头） | 留空（无需手动填 project） | 适合容器化或无法单独上传文件的部署环境，自动解析 `project_id` |
| **3. Vertex Express Mode / GCP API Key** | 填写以 `AIzaSy...` 或 `AQ...` 开头的 API Key | 添加 `project: 你的GCP项目ID` | 类似 Gemini Studio API Key 的简易模式 |
| **4. OAuth2 Bearer Access Token** | 填写以 `ya29.` 开头的临时访问令牌 | 添加 `project: 你的GCP项目ID` | 适合通过外部脚本/网关自动化注入 Access Token（有效期通常 1 小时） |
| **5. 环境变量 `GOOGLE_APPLICATION_CREDENTIALS`** | **留空不填** | 留空（若 JSON 内含 project）或填 `project: 你的GCP项目ID` | 运行 MaiBot 的环境已导出凭据文件路径变量 |
| **6. 本地默认凭据 (ADC)** | **留空不填** | 添加 `project: 你的GCP项目ID` | 服务器执行过 `gcloud auth application-default login` |
| **7. 本地 `gcloud` CLI 自动探测** | **留空不填** | **留空不填** | 服务器已安装并登录 `gcloud`，插件自动提取 Token 与当前激活的 Project ID |

> [!TIP]
> **关于 Base URL 与 Location（区域）自动推导机制与注意事项**：
>
> 插件内置了 **URL 区域智能提取与推导逻辑**，规则如下：
> - **使用 Google 官方区域 URL（推荐，无需手动配置 location）**：
>   - 单区域端点（如 `https://us-central1-aiplatform.googleapis.com`）：插件会自动捕获并设定 `location = "us-central1"`；
>   - 多区域端点（如 `https://aiplatform.us.rep.googleapis.com`）：插件会自动捕获并设定 `location = "us"`；
>   - 使用这类带区域的官方 URL 时，**请求头中的 `location` 留空即可**，插件会自动保持二者一致。
> - **使用官方全局端点（`https://aiplatform.googleapis.com`）**：
>   - 默认使用 `global`。如所请求的模型仅在特定区域开放（例如 `us` / `us-central1`），请在请求头中添加 `location: 目标区域`。
> - **使用第三方自建反向代理（如自定义域名）**：
>   - 插件无法从自定义域名识别出所属区域，若不配置将降级使用 `global`。若后端代理转接到特定区域模型，**必须在请求头中显式指定 `location`**（如 `us` 或 `us-central1`），否则 Google 会因资源路径不匹配返回 `404 Not Found` 或 `400 Location not supported`。
> - **避免冲突**：切勿将 URL 区域与手动配置的 `location` 混配为不同区域（例如配了 `us-central1` 的 URL 却在请求头指定 `europe-west4`），否则 Google 区域网关会直接返回 400 校验拒绝错误。

3. **添加模型**：
   - 在 WebUI「模型管理」中点击「添加模型」，所属提供商选择刚创建的 Vertex 提供商。
   - **模型标识符**：填入 Vertex 官方模型名称（如 `gemini-2.5-flash`、`gemini-2.5-pro`）。
   - **视觉模型（Visual）**：勾选（Gemini 原生支持多模态看图与表情包审核）。
   - **思考参数（可选）**：可在「额外参数（Extra Params）」中添加 `thinking_level = "low"`（支持 `low` / `normal` / `high`）。

---

#### 方式二：直接在配置文件中配置（TOML）

如果你习惯直接修改配置文件，可在 `config/model_config.toml`（或 `host/model_catalog.toml`）中添加提供商与模型定义：

```toml
[[api_providers]]
name = "Google-Vertex-Connect"
base_url = "https://aiplatform.googleapis.com"
# 凭据可填入：Service Account JSON 绝对路径、完整 JSON 字符串、ya29 Token、API Key，或留空走本地 ADC/gcloud
api_key = "/path/to/service-account.json"
auth_type = "none"
client_type = "vertex_connect"             # 必须为 vertex_connect（对应插件注册的通道）
timeout = 120
max_retry = 2
retry_interval = 3

[api_providers.default_headers]
project = "your-gcp-project-id"             # GCP 项目 ID（若 api_key 为 SA JSON 可留空自动提取）
location = "global"                         # Vertex AI 区域（默认 global；若 base_url 已含区域如 us/us-central1 可留空自动提取）

[[models]]
model_identifier = "gemini-2.5-flash"
name = "vertex-gemini-2.5-flash"
api_provider = "Google-Vertex-Connect"
model_type = "llm"
visual = true

[models.extra_params]
thinking_level = "low"                      # 可选：low / normal / high

[[models]]
model_identifier = "gemini-2.5-pro"
name = "vertex-gemini-2.5-pro"
api_provider = "Google-Vertex-Connect"
model_type = "llm"
visual = true
```

---

#### 模型任务配置

模型任务配置（如 `replyer` 对话回复、`planner` 动作规划、`vlm` 识图与表情包审核、`voice` 语音转写等）**与 MaiBot 正常的模型任务配置方法完全一样**。

你只需在 WebUI 的「任务分配」界面，或者在 `model_config.toml` 的 `[model_task_config.*]` 中，将对应的任务分配给上面添加的 Vertex 模型名称（如 `vertex-gemini-2.5-flash`）即可，无需进行任何特殊处理。

### 3. 插件配置（可选）
在 WebUI「插件配置」或插件生成的 `config.toml` 中，可配置默认 Project ID 和区域：
```toml
[plugin]
enabled = true
default_project = "your-gcp-project-id"
default_location = "global"
```

---

## 许可证

本项目采用 [MIT](LICENSE) 许可证。
