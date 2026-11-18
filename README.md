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
   - **文本对话与规划**：`replyer`, `planner`, `utils`, `memory` 等（支持现役 Gemini 2.5 / 3 / 4 系列）
   - **视觉多模态**：`vlm`（识图、表情包识别、聊天带图）
   - **语音消息识别**：`voice`（音频 Base64 直传 Gemini 进行精准 ASR 转写）
   - **文本与多模态图片嵌入**：`embedding`、`image_embedding`（全面适配最新 **Gemini Embedding 2** 统一向量空间协议）
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
   - **模型标识符**：填入 Vertex 官方模型名称（如 `gemini-2.5-flash`、`gemini-3.8-flash`、`gemini-4-argon` 等）。
   - **视觉模型（Visual）**：勾选（Gemini 原生支持多模态看图与表情包审核）。
   - **思考参数（可选）**：可在「额外参数（Extra Params）」中配置 `enable_thinking`、`thinking_level` 或 `thinking_budget`（详见下方分代说明）。

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
thinking_budget = 1024                      # Gemini 2.5 系列使用 thinking_budget（0 为关闭，-1 为动态）

[[models]]
model_identifier = "gemini-4-argon"
name = "vertex-gemini-4-argon"
api_provider = "Google-Vertex-Connect"
model_type = "llm"
visual = true

[models.extra_params]
thinking_level = "low"                      # Gemini 3/4 系列使用 thinking_level（low / medium / high）

[[models]]
model_identifier = "gemini-embedding-2-preview"
name = "vertex-gemini-embedding-2-image"
api_provider = "Google-Vertex-Connect"
model_type = "image_embedding"              # 多模态图片嵌入需指定为 image_embedding（纯文本嵌入为 embedding）

[models.extra_params]
dimensions = 1536                           # 可选降维（如 1536 / 768，不填默认输出 3072 维）
```

> [!WARNING]
> **已移除老一代模型支持（重要兼容说明）**：
>
> 由于上游 Google Cloud 已逐步下线并停止提供老一代模型服务，本插件**仅支持现役世代模型**，配置已淘汰模型名将直接报错提示升级：
> - **对话 / 多模态 / 语音模型**：仅支持 **Gemini 2.5 / 3 / 4 系列**（已移除 Gemini 1.0 / 1.5 / 2.0 支持）；
> - **文本与图片嵌入模型**：仅支持 **Gemini Embedding 2 系列**（如 `gemini-embedding-2-preview`、`gemini-embedding-2`），已彻底移除 `text-embedding-004/005`、`text-multilingual-embedding-002`、`gemini-embedding-001` 及旧版 `multimodalembedding@001` 的支持。

> [!IMPORTANT]
> **Gemini Embedding 2（图片/文本嵌入）配置注意事项**：
>
> 1. **禁止填写 `task_type` / `title`**：与老一代嵌入模型不同，**Gemini Embedding 2 已不支持 `task_type` / `task` / `title` 字段**（传入会导致请求报错）。如需区分文本用途请在输入前缀体现；若配置中误填了这些字段，插件会自动过滤并打印 `WARNING` 告警。
> 2. **维度配置建议使用 `dimensions`**：Gemini Embedding 2 默认输出 **3072** 维向量，并支持通过 MRL 降维（如 `1536` / `768`）。在 `[models.extra_params]` 中请优先使用 **`dimensions = 1536`**（插件会自动映射为 Google SDK 的 `output_dimensionality`，同时配合 MaiBot 宿主准确计算向量空间指纹）。
> 3. **与文本记忆维度独立**：在 MaiBot 的 `A_Memorix` 架构中，`image_embedding` 与纯文本 `embedding` 属于两套完全隔离的向量库，**图片维度无需与文本嵌入维度保持一致**；但一旦开始积累图片记忆，**切勿中途修改 `dimensions`**，否则会触发重新向量化重建索引。

> [!IMPORTANT]
> **思考参数设置注意事项（Gemini 2.5 / 3 / 4 分代规范）**：
>
> Google 对不同世代模型的思考参数有严格约束，错误的参数会触发 `HTTP 400` 校验失败：
> - **Gemini 4 系列**（如 `gemini-4-argon` / `hydrogen`）：使用 `thinking_level`，仅支持 `LOW`、`MEDIUM`、`HIGH`、`XHIGH`。**严禁设为 `MINIMAL`**（会触发 HTTP 400 错误）。
> - **Gemini 3 系列**：使用 `thinking_level`。Flash 等轻量系列支持 `MINIMAL`、`LOW`、`MEDIUM`、`HIGH`；Pro 系列不支持 `MINIMAL`，最低为 `LOW`。
> - **Gemini 2.5 系列**：使用 `thinking_budget`（整数 Token 预算）。Flash 系列设为 `0` 可彻底关闭思考，开启时最低为 `128`；Pro 系列**不支持设为 `0` 关闭**，最低预算为 `128`（或 `-1` 动态思考）。
> - **严禁双传**：`thinking_level` 与 `thinking_budget` **不可同时配置**，否则 Google API 会报 400 拒绝请求。
>
> **插件内置智能适配与告警机制**：
> 1. **`enable_thinking = true`（快捷最低配置）**：当开启 `enable_thinking = true` 且未指定具体级别/预算时，插件会自动为当前模型匹配**可用最低思考配置**（Gemini 4 / 3 Pro -> `LOW`；Gemini 3 Flash -> `MINIMAL`；Gemini 2.5 -> `128`）。
> 2. **`enable_thinking = false`（关闭/最小化思考）**：Gemini 2.5 Flash 会被设为 `budget = 0` 彻底关闭；Gemini 3 Flash 会降至 `MINIMAL`；不可关闭的模型系列（2.5 Pro / 3 Pro / 4）会降级至各自最低合法配置并隐藏思考文本（`include_thoughts = false`）。
> 3. **异常兜底与 WARNING 日志**：当出现参数双传冲突、跨世代错配（如给 2.5 配 level 或给 3/4 配 budget），或设置模型不支持的级别/预算（如给 Gemini 4 传 `MINIMAL`、给 2.5 Pro 传 `0`）时，插件会自动进行安全纠偏防止崩溃，并在后台日志中输出 **`WARNING`** 级别告警提示修正。

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

## 常见问题与排障（Troubleshooting）

### 1. 快速检查运行状态与凭据
- 在聊天框直接向 Bot 发送指令 **`/vertex_status`**：
  - 插件会输出当前识别到的 **Project ID**、**Location** 以及**激活的鉴权模式**（如 Service Account / Express Key / ADC）。
  - 若提示未识别到 Project ID，请检查 Provider 设置或插件配置。
- 发送 **`/vertex_models`**：可查看官方推荐的常用 Gemini 模型标识列表。

### 2. `PermissionDenied (403)` / IAM 权限不足
- 确保你的 Google Cloud 项目中已**启用 Vertex AI API**：
  - 前往 [Google Cloud Console API 库](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com) 启用 `aiplatform.googleapis.com`。
- 如果使用 Service Account，确保该服务账号在 IAM 中被授予了 **`Vertex AI User`（`roles/aiplatform.user`）** 角色。

### 3. `404 Publisher Model ... not found` 或 `400 Location not supported`
- 检查你请求的模型是否在该区域提供。推荐优先使用 `global` 或 `us-central1`；
- 检查 `base_url` 与手动配置的 `location` 是否冲突（例如配了 `us-central1` 的端点却在请求头指定了 `europe-west4`）。

### 4. 模型返回空内容或触发安全审核
- Google Gemini 模型内置了严格的 Safety 过滤与 Recitation（防版权侵权）检测；
- 若触发安全拦截，插件会捕获并暴露具体的原因（如 `finish_reason: SAFETY` 或 `block_reason`），并在 MaiBot 日志中清晰记录，以便排查输入提示词。

---

## 隐私与数据安全（Privacy & Security）

- **本地凭据隔离**：你的 Google Cloud Service Account JSON 密钥、API Key 或访问令牌仅保存在本地服务器中，**绝不上传或回传至任何第三方服务器**；
- **官方 API 直连**：所有的对话消息、图片多模态数据、语音识别切片以及向量嵌入请求，均**直连 Google 官方 Vertex AI 端点**（或你明确指定的自建网关），中间没有任何开发者中转代理；
- **零遥测与埋点**：本插件源码完全公开透明，无任何数据埋点、使用追踪或上报机制；
- **防意外泄露**：项目自带的 `.gitignore` 已内置过滤 `config.toml`、`config_back/` 以及各类 JSON 密钥文件，防止在版本管理或共享时意外泄露私钥。

---

## 开发与测试验证（Testing）

在贡献代码或安装后验证环境时，可在 MaiBot 根目录下直接运行插件的单元测试套件：

```bash
# 运行全部插件单测
.venv/bin/pytest plugins/vertexai-maibot-connect/tests -v
```

测试集涵盖：
- 7 种鉴权模式（Bearer Token, Express Key, SA JSON 文件/字符串, ADC 等）与 Project ID 解析；
- 多模态 Context Items（文本/图片/System 指令）转 Gemini Contents；
- 工具定义与思考级别（`thinking_level`）配置映射；
- 空响应安全拦截与 `finish_reason` 诊断；
- 插件动态加载与卸载生命周期。

---

## 许可证

本项目采用 [MIT](LICENSE) 许可证。
