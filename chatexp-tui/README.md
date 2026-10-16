# ChatExp

`ChatExp` 是一个面向实验设计与工艺优化的终端对话系统。它把大语言模型作为人机语义接口，用来理解自然语言、整理粘贴数据、解释分析结果；全量 Excel/工作簿导入、筛选、评分和贝叶斯优化由 Python 执行，实验状态保存在本地工作簿中，方便审计和复现。

## 主要能力

- **Human-in-the-loop 实验优化**：人负责目标、约束、可行性判断和 go/no-go 决策，系统负责结构化记录、分析和下一批建议。
- **LLM 语义控制**：识别粘贴的实验记录、自然语言修改、结果汇报和分析请求，输出结构化 action。
- **大数据边界清晰**：用户直接粘贴的数据可由 LLM 结构化；Excel/工作簿等文件数据只把样本窗口/统计摘要发给 LLM，全量操作由 Python 完成。
- **混合变量与多目标**：支持 continuous / ordinal / categorical 因素，以及 maximize / minimize / target 目标。
- **可审计工作簿**：默认写入 `META / PARAMETERS / CATEGORY / MPO / EXPERIMENT / DESIGNS` 等 sheet。

## 目录结构

```text
ChatExp/
├── chatexp-tui/             # 终端/TUI 和 MCP
│   ├── chatexp/             # Python 包（pip install -e . 后 import chatexp.*）
│   │   ├── cli.py           # CLI 入口（typer）
│   │   ├── chat.py          # 对话协调与 action 执行
│   │   ├── service.py       # 项目/数据/分析/优化服务
│   │   ├── design.py        # BO 设计建议
│   │   ├── desirability.py  # 多目标 desirability 评分
│   │   ├── llm.py           # requests-based OpenAI-compatible LLM client
│   │   ├── storage.py       # 工作簿读写与审计 sheet 布局
│   │   ├── skills.py / skill_specs/   # action/能力契约
│   │   └── tui.py / terminal_harness.py  # Textual 界面与终端审计
├── chatexp-seatable-plugin/ # SeaTable 插件
├── chatexp-api/             # 独立 BO API 后端
└── data/workspace/          # 本地运行时生成的工作簿和会话
```

## 1. 使用 `.venv310` 本地虚拟环境

本项目现在推荐使用仓库内的 `.venv310`，不再把 conda/mamba 作为默认安装路径。

```bash
cd chatexp-tui               # 仓库内相对路径；按你的实际 checkout 位置调整
python3.10 -m venv .venv310
source .venv310/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

如果系统提示 `ensurepip is not available`，说明缺少系统 venv 支持。Debian/Ubuntu 可先安装：

```bash
sudo apt install python3.10-venv python3-pip
```

然后删除旧环境并重建：

```bash
rm -rf .venv310
python3.10 -m venv .venv310
source .venv310/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

`pip install -e .` 之后，除了 `python -m chatexp.cli`，也会安装两个入口命令：`chatexp`（CLI）和 `chatexp-mcp`（MCP server）。`.venv310/` 已在仓库根 `.gitignore` 中，不会误提交。

## 2. 验证环境

```bash
source .venv310/bin/activate
python -c "import pandas, numpy, torch, openpyxl, typer, rich, textual, pydantic, requests; print('ChatExp env OK')"
```

## 3. 配置 LLM

复制模板并填写你的 OpenAI-compatible endpoint：

```bash
cp .env.example .env
```

常用字段：

```bash
CHATEXP_OPENAI_API_KEY=你的key
CHATEXP_OPENAI_BASE_URL=https://your-endpoint/v1
CHATEXP_OPENAI_MODEL=你的模型名
CHATEXP_OPENAI_TIMEOUT_SECONDS=180
CHATEXP_OPENAI_MAX_TOKENS=131072
```

如果你的模型支持 128k 上下文，可以把 `CHATEXP_OPENAI_MAX_TOKENS` 继续调大。注意：这个值控制输出预算，不代表系统会把 Excel 全量数据发给 LLM。

### 3.1 BO 后端（chatexp-api）

`.env` 中配置 `CHATEXP_BO_API_ENDPOINT` 后，`generate_next_design` 优先调用 chatexp-api 服务（`POST /v1/bo/suggest`，202+轮询+取消，Bearer 鉴权）生成批次；API 未配置或不可用时自动回退到本地 torch 引擎，并写入审计事件。运行档位可用 `CHATEXP_BO_RUNTIME_CANDIDATE_POOL_SIZE / _ENSEMBLE_SIZE / _TRAINING_STEPS` 覆盖（默认 2048/5/300）。

### 3.2 Campaign Strategist（LLM 帮 BO：逐批目标状态策略器）

`.env` 中 `CHATEXP_STRATEGIST=llm|off`（默认 off）。这是 LLM 对 BO 的直接帮助，**没有规则兜底**：开启后，每批设计前由 LLM 策略器基于战役摘要（目标观测跨度、因子重要性、最优轨迹、代理器状态）产出结构化指令 `{phase, weight_adjustments, soft_constraints, range_extensions, stop_recommendation, rationale}`：
- 权重调整被钳制在操作者权重的 [0.5×, 2×] 且绝对值 [0.05, 5.0] 内，仅作用于本批采集，绝不改写工作簿中的操作者契约；
- 指令摘要在设计确认消息中展示（操作者知情批准），stop 建议是显式提示而非静默停机；
- 范围扩展（range_extensions）：当最优观测钉在声明边界（5% 内）时，策略器可建议扩展该因素范围——仅扩展、每侧最多 2 倍当前跨度、每轮最多 2 个因素、经操作者确认后持久写入工作簿并审计（`range_extended` 事件）；
- 每次决策写入审计事件日志（`strategy_decision`，含理由）；LLM 不可用或输出无效时本批**没有策略**（如实记录 `strategist unavailable` 事件），绝不用规则替代模型判断。

### 3.3 文件证据（LLM 自主研究你指定的 word/excel/pdf）

在对话中直接给出文件路径（如 `看看 D:/data/runs.xlsx 里的实验数据`），规划器会输出 `evidence_requests` 请求解析，而不是凭文件名猜内容。解析顺序：配置了 `CHATEXP_BO_API_ENDPOINT` 时走 chatexp-api 文档服务（docling，与 seatable 插件同一条 `/v1/files/parse` 管线）；否则用本地真实解析器（openpyxl / python-docx / pypdf / csv，其中 python-docx 与 pypdf 为可选依赖，`pip install python-docx pypdf`）。两者产出的 chunk 形状一致，且本地 xlsx 额外生成 sheet 摘要（工作表、维度、列名、前 12 行）。解析失败是结构化失败（`file_not_found` / `too_large` / `missing_dependency` ...），planner 对失败文件的任何内容描述都必须明确承认失败——"假装读过"会被拦截并要求重答。证据预算：每次最多 3 个文件、6 个 chunk。

### 3.4 审计与溯源

- `event-log-v1.jsonl`（任务目录）：`user_message / plan_executed / observation / error / budget / strategy_decision / design_requested / campaign_*` 事件，`replay_for_recovery()` 支持会话恢复回放；
- `llm_provenance.json`（任务目录）：模型、各阶段温度、全部提示词模板 SHA-256——用于核对运行配置；
- `llm_traces/*.json`：每次 LLM 调用的请求/响应/用量留档，预算事件汇总每次请求的 token 消耗。

## 4. 启动终端界面

```bash
source .venv310/bin/activate
python -m chatexp.cli chat
```

也可以先查看路径：

```bash
python -m chatexp.cli paths
```

默认产物会写到：

- `../data/workspace/session.json`
- `../data/workspace/*.xlsx`

## 5. CLI 对话示例

```text
新建excel，文件名叫电池优化.xlsx
因素定义：温度: 连续 20到80；时间: 离散 10到60 步长 10；催化剂: 类别 A/B/C
目标是产率最大化
初始化设计 12 组
从 data.xlsx 导入实验数据，目标列为产率
追加实验：温度=40，时间=20，催化剂=A，产率=83.2
分析现在的数据，看看还缺什么
生成新的设计 5 组
把目标改成成本最小化
导出
```

用户直接粘贴一长串实验记录时，ChatExp 会让 LLM 结构化 rows/factors/objectives；如果是 Excel/工作簿，ChatExp 只让 LLM 识别 schema，之后由 Python 全量导入和分析。

## 6. MCP server

```bash
source .venv310/bin/activate
python -m chatexp.mcp_server
```

MCP 暴露的主要工具：

- `chat`
- `confirm_pending_action`
- `create_project`
- `initialize_design`
- `import_experiment_sheet`
- `append_results`
- `generate_next_design`
- `update_objective`
- `update_factor`
- `refresh_scores`
- `inspect_workbook`
- `project_state`
