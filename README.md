# ChatExp

ChatExp 是一个面向实验设计与工艺优化的对话系统，支持实验数据整理、多目标评分、贝叶斯优化，以及文件证据分析。

## 项目结构

- [chatexp-tui/](chatexp-tui/README.md)：终端客户端、CLI 和 MCP 服务。
- [chatexp-api/](chatexp-api/README.md)：优化 API、LLM 代理、文件分析服务和部署模板。
- [SeaTable 插件安装包](chatexp-seatable-plugin/chatexp-0.2.4.zip)：编译后的插件 ZIP；本仓库不提供插件源码。

## 启动终端客户端

建议使用 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./chatexp-tui
cp chatexp-tui/.env.example chatexp-tui/.env
# 在 .env 中填写自己的模型连接配置。
cd chatexp-tui
python -m chatexp.cli chat
```

详细配置与 MCP 使用方式见 [TUI 文档](chatexp-tui/README.md)。

## 部署后端

```bash
cd chatexp-api
cp .env.example .env
# 配置 API 密钥、模型连接和部署参数。
docker compose up -d --build
```

服务配置、HTTPS 部署及接口说明见 [API 文档](chatexp-api/README.md)。

## 安装 SeaTable 插件

下载 [chatexp-0.2.4.zip](chatexp-seatable-plugin/chatexp-0.2.4.zip)，通过 SeaTable 插件管理上传并启用，然后在插件设置中填写后端和模型连接。安装包不含预置连接凭据。

## 许可证

见 [LICENCE](LICENCE)。
