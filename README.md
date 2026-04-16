# RepoCoder-Agent MVP

一个从 0 到 1 的可运行 coding agent MVP，面向真实代码仓库（当前优先 Python），支持：

- 任务输入（goal + commands + patch 指令）
- 仓库扫描与轻量索引
- 相关文件检索（基于 token overlap）
- 任务规划
- 最小补丁修改（replace/append/create）
- 命令/测试执行
- 失败后的自动修复
- 通过 FastAPI 暴露接口
- 可选接入 OpenAI 兼容 API 远程大模型

当前默认对接的是 ModelScope 的 OpenAI 兼容 API：

- `OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1`
- `OPENAI_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct`

当 `OPENAI_API_KEY` 未配置，或远程 LLM 调用失败时，程序会优雅降级到现有的规则式 fallback 逻辑，不会因为模型不可用而退出。

## 1. 项目结构

```text
repocoder-agent/
├─ app/repocoder_agent/
│  ├─ __init__.py
│  ├─ __main__.py
│  ├─ agent.py          # 主流程编排
│  ├─ autofix.py        # 基于报错的自动修复规则
│  ├─ executor.py       # 命令执行
│  ├─ llm_client.py     # OpenAI 兼容 LLM 客户端封装
│  ├─ main.py           # FastAPI 入口
│  ├─ models.py         # Pydantic 模型
│  ├─ patcher.py        # 最小补丁应用
│  ├─ planner.py        # 任务规划
│  └─ repository.py     # 仓库扫描与相关文件检索
├─ scripts/example_run.py
├─ tests/
│  ├─ test_agent_autofix.py
│  ├─ test_api.py
│  ├─ test_llm_integration.py
│  ├─ test_patcher.py
│  └─ test_repository.py
├─ pyproject.toml
└─ README.md
```

## 2. 快速开始

```bash
python -m venv .venv
# Windows
#.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e .[dev]
```

## 3. ModelScope OpenAI 兼容 API 配置

代码会按下面的方式初始化 OpenAI Python SDK：

```python
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

环境变量：

- `OPENAI_API_KEY`：ModelScope Token
- `OPENAI_BASE_URL`：默认 `https://api-inference.modelscope.cn/v1`
- `OPENAI_MODEL`：默认 `Qwen/Qwen3-Coder-30B-A3B-Instruct`

`.env` 示例：

```dotenv
OPENAI_API_KEY=ms-your-token
OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1
OPENAI_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
```

如果你用 shell 启动服务，可以先把 `.env` 导入当前进程环境：

```bash
set -a
source .env
set +a
```

如果不配置 `OPENAI_API_KEY`，系统仍然可以运行，只是会走原有规则式规划 / patch / autofix fallback。

## 4. 启动 API

```bash
uvicorn repocoder_agent.main:app --app-dir app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

## 5. API 示例

### 5.1 扫描仓库

```bash
curl -X POST http://127.0.0.1:8000/scan \
  -H "Content-Type: application/json" \
  -d '{"repository_path":"/path/to/repo"}'
```

### 5.2 规划任务

```bash
curl -X POST http://127.0.0.1:8000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "repository_path":"/path/to/repo",
    "goal":"fix failing tests in parser module",
    "commands":["python -m pytest -q"]
  }'
```

### 5.3 执行任务（含最小补丁 + 运行命令）

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "repository_path":"/path/to/repo",
    "goal":"replace typo",
    "commands":["python -m pytest -q"],
    "patches":[
      {
        "file_path":"src/config.py",
        "operation":"replace",
        "find_text":"TIMEOT = 30",
        "replace_text":"TIMEOUT = 30"
      }
    ],
    "max_iterations":3,
    "auto_fix":true
  }'
```

## 6. 最小远程 LLM 运行示例

当你已经配置好 ModelScope Token 后，可以直接让 agent 依赖远程模型做规划、补丁生成和失败反思：

```bash
set -a
source .env
set +a

curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "repository_path":"/path/to/repo",
    "goal":"make the failing parser tests pass with the smallest safe change",
    "commands":["python -m pytest -q tests/test_parser.py"],
    "max_iterations":3,
    "auto_fix":true
  }'
```

你也可以直接运行示例脚本：

```bash
python scripts/example_run.py /root/autodl-tmp/test_repo
```

## 7. 自动修复策略

当前执行链路如下：

- 仓库扫描与检索保持本地实现
- planner 优先尝试远程 LLM，失败时回退到本地规则计划
- patch 生成优先尝试远程 LLM，失败时回退到目标文本规则提取
- 命令失败后的反思 / retry patch 优先尝试远程 LLM，失败时回退到现有报错规则
- patch 应用、命令执行和 FastAPI 接口保持原有结构

当前内置规则 autofix 仍支持这些错误模式：

- `NameError: name 'xxx' is not defined`
- `ImportError: cannot import name 'X' from 'module'`
- `ModuleNotFoundError: No module named 'xxx'`

## 8. 运行测试

```bash
pytest
```

测试覆盖：

- 仓库扫描与检索
- 补丁应用
- NameError 自动修复闭环
- FastAPI 端点（scan/run）
- 未配置 API key 时的 fallback 行为
- 注入 mock LLM client 后的 plan / patch / retry 流程

## 9. 设计取舍

- 依赖尽量少：`fastapi + uvicorn + openai + pytest`
- 检索采用轻量 token overlap，便于理解与扩展
- OpenAI 兼容 client 独立封装在 `llm_client.py`，后续替换服务商成本低
- 补丁策略强调可控和可回放，仍然坚持结构化 patch 指令
- LLM 只增强 planner / patch / reflection，不替代现有扫描、执行和 patcher 基础设施
