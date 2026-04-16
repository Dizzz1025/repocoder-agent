# RepoCoder-Agent MVP

一个从 0 到 1 的可运行 coding agent MVP，面向真实代码仓库（当前优先 Python），支持：

- 任务输入（goal + commands + patch 指令）
- 仓库扫描与轻量索引
- 相关文件检索（基于 token overlap）
- 任务规划（可解释步骤）
- 最小补丁修改（replace/append/create）
- 命令/测试执行
- 基于报错的迭代修复（MVP 规则）
- 通过 FastAPI 暴露接口

## 1. 项目结构

```text
repocoder-agent/
├─ app/repocoder_agent/
│  ├─ __init__.py
│  ├─ agent.py          # 主流程编排
│  ├─ autofix.py        # 基于报错的自动修复规则
│  ├─ executor.py       # 命令执行
│  ├─ main.py           # FastAPI 入口
│  ├─ models.py         # Pydantic 模型
│  ├─ patcher.py        # 最小补丁应用
│  ├─ planner.py        # 任务规划
│  └─ repository.py     # 仓库扫描与相关文件检索
├─ scripts/example_run.py
├─ tests/
│  ├─ test_agent_autofix.py
│  ├─ test_api.py
│  ├─ test_patcher.py
│  └─ test_repository.py
├─ pyproject.toml
└─ README.md
```

## 2. 快速开始

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -e .[dev]
```

启动 API：

```bash
uvicorn repocoder_agent.main:app --app-dir app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

## 3. API 示例

### 3.1 扫描仓库

```bash
curl -X POST http://127.0.0.1:8000/scan \
  -H "Content-Type: application/json" \
  -d "{\"repository_path\":\"C:/path/to/repo\"}"
```

### 3.2 规划任务

```bash
curl -X POST http://127.0.0.1:8000/plan \
  -H "Content-Type: application/json" \
  -d "{
    \"repository_path\":\"C:/path/to/repo\",
    \"goal\":\"fix failing tests in parser module\",
    \"commands\":[\"python -m pytest -q\"]
  }"
```

### 3.3 执行任务（含最小补丁 + 运行命令）

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d "{
    \"repository_path\":\"C:/path/to/repo\",
    \"goal\":\"replace typo\",
    \"commands\":[\"python -m pytest -q\"],
    \"patches\":[
      {
        \"file_path\":\"src/config.py\",
        \"operation\":\"replace\",
        \"find_text\":\"TIMEOT = 30\",
        \"replace_text\":\"TIMEOUT = 30\"
      }
    ],
    \"max_iterations\":3,
    \"auto_fix\":true
  }"
```

## 4. 自动修复（MVP 规则）

当前支持的错误模式：

- `NameError: name 'xxx' is not defined`
- `ImportError: cannot import name 'X' from 'module'`
- `ModuleNotFoundError: No module named 'xxx'`

自动修复会生成最小补丁并重新执行命令，直到成功或达到 `max_iterations`。

## 5. 运行测试

```bash
pytest
```

测试覆盖：

- 仓库扫描与检索
- 补丁应用
- NameError 自动修复闭环
- FastAPI 端点（scan/run）

## 6. 示例脚本

可直接运行：

```bash
python scripts/example_run.py C:/path/to/repo
```

脚本会构造一个任务并输出结构化 JSON 执行结果。

## 7. 设计取舍（MVP）

- 依赖尽量少：`fastapi + uvicorn + pytest`
- 检索采用轻量 token overlap（便于理解与扩展）
- 补丁策略强调可控和可回放（结构化 patch 指令）
- 自动修复先做规则式闭环，后续可接 LLM 生成补丁
