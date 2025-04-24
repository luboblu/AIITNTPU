# Commands

## venv

- `.venv\Scripts\activate`: 啟用虛擬環境
- `deactivate`: 停用虛擬環境

## uv

- `uv init`: 初始化當前專案
- `uv venv`: 建立 venv
- `uv add <packages...>`: 新增套件
- `uv remove <packages...>`: 移除套件
- `uv sync`: 安裝所有 project.toml 裡面套件
- `uv lock`: 建立 lockfile
- `uv pip freeze > requirements.txt`: 產出 requirements.txt

### 安裝

- [Github](https://github.com/astral-sh/uv)

```pwsh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
