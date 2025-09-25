# Codex Newsletter 工具集

用于增量抓取 Gmail 邮件为 Markdown（存放于 `runtime/mails/`），并在每轮抓取后自动调用本地 Codex CLI 对新增邮件进行解析，输出到 `runtime/read/*.json`。

## 环境与安装
- 需要 Python 3.9+
- 建议使用虚拟环境 `python -m venv .venv && source .venv/bin/activate`
- 一键脚本：`bash setup.sh`
  - 安装依赖：`pyyaml imapclient html2text python-dotenv`
  - 生成 `newsletter.yml` 和可选的 `.env`

## 配置说明（newsletter.yml）
核心字段（示例见仓库中的 `newsletter.yml`）：
- `email`: Gmail 账号、IMAP 设置
- `runtime.out_dir`: 抓取的 Markdown 输出目录（默认 `runtime/mails`）
- `runtime.state_file`: 状态文件（记录上次处理的 UID）
- `fetch.batch_size`: 批量抓取大小
- `proxy`: 可选，HTTP/HTTPS 代理，例如：`http://127.0.0.1:1087`

提示：`runloop.py` 会在每次运行前读取 `newsletter.yml` 中的 `proxy`，若有值，会设置环境变量 `HTTP_PROXY/HTTPS_PROXY/http_proxy/https_proxy`，并在控制台醒目输出。

## 主要脚本
- `getmail.py`: 从 Gmail 增量抓取邮件，输出为 Markdown 文件（带 YAML front matter）。
  - 例：`python getmail.py --config newsletter.yml -v --limit 50`
- `readmail.py`: 读取 `runtime/mails/*.md` 中尚未解析的新增邮件，调用 Codex CLI 生成 `runtime/read/*.json`。
  - 需要已安装 `codex` CLI，并在 PATH 中可用。
  - 常用参数：
    - `--dry-run` 仅预览将处理的文件与命令
    - `--limit N` 仅处理最近 N 封新增邮件
    - `-v/--verbose` 输出更多信息
  - 例：`python readmail.py --dry-run -v`
- `runloop.py`: 周期调度器。每轮先运行 `getmail.py`，紧接着运行 `readmail.py`，然后倒计时等待下一轮。
  - 例：
    - 默认（5 分钟一轮）：`python runloop.py`
    - 自定义间隔（10 分钟）：`python runloop.py --interval 600`
    - 传参给 `getmail.py`：`python runloop.py -- --config newsletter.yml -v`
  - 日志标记：子进程输出会带 `[getmail ...]` 和 `[readmail ...]` 前缀，便于分辨。

## 目录结构
- `runtime/mails/`: 邮件 Markdown 输出目录
- `runtime/read/`: Codex 解析结果（JSON）输出目录
- `runtime/state.json`: 抓取进度状态

## 常见问题
- 无法连接 IMAP：检查 `email` 配置与应用专用密码（或 `.env` 中的变量）
- 代理问题：在 `newsletter.yml` 中设置 `proxy` 后，通过 `runloop.py` 启动会自动生效；单独运行 `getmail.py`/`readmail.py` 时请自行导出代理环境变量
- 未找到 `codex`：请安装 Codex CLI 并确保在 PATH 中

## 开发与调试
- 提高日志级别：为 `getmail.py` 或 `readmail.py` 添加 `-v`
- 只跑一轮：直接运行单个脚本；或 `runloop.py` 启动后 Ctrl+C 结束
