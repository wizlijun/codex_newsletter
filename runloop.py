#!/usr/bin/env python3
"""
周期运行器：每隔固定时间执行一次任务，并在控制台显示倒计时和详细日志。
每轮会先运行 getmail.py，随后立即运行 readmail.py。

用法示例：
  - 直接运行（默认 5 分钟，默认使用 newsletter.yml）：
      python runloop.py
  - 指定间隔为 10 分钟：
      python runloop.py --interval 600
  - 传递参数给 getmail.py（在 -- 之后的内容都会原样传递）：
      python runloop.py -- --config newsletter.yml -v
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from typing import Optional as _Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def human_ts(dt: Optional[float] = None) -> str:
    t = datetime.fromtimestamp(dt if dt is not None else time.time())
    return t.strftime("%Y-%m-%d %H:%M:%S")


def stream_process(cmd: List[str], tag: str) -> int:
    """
    启动子进程并实时逐行转发输出，返回退出码。
    """
    print(f"[runloop] 启动命令: {cmd}")
    print(f"[runloop] 开始时间: {human_ts()}")
    start = time.time()

    # 继承当前环境；将 stdout+stderr 合并，逐行读取
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            # 去除尾随换行，带上时间戳转发
            sys.stdout.write(f"[{tag} {human_ts()}] {line}")
            sys.stdout.flush()
    except KeyboardInterrupt:
        # 将中断转发给子进程，再次 Ctrl+C 可强制退出
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
    finally:
        proc.wait()

    rc = proc.returncode or 0
    dur = time.time() - start
    print(f"[runloop] 结束时间: {human_ts()} (耗时 {dur:.2f}s, 退出码 {rc})")
    if rc != 0:
        print("[runloop] 警告：子进程返回非零退出码，稍后将继续重试…")
    return rc


def countdown(seconds: int) -> None:
    """
    逐秒显示倒计时，可用 Ctrl+C 中断整个循环。
    """
    for remaining in range(seconds, 0, -1):
        m, s = divmod(remaining, 60)
        sys.stdout.write(f"\r[runloop] 下次运行倒计时: {m:02d}:{s:02d} (按 Ctrl+C 退出)")
        sys.stdout.flush()
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            # 清理行尾并抛出，外层接住
            sys.stdout.write("\r\n")
            sys.stdout.flush()
            raise
    sys.stdout.write("\r[runloop] 倒计时结束，开始下一轮运行…             \n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="周期执行 getmail.py 的运行器")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="两次运行之间的间隔秒数（默认 300=5 分钟）",
    )
    # 将 -- 之后的参数原样传递给 getmail.py
    parser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="在 -- 之后的参数会原样传递给 getmail.py",
    )
    args = parser.parse_args()

    # 工作目录切到脚本所在目录，确保相对路径（如 newsletter.yml）正确
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    # 组装命令：默认使用当前解释器调用 getmail.py，并默认配置文件 newsletter.yml
    base_cmd = [sys.executable, "getmail.py"]
    # 处理透传参数：若未提供 -- 则给它一个默认配置参数
    rest = list(args.rest or [])
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        # 默认参数：使用 repo 根目录中的 newsletter.yml
        default_cfg = "newsletter.yml"
        rest = ["--config", default_cfg]

    def detect_config_path(argv: List[str], default_path: str = "newsletter.yml") -> str:
        """
        从传给 getmail.py 的参数中解析 --config 路径（支持 --config path 以及 --config=path），
        若未提供则回退到默认值。
        """
        i = 0
        while i < len(argv):
            a = argv[i]
            if a == "--config" and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith("--config="):
                return a.split("=", 1)[1]
            i += 1
        return default_path

    def load_proxy_from_yaml(cfg_path: Path) -> _Optional[str]:
        """
        优先使用 PyYAML 解析 proxy 字段；若不可用则做简易行扫描（仅识别顶层 'proxy: <value>'）。
        返回解析出的代理字符串或 None。
        """
        try:
            if yaml is not None and cfg_path.exists():
                data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict):
                    val = data.get("proxy")
                    if isinstance(val, str) and val.strip():
                        return val.strip()
            # 回退：简易行扫描
            if cfg_path.exists():
                for raw in cfg_path.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.lower().startswith("proxy:"):
                        v = line.split(":", 1)[1].strip()
                        # 去掉可能的引号
                        if (v.startswith("\"") and v.endswith("\"")) or (v.startswith("'") and v.endswith("'")):
                            v = v[1:-1].strip()
                        return v or None
        except Exception:
            # 解析失败则忽略
            pass
        return None

    def prepare_proxy_env(cfg_path: Path) -> None:
        proxy = load_proxy_from_yaml(cfg_path)
        if proxy:
            os.environ["HTTP_PROXY"] = proxy
            os.environ["HTTPS_PROXY"] = proxy
            os.environ["http_proxy"] = proxy
            os.environ["https_proxy"] = proxy
            print("=" * 80)
            print(f"[runloop] 已从 {cfg_path} 读取代理并设置环境变量：")
            print(f"[runloop] HTTP_PROXY/HTTPS_PROXY = {proxy}")
            print("=" * 80)

    cfg_path = Path(detect_config_path(rest, default_path="newsletter.yml"))
    prepare_proxy_env(cfg_path)

    print("[runloop] 配置：")
    print(f"  - 间隔: {args.interval} 秒")
    print(f"  - 命令: {base_cmd + rest}")
    print("[runloop] 按 Ctrl+C 可随时退出。")

    run_no = 0
    try:
        while True:
            run_no += 1
            print("=" * 80)
            print(f"[runloop] 第 {run_no} 次运行开始 @ {human_ts()}")
            rc = stream_process(base_cmd + rest, tag="getmail")
            print(f"[runloop] 第 {run_no} 次运行 getmail 结束，退出码 {rc}")
            # 紧接执行 readmail.py，使用默认参数
            rc_readmail = stream_process([sys.executable, "readmail.py"], tag="readmail")
            print(f"[runloop] 第 {run_no} 次运行 readmail 结束，退出码 {rc_readmail}")
            print("-" * 80)
            countdown(max(0, int(args.interval)))
    except KeyboardInterrupt:
        print("\n[runloop] 已收到中断信号，退出。")


if __name__ == "__main__":
    main()
