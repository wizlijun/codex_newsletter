#!/usr/bin/env python3
"""
读取 runtime/mails 下新增的邮件 .md 文件（或兼容 runtime/mail），根据 runtime/read/*.json 是否存在来判断是否已处理。
对每个新增的 id.md，调用本地 codex CLI，使用 readmailprompt.md 作为 prompt，将标准输出写入 runtime/read/id.json。

默认约定：
- 邮件目录优先使用 runtime/mails，若不存在则回退到 runtime/mail
- 结果输出到 runtime/read
- prompt 文件默认使用仓库根目录的 readmailprompt.md
- codex 调用：codex -p <prompt> -f <mail_md>（捕获 stdout 写入 json 文件）

可选参数可覆盖以上路径；支持 --dry-run 查看将要处理的项。
"""
from __future__ import annotations

import argparse
import os
import time
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Paths:
    repo_root: Path
    mail_dir: Path
    read_dir: Path
    prompt_file: Path


def detect_mail_dir(root: Path, override: Optional[str]) -> Path:
    if override:
        return (root / override).resolve()
    prefer = root / "runtime" / "mails"
    fallback = root / "runtime" / "mail"
    if prefer.exists():
        return prefer.resolve()
    return fallback.resolve()


def build_paths(
    mail_dir_arg: Optional[str], read_dir_arg: Optional[str], prompt_arg: Optional[str]
) -> Paths:
    repo_root = Path(__file__).resolve().parent
    # 将工作目录切到仓库根，确保相对路径稳定
    os.chdir(repo_root)

    mail_dir = detect_mail_dir(repo_root, mail_dir_arg)
    read_dir = (repo_root / (read_dir_arg or "runtime/read")).resolve()
    prompt_file = (repo_root / (prompt_arg or "readmailprompt.md")).resolve()
    return Paths(repo_root=repo_root, mail_dir=mail_dir, read_dir=read_dir, prompt_file=prompt_file)


def list_new_mail_ids(mail_dir: Path, read_dir: Path) -> List[str]:
    if not mail_dir.exists():
        return []
    md_files = [p for p in mail_dir.glob("*.md") if p.is_file()]

    pending: List[tuple[str, Path]] = []
    for p in md_files:
        mail_id = p.stem
        out_json = read_dir / f"{mail_id}.json"
        if not out_json.exists():
            pending.append((mail_id, p))

    def sort_key(item: tuple[str, Path]):
        stem, path = item
        if stem.isdigit():
            return (0, int(stem))
        # 次选：按文件修改时间（较早在前）
        try:
            return (1, path.stat().st_mtime)
        except Exception:
            return (1, float("inf"))

    pending.sort(key=sort_key)
    return [mail_id for mail_id, _ in pending]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_codex(
    prompt_file: Path,
    input_md: Path,
    output_json: Path,
    timeout: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> subprocess.CompletedProcess[str]:
    """
    使用 Codex CLI 的非交互模式，将合并后的 prompt+邮件内容通过 stdin 传入：
      codex exec --output-last-message <output_json> -

    说明：
    - 早期设计中的 `codex -f` 参数在当前 Codex CLI 中不可用；改为通过 stdin 提供完整输入。
    - 通过 `--output-last-message` 让 Codex 直接把最终消息写入目标 JSON 文件。
    - 依然返回 CompletedProcess，以便上层判断 returncode 与 stderr。
    """
    if shutil.which("codex") is None:
        raise FileNotFoundError("未找到 codex 可执行文件，请确保已安装并在 PATH 中可用。")

    prompt_text = prompt_file.read_text(encoding="utf-8")
    mail_text = input_md.read_text(encoding="utf-8")
    combined = f"{prompt_text}\n{mail_text}\n"

    # 确保输出目录存在，避免 Codex 无法写入
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--output-last-message",
        str(output_json),
        "-",  # 从 stdin 读取合并后的内容
    ]
    if extra_args:
        # 插入到 '-' 之前
        cmd = cmd[:-1] + list(extra_args) + cmd[-1:]

    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=combined,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 codex 处理新增邮件 Markdown 并生成 JSON")
    parser.add_argument("--mail-dir", default=None, help="邮件 .md 目录（默认优先 runtime/mails，其次 runtime/mail）")
    parser.add_argument("--read-dir", default="runtime/read", help="输出 JSON 目录（默认 runtime/read）")
    parser.add_argument("--prompt", default="readmailprompt.md", help="prompt 文件路径（默认仓库根的 readmailprompt.md）")
    parser.add_argument("--limit", type=int, default=None, help="仅处理最近 N 个新增邮件")
    parser.add_argument("--timeout", type=int, default=None, help="单封邮件处理超时时间（秒）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要处理的文件和命令，不执行")
    parser.add_argument(
        "--codex-arg",
        action="append",
        default=[],
        help="透传给 codex exec 的额外参数（可重复），例如 --codex-arg --model --codex-arg o4-mini",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出更多调试信息")
    args = parser.parse_args()

    paths = build_paths(args.mail_dir, args.read_dir, args.prompt)

    if args.verbose:
        print(f"[readmail] repo_root: {paths.repo_root}")
        print(f"[readmail] mail_dir:  {paths.mail_dir}")
        print(f"[readmail] read_dir:  {paths.read_dir}")
        print(f"[readmail] prompt:    {paths.prompt_file}")

    if not paths.mail_dir.exists():
        print(f"[readmail] 邮件目录不存在: {paths.mail_dir}", file=sys.stderr)
        sys.exit(1)

    if not paths.prompt_file.exists():
        print(f"[readmail] 缺少 prompt 文件: {paths.prompt_file}", file=sys.stderr)
        sys.exit(1)

    ensure_dir(paths.read_dir)

    pending_ids = list_new_mail_ids(paths.mail_dir, paths.read_dir)
    if args.limit is not None and args.limit > 0:
        pending_ids = pending_ids[-args.limit :]

    if not pending_ids:
        if args.verbose:
            print("[readmail] 没有发现新增邮件。")
        return

    print(f"[readmail] 待处理新增邮件: {len(pending_ids)} 条 -> {', '.join(pending_ids)}")

    if args.dry_run:
        # 打印即将执行的命令预览（新 CLI 方式）
        for mail_id in pending_ids:
            input_md = paths.mail_dir / f"{mail_id}.md"
            output_json = paths.read_dir / f"{mail_id}.json"
            extra = " ".join(args.codex_arg) if args.codex_arg else ""
            print(
                f"[dry-run] cat {paths.prompt_file} + {input_md} | codex exec --skip-git-repo-check {extra} --output-last-message {output_json} -"
            )
        return

    errors = 0
    for idx, mail_id in enumerate(pending_ids, start=1):
        input_md = paths.mail_dir / f"{mail_id}.md"
        output_json = paths.read_dir / f"{mail_id}.json"
        if args.verbose:
            print(f"[readmail] ({idx}/{len(pending_ids)}) 处理 {input_md.name} …")
        success = False
        last_err_msg = ""
        max_attempts = 3  # 首次 + 重试 2 次
        for attempt in range(1, max_attempts + 1):
            if args.verbose:
                retry_note = "" if attempt == 1 else f"（重试 {attempt - 1}/2）"
                print(f"[readmail] 调用 codex 处理 {input_md.name}{retry_note} …")
            try:
                proc = run_codex(
                    paths.prompt_file,
                    input_md,
                    output_json,
                    timeout=args.timeout,
                    extra_args=args.codex_arg or None,
                )
            except Exception as e:
                last_err_msg = f"异常: {e}"
                if args.verbose:
                    print(f"[readmail] 调用异常，{last_err_msg}", file=sys.stderr)
            else:
                if proc.returncode != 0:
                    stderr = (proc.stderr or "").strip()
                    last_err_msg = (
                        f"codex 非零退出码 {proc.returncode}，stderr=\n{stderr}"
                    )
                    if args.verbose:
                        print(f"[readmail] {last_err_msg}", file=sys.stderr)
                else:
                    # 成功返回，确认文件已生成
                    if output_json.exists():
                        success = True
                        break
                    else:
                        last_err_msg = "codex 成功返回但未生成预期输出文件"
                        if args.verbose:
                            print(f"[readmail] {last_err_msg}", file=sys.stderr)

            # 未成功且仍有机会时，稍等再重试
            if attempt < max_attempts:
                time.sleep(1)

        if success:
            if args.verbose:
                print(f"[readmail] 已生成: {output_json}")
            continue
        else:
            # 连续失败则跳过该邮件；确保没有残留的部分输出
            if output_json.exists():
                try:
                    output_json.unlink()
                except Exception:
                    pass
            errors += 1
            print(
                f"[readmail] 处理 {input_md.name} 失败（已重试 2 次），跳过生成 JSON。最后错误：{last_err_msg}",
                file=sys.stderr,
            )

    if errors:
        print(f"[readmail] 完成，存在 {errors} 个失败。", file=sys.stderr)
        sys.exit(2)
    else:
        print("[readmail] 全部完成。")


if __name__ == "__main__":
    main()
