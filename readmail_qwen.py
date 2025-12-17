#!/usr/bin/env python3
"""
读取 runtime/mails 下新增的邮件 .md 文件（或兼容 runtime/mail），根据 runtime/read/*.json 是否存在来判断是否已处理。
对每个新增的 id.md，调用 Qwen API，使用 readmailprompt.md 作为 prompt，将结果写入 runtime/read/id.json。

默认约定：
- 邮件目录优先使用 runtime/mails，若不存在则回退到 runtime/mail
- 结果输出到 runtime/read
- prompt 文件默认使用仓库根目录的 readmailprompt.md
- qwen API 调用：使用 OpenAI 兼容接口调用通义千问

可选参数可覆盖以上路径；支持 --dry-run 查看将要处理的项。
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
import sys
from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional

try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

from config_loader import config


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


def run_qwen_api(
    prompt_file: Path,
    input_md: Path,
    output_json: Path,
    timeout: Optional[int] = None,
    model: Optional[str] = None,
    max_retries: int = 2,
) -> tuple[bool, str, bool]:
    """
    使用 Qwen API 处理邮件，将合并后的 prompt+邮件内容发送给 Qwen。

    参数：
    - prompt_file: prompt 文件路径
    - input_md: 输入邮件 markdown 文件路径
    - output_json: 输出 JSON 文件路径
    - timeout: API 调用超时时间（秒）
    - model: 使用的模型名称，默认从配置读取
    - max_retries: 最大重试次数

    返回：
    - (成功标志, 错误信息, 是否安全拦截)
    """
    # 检查 OpenAI 库是否可用
    if not openai_available:
        return False, "OpenAI 库不可用，请安装: pip install openai", False

    # 检查配置
    if config is None:
        return False, "配置加载失败", False

    # 获取 Qwen 配置
    qwen_config = config.get_qwen_config()
    api_key = qwen_config.get('api_key', '')
    base_url = qwen_config.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    default_model = qwen_config.get('model', 'qwen-turbo')
    config_timeout = qwen_config.get('timeout', 120)

    if not api_key:
        return False, "Qwen API Key 未配置", False

    # 使用指定的模型或配置中的默认模型
    use_model = model or default_model
    use_timeout = timeout or config_timeout

    # 读取 prompt 和邮件内容
    try:
        prompt_text = prompt_file.read_text(encoding="utf-8")
        mail_text = input_md.read_text(encoding="utf-8")
        combined = f"{prompt_text}\n\n{mail_text}"
    except Exception as e:
        return False, f"读取文件失败: {e}", False

    # 确保输出目录存在
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # 调用 Qwen API
    for attempt in range(max_retries + 1):
        try:
            # 创建 OpenAI 客户端
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=use_timeout
            )

            # 构建消息
            messages = [
                {"role": "user", "content": combined}
            ]

            # 发送请求
            completion = client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=0.7,
                max_tokens=4000,
            )

            # 提取响应内容
            if completion.choices and len(completion.choices) > 0:
                response_content = completion.choices[0].message.content.strip()

                if response_content:
                    # 写入输出文件
                    output_json.write_text(response_content, encoding="utf-8")
                    return True, "", False
                else:
                    if attempt < max_retries:
                        time.sleep(1)
                        continue
                    return False, "Qwen 返回空响应", False
            else:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                return False, "Qwen API 响应为空", False

        except Exception as e:
            error_msg = str(e)

            # 检测是否为安全拦截错误
            is_content_inspection = False
            if "data_inspection_failed" in error_msg or "inappropriate content" in error_msg.lower():
                is_content_inspection = True
                # 安全拦截错误不重试，直接返回
                return False, f"Qwen API 安全拦截: {error_msg}", True

            # 非安全拦截错误，继续重试
            if attempt < max_retries:
                time.sleep(1)
                continue
            return False, f"Qwen API 调用异常: {error_msg}", False

    return False, "所有重试都失败了", False


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Qwen API 处理新增邮件 Markdown 并生成 JSON")
    parser.add_argument("--mail-dir", default=None, help="邮件 .md 目录（默认优先 runtime/mails，其次 runtime/mail）")
    parser.add_argument("--read-dir", default="runtime/read", help="输出 JSON 目录（默认 runtime/read）")
    parser.add_argument("--prompt", default="readmailprompt.md", help="prompt 文件路径（默认仓库根的 readmailprompt.md）")
    parser.add_argument("--limit", type=int, default=None, help="仅处理最近 N 个新增邮件")
    parser.add_argument("--timeout", type=int, default=None, help="单封邮件处理超时时间（秒）")
    parser.add_argument("--model", default=None, help="指定使用的 Qwen 模型（默认使用配置文件中的 model）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要处理的文件，不执行")
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

    # 确保错误目录存在
    err_dir = paths.mail_dir.parent / "err"
    ensure_dir(err_dir)

    pending_ids = list_new_mail_ids(paths.mail_dir, paths.read_dir)
    if args.limit is not None and args.limit > 0:
        pending_ids = pending_ids[-args.limit :]

    if not pending_ids:
        if args.verbose:
            print("[readmail] 没有发现新增邮件。")
        return

    print(f"[readmail] 待处理新增邮件: {len(pending_ids)} 条 -> {', '.join(pending_ids)}")

    if args.dry_run:
        # 打印即将执行的处理预览
        for mail_id in pending_ids:
            input_md = paths.mail_dir / f"{mail_id}.md"
            output_json = paths.read_dir / f"{mail_id}.json"
            print(
                f"[dry-run] 将调用 Qwen API 处理: {input_md} -> {output_json}"
            )
        return

    errors = 0
    blocked_by_safety = 0
    for idx, mail_id in enumerate(pending_ids, start=1):
        input_md = paths.mail_dir / f"{mail_id}.md"
        output_json = paths.read_dir / f"{mail_id}.json"
        if args.verbose:
            print(f"[readmail] ({idx}/{len(pending_ids)}) 处理 {input_md.name} …")

        # 调用 Qwen API（内部已包含重试逻辑）
        success, error_msg, is_safety_blocked = run_qwen_api(
            paths.prompt_file,
            input_md,
            output_json,
            timeout=args.timeout,
            model=args.model,
            max_retries=2,  # API 内部重试 2 次
        )

        if success:
            # 生成成功后校验并清理可能的 Markdown 代码围栏
            try:
                raw = output_json.read_text(encoding="utf-8")
                raw_stripped = raw.lstrip()
                # 若已是合法 JSON，则无需处理
                try:
                    json.loads(raw)
                except Exception:
                    # 若以 ``` 开头（常见为 ```json\n...\n```），则仅保留 [ 到 ] 的部分
                    if raw_stripped.startswith("```"):
                        start = raw.find("[")
                        end = raw.rfind("]")
                        if start != -1 and end != -1 and start < end:
                            trimmed = raw[start : end + 1]
                            try:
                                json.loads(trimmed)
                            except Exception:
                                # 裁剪后仍非合法 JSON，保留原文件，仅提示
                                if args.verbose:
                                    print(
                                        f"[readmail] 警告: 代码围栏裁剪后 JSON 仍不合法，未修改: {output_json}",
                                        file=sys.stderr,
                                    )
                            else:
                                output_json.write_text(trimmed, encoding="utf-8")
                                if args.verbose:
                                    print(
                                        f"[readmail] 检测到围栏代码块，已裁剪为纯 JSON 数组: {output_json}"
                                    )
                    # 非 ``` 开头则不做特殊处理，只在 verbose 下提示
                    else:
                        if args.verbose:
                            print(
                                f"[readmail] 警告: 输出看起来不是合法 JSON，但未检测到 ``` 开头，未修改: {output_json}",
                                file=sys.stderr,
                            )
            except Exception as e:
                if args.verbose:
                    print(f"[readmail] 后处理检查失败: {e}", file=sys.stderr)

            # 检查生成的文件是否为空
            try:
                size = output_json.stat().st_size
                if size == 0:
                    # 生成了 0 字节文件，删除并标记为失败
                    output_json.unlink()
                    errors += 1
                    print(
                        f"[readmail] 处理 {input_md.name} 失败：生成了空文件（0 字节），已删除。",
                        file=sys.stderr,
                    )
                else:
                    if args.verbose:
                        print(f"[readmail] 已生成: {output_json}")
            except Exception:
                errors += 1
                print(
                    f"[readmail] 处理 {input_md.name} 失败：无法检查输出文件。",
                    file=sys.stderr,
                )
        else:
            # 调用失败，清理可能的残留文件
            if output_json.exists():
                try:
                    output_json.unlink()
                except Exception:
                    pass

            # 检查是否为安全拦截
            if is_safety_blocked:
                # 安全拦截：将邮件移动到 err 目录
                try:
                    err_file = err_dir / input_md.name
                    # 如果 err 目录中已存在同名文件，先删除
                    if err_file.exists():
                        err_file.unlink()
                    # 移动文件
                    shutil.move(str(input_md), str(err_file))
                    blocked_by_safety += 1
                    print(
                        f"[readmail] 邮件 {input_md.name} 被安全拦截，已移动到 {err_dir}/",
                        file=sys.stderr,
                    )
                except Exception as e:
                    errors += 1
                    print(
                        f"[readmail] 邮件 {input_md.name} 被安全拦截，但移动到 err 目录失败: {e}",
                        file=sys.stderr,
                    )
            else:
                # 其他错误
                errors += 1
                print(
                    f"[readmail] 处理 {input_md.name} 失败（已重试 2 次），跳过生成 JSON。最后错误：{error_msg}",
                    file=sys.stderr,
                )

    # 输出最终统计
    if errors or blocked_by_safety:
        summary_parts = []
        if errors:
            summary_parts.append(f"{errors} 个失败")
        if blocked_by_safety:
            summary_parts.append(f"{blocked_by_safety} 个被安全拦截并移至 err/")
        print(f"[readmail] 完成，{', '.join(summary_parts)}。", file=sys.stderr)
        if errors:
            sys.exit(2)
    else:
        print("[readmail] 全部完成。")


if __name__ == "__main__":
    main()
