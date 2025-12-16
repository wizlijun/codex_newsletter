#!/usr/bin/env python3
"""
快速运行提示：
- pip install pyyaml imapclient html2text python-dotenv
- 准备 newsletter.yml（见题述配置示例）
- 在环境中导出 GMAIL_APP_PASSWORD（建议使用 Gmail 应用专用密码）
- python getmail.py --config newsletter.yml
"""
from __future__ import annotations

import argparse
import email
import email.policy
import email.utils
import json
import logging
import os
import re
import shlex
import ssl
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

# Optional deps
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from imapclient import IMAPClient  # type: ignore
except Exception:
    IMAPClient = None  # type: ignore

try:
    import html2text  # type: ignore
except Exception:
    html2text = None  # type: ignore

try:
    import socks  # type: ignore
except Exception:
    socks = None  # type: ignore

import imaplib
import yaml


# ----------------------------- Configuration -----------------------------


@dataclass
class ImapSettings:
    host: str
    port: int
    ssl: bool
    mailbox: str


@dataclass
class ProxySettings:
    enabled: bool
    proxy_type: str  # http, socks4, socks5
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass
class EmailConfig:
    provider: str
    username: str
    password: str
    imap: ImapSettings


@dataclass
class RuntimeConfig:
    out_dir: str
    state_file: str


@dataclass
class FetchConfig:
    batch_size: int = 200


@dataclass
class AppConfig:
    email: EmailConfig
    runtime: RuntimeConfig
    fetch: FetchConfig
    proxy: Optional[ProxySettings] = None


def load_config(path: Path, override_out: Optional[str], override_mailbox: Optional[str], override_batch: Optional[int]) -> AppConfig:
    """
    Load and validate configuration from YAML. Apply CLI overrides.
    """
    if load_dotenv:
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(dotenv_path=str(env_path))

    if not path.exists():
        print(f"ERROR: Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"ERROR: Failed to parse YAML config: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        email_node = data["email"]
        provider = str(email_node.get("provider", "")).strip().lower()
        if provider != "gmail":
            raise ValueError("email.provider must be 'gmail'")

        username = str(email_node["username"]).strip()
        password = str(email_node["password"]).strip()
        imap_node = email_node["imap"]
        host = str(imap_node.get("host", "imap.gmail.com")).strip()
        port = int(imap_node.get("port", 993))
        ssl_enabled = bool(imap_node.get("ssl", True))
        mailbox = str(imap_node.get("mailbox", "[Gmail]/All Mail"))

        runtime_node = data["runtime"]
        out_dir = str(runtime_node.get("out_dir", "runtime/mails"))
        state_file = str(runtime_node.get("state_file", "runtime/state.json"))

        fetch_node = data.get("fetch", {}) or {}
        batch_size = int(fetch_node.get("batch_size", 200))

    except KeyError as e:
        print(f"ERROR: Missing required config key: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Invalid configuration: {e}", file=sys.stderr)
        sys.exit(1)

    if override_out:
        out_dir = override_out
    if override_mailbox:
        mailbox = override_mailbox
    if override_batch:
        batch_size = override_batch

    # Validate required fields
    if not username:
        print("ERROR: email.username is required", file=sys.stderr)
        sys.exit(1)
    if not password:
        print("ERROR: email.password is required", file=sys.stderr)
        sys.exit(1)

    # Parse proxy settings
    proxy_settings = None
    proxy_node = data.get("proxy")
    if proxy_node:
        try:
            if isinstance(proxy_node, str):
                # Parse proxy URL like http://127.0.0.1:1087 or socks5://user:pass@host:port
                parsed = urlparse(proxy_node)
                proxy_type = parsed.scheme.lower() if parsed.scheme else "http"
                proxy_host = parsed.hostname or "127.0.0.1"
                proxy_port = parsed.port or 1080
                proxy_username = parsed.username
                proxy_password = parsed.password

                # Map http/https to http proxy type
                if proxy_type in ("http", "https"):
                    proxy_type = "http"
                elif proxy_type.startswith("socks"):
                    proxy_type = proxy_type  # socks4, socks5, socks5h

                proxy_settings = ProxySettings(
                    enabled=True,
                    proxy_type=proxy_type,
                    host=proxy_host,
                    port=proxy_port,
                    username=proxy_username,
                    password=proxy_password,
                )
            elif isinstance(proxy_node, dict):
                # Parse proxy dict
                proxy_settings = ProxySettings(
                    enabled=bool(proxy_node.get("enabled", True)),
                    proxy_type=str(proxy_node.get("type", "http")).lower(),
                    host=str(proxy_node.get("host", "127.0.0.1")),
                    port=int(proxy_node.get("port", 1080)),
                    username=proxy_node.get("username"),
                    password=proxy_node.get("password"),
                )
        except Exception as e:
            logging.warning("Failed to parse proxy settings: %s", e)
            proxy_settings = None

    cfg = AppConfig(
        email=EmailConfig(
            provider=provider,
            username=username,
            password=password,
            imap=ImapSettings(host=host, port=port, ssl=ssl_enabled, mailbox=mailbox),
        ),
        runtime=RuntimeConfig(out_dir=out_dir, state_file=state_file),
        fetch=FetchConfig(batch_size=batch_size),
        proxy=proxy_settings,
    )
    return cfg


# ----------------------------- State Store -----------------------------


class StateStore:
    """
    Simple JSON state store to track last processed UID per account/mailbox key.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, int] = {}

    def load(self) -> None:
        if not self.path.exists():
            self.data = {}
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, dict):
                    # Coerce to int values
                    self.data = {str(k): int(v) for k, v in obj.items()}
                else:
                    self.data = {}
        except Exception as e:
            logging.error("Failed to read state file %s: %s", self.path, e)
            self.data = {}

    def get_last_uid(self, key: str) -> int:
        return int(self.data.get(key, 0))

    def set_last_uid(self, key: str, uid: int) -> None:
        prev = self.data.get(key, 0)
        if uid > int(prev):
            self.data[key] = int(uid)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error("Failed to write state file %s: %s", self.path, e)


# ----------------------------- Utilities -----------------------------


def decode_mime_words(value: Optional[str]) -> str:
    """
    Decode RFC2047 encoded words in headers like Subject, Name.
    """
    if not value:
        return ""
    try:
        parts = email.header.decode_header(value)
        decoded = []
        for text, enc in parts:
            if isinstance(text, bytes):
                try:
                    decoded.append(text.decode(enc or "utf-8", errors="replace"))
                except Exception:
                    decoded.append(text.decode("utf-8", errors="replace"))
            else:
                decoded.append(text)
        return "".join(decoded)
    except Exception:
        return value


def parse_addresses(value: Optional[str]) -> List[str]:
    """
    Return addresses in 'Name <email>' format, decoded.
    """
    if not value:
        return []
    addrs = email.utils.getaddresses([value])
    formatted = []
    for name, addr in addrs:
        name_dec = decode_mime_words(name).strip()
        if addr:
            if name_dec:
                formatted.append(f"{name_dec} <{addr}>")
            else:
                formatted.append(addr)
        elif name_dec:
            formatted.append(name_dec)
    # Deduplicate preserving order
    seen = set()
    result = []
    for a in formatted:
        if a not in seen:
            seen.add(a)
            result.append(a)
    return result


def parse_date_to_iso(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(value)
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return None


def best_effort_text_from_message(msg: email.message.Message) -> Tuple[str, List[str]]:
    """
    Extract text body preferring text/plain; fallback to html -> markdown.
    Returns (text, attachments_list)
    """
    attachments: List[str] = []

    def is_attachment(part: email.message.Message) -> bool:
        cd = part.get_content_disposition()
        if cd == "attachment":
            return True
        # Some emails mark inline with filename as attachments
        if cd == "inline" and part.get_filename():
            return True
        return False

    text_plain_candidates: List[str] = []
    text_html_candidates: List[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            if is_attachment(part):
                fn = part.get_filename()
                if fn:
                    attachments.append(decode_mime_words(fn))
                continue
            ctype = part.get_content_type()
            try:
                payload = part.get_content()
            except Exception:
                try:
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        payload = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                except Exception:
                    payload = ""
            if ctype == "text/plain":
                if isinstance(payload, str):
                    text_plain_candidates.append(payload)
            elif ctype == "text/html":
                if isinstance(payload, str):
                    text_html_candidates.append(payload)
    else:
        ctype = msg.get_content_type()
        if ctype == "text/plain":
            try:
                payload = msg.get_content()
                text_plain_candidates.append(payload if isinstance(payload, str) else "")
            except Exception:
                try:
                    payload = msg.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        text_plain_candidates.append(payload.decode(msg.get_content_charset() or "utf-8", errors="replace"))
                except Exception:
                    pass
        elif ctype == "text/html":
            try:
                payload = msg.get_content()
                text_html_candidates.append(payload if isinstance(payload, str) else "")
            except Exception:
                try:
                    payload = msg.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        text_html_candidates.append(payload.decode(msg.get_content_charset() or "utf-8", errors="replace"))
                except Exception:
                    pass

    body = ""
    if text_plain_candidates:
        body = "\n\n".join(text_plain_candidates).strip()
    elif text_html_candidates:
        html = "\n\n".join(text_html_candidates).strip()
        body = html_to_markdown(html)
    else:
        body = ""

    return body, attachments


def html_to_markdown(html: str) -> str:
    if html2text:
        try:
            h = html2text.HTML2Text()
            h.ignore_images = True
            h.ignore_links = False
            h.body_width = 0
            return h.handle(html)
        except Exception:
            pass
    # Fallback: naive tag strip + unescape
    try:
        import html as htmlmod  # stdlib
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", html)
        text = re.sub(r"(?is)<br\s*/?>", "\n", text)
        text = re.sub(r"(?is)</p\s*>", "\n\n", text)
        text = re.sub(r"(?is)<.*?>", "", text)
        text = htmlmod.unescape(text)
        return text.strip()
    except Exception:
        return html


def chunked(seq: Sequence[int], size: int) -> Iterable[List[int]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_yaml_dump(data: Mapping[str, Any]) -> str:
    # Ensure string values that might cause YAML parsing issues are quoted
    def clean_value(value):
        if isinstance(value, str):
            # If the string contains special characters that might confuse YAML, ensure it's treated as string
            if any(char in value for char in ['<', '>', '{', '}', '[', ']', ':', '@']):
                return value  # yaml.safe_dump will handle quoting
            return value
        elif isinstance(value, list):
            return [clean_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: clean_value(v) for k, v in value.items()}
        return value
    
    cleaned_data = clean_value(data)
    return yaml.safe_dump(cleaned_data, allow_unicode=True, sort_keys=False, default_flow_style=False)


def exponential_backoff_delays(retries: int) -> List[float]:
    # 1s, 2s, 4s, ...
    return [2 ** i for i in range(retries)]


# ----------------------------- IMAP Sessions -----------------------------


def create_proxy_socket(host: str, port: int, proxy: ProxySettings) -> Any:
    """
    Create a socket connection through a proxy.
    Requires PySocks library (pip install PySocks).
    """
    if socks is None:
        raise RuntimeError("PySocks library not available. Install it with: pip install PySocks")

    # Map proxy type to socks constants
    proxy_type_map = {
        "http": socks.HTTP,
        "socks4": socks.SOCKS4,
        "socks5": socks.SOCKS5,
    }

    proxy_type_value = proxy_type_map.get(proxy.proxy_type.lower())
    if proxy_type_value is None:
        raise ValueError(f"Unsupported proxy type: {proxy.proxy_type}")

    # Create socket with proxy
    sock = socks.socksocket()
    sock.set_proxy(
        proxy_type=proxy_type_value,
        addr=proxy.host,
        port=proxy.port,
        username=proxy.username,
        password=proxy.password,
    )
    sock.connect((host, port))
    return sock


class BaseIMAPSession:
    def __init__(self, username: str, password: str, host: str, port: int, use_ssl: bool, mailbox: str, proxy: Optional[ProxySettings] = None) -> None:
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.mailbox = mailbox
        self.proxy = proxy

    def connect(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def select_mailbox(self, mailbox: Optional[str] = None) -> str:
        raise NotImplementedError

    def search_uids_greater_than(self, last_uid: int) -> List[int]:
        raise NotImplementedError

    def fetch_messages(self, uids: Sequence[int]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class IMAPClientSession(BaseIMAPSession):
    def __init__(self, username: str, password: str, host: str, port: int, use_ssl: bool, mailbox: str, proxy: Optional[ProxySettings] = None) -> None:
        super().__init__(username, password, host, port, use_ssl, mailbox, proxy)
        self.client: Optional[IMAPClient] = None  # type: ignore
        self._proxy_set = False

    def _setup_proxy(self) -> None:
        """Setup global proxy for socks module."""
        if not self.proxy or not self.proxy.enabled or self._proxy_set:
            return

        if socks is None:
            raise RuntimeError("PySocks library not available. Install it with: pip install PySocks")

        # Map proxy type to socks constants
        proxy_type_map = {
            "http": socks.HTTP,
            "socks4": socks.SOCKS4,
            "socks5": socks.SOCKS5,
        }

        proxy_type_value = proxy_type_map.get(self.proxy.proxy_type.lower())
        if proxy_type_value is None:
            raise ValueError(f"Unsupported proxy type: {self.proxy.proxy_type}")

        logging.info("Setting up %s proxy %s:%s for socket connections", self.proxy.proxy_type, self.proxy.host, self.proxy.port)

        # Set default proxy for all socket connections
        socks.set_default_proxy(
            proxy_type=proxy_type_value,
            addr=self.proxy.host,
            port=self.proxy.port,
            username=self.proxy.username,
            password=self.proxy.password,
        )
        # Replace socket module's socket with socksocket
        import socket as socket_module
        socket_module.socket = socks.socksocket
        self._proxy_set = True

    def connect(self) -> None:
        assert IMAPClient is not None

        # Setup proxy if configured
        self._setup_proxy()

        ssl_context = ssl.create_default_context()

        if self.proxy and self.proxy.enabled:
            logging.info("Connecting to %s:%s via %s proxy %s:%s", self.host, self.port, self.proxy.proxy_type, self.proxy.host, self.proxy.port)

        # Now create IMAPClient normally - it will use the proxied socket
        self.client = IMAPClient(self.host, port=self.port, ssl=self.use_ssl, ssl_context=ssl_context)
        self.client.login(self.username, self.password)
        self.select_mailbox(self.mailbox)

    def close(self) -> None:
        try:
            if self.client is not None:
                self.client.logout()
        except Exception:
            pass
        self.client = None

    def _detect_all_mail_folder(self) -> Optional[str]:
        """Attempt to find Gmail's All Mail folder using special-use flags."""
        assert self.client is not None
        try:
            folders = self.client.list_folders()
        except Exception:
            return None
        for flags, delim, name in folders:
            # Normalize flags to strings
            norm_flags = []
            for f in (flags or []):
                try:
                    norm_flags.append(f.decode() if isinstance(f, (bytes, bytearray)) else str(f))
                except Exception:
                    norm_flags.append(str(f))
            if any(flag.upper() == r"\ALL" for flag in norm_flags):
                return name
        return None

    def select_mailbox(self, mailbox: Optional[str] = None) -> str:
        assert self.client is not None
        target = mailbox or self.mailbox
        try:
            self.client.select_folder(target, readonly=True)
            self.mailbox = target
            return target
        except Exception:
            # Try auto-detecting All Mail if target looks like All Mail
            looks_all_mail = bool(target) and ("all mail" in target.lower() or target.strip() in ("[Gmail]/All Mail", "[Google Mail]/All Mail"))
            if looks_all_mail:
                detected = self._detect_all_mail_folder()
                if detected:
                    try:
                        self.client.select_folder(detected, readonly=True)
                        self.mailbox = detected
                        logging.info("Selected detected All Mail folder: %s", detected)
                        return detected
                    except Exception:
                        pass
            logging.warning("Failed to select mailbox '%s'. Falling back to INBOX.", target)
            self.client.select_folder("INBOX", readonly=True)
            self.mailbox = "INBOX"
            return "INBOX"

    def _get(self, d: Mapping[Any, Any], key: str) -> Any:
        return d.get(key) or d.get(key.encode())

    def search_uids_greater_than(self, last_uid: int) -> List[int]:
        assert self.client is not None
        min_uid = last_uid + 1 if last_uid > 0 else 1
        try:
            uids = self.client.search(["UID", f"{min_uid}:*"])
        except Exception:
            # Fallback to ALL then filter
            uids = self.client.search("ALL")
            uids = [u for u in uids if int(u) > last_uid]
        uids = [int(u) for u in uids]
        uids.sort()
        return uids

    def fetch_messages(self, uids: Sequence[int]) -> List[Dict[str, Any]]:
        assert self.client is not None
        if not uids:
            return []
        # Try with Gmail extensions
        attrs = ["RFC822", "RFC822.SIZE", "X-GM-LABELS", "X-GM-THRID"]
        # IMAPClient handles batching internally for list argument
        res = self.client.fetch(list(uids), attrs)
        out: List[Dict[str, Any]] = []
        for uid in uids:
            data = res.get(uid) or res.get(int(uid)) or res.get(str(uid))
            if not data:
                continue
            raw = self._get(data, "RFC822") or self._get(data, "BODY[]")
            size = self._get(data, "RFC822.SIZE")
            labels = self._get(data, "X-GM-LABELS")
            thrid = self._get(data, "X-GM-THRID")
            # Normalize
            if isinstance(labels, (list, tuple)):
                labels_list = [l.decode() if isinstance(l, bytes) else str(l) for l in labels]
            elif isinstance(labels, bytes):
                labels_list = [labels.decode(errors="replace")]
            elif labels is None:
                labels_list = None
            else:
                labels_list = [str(labels)]
            if isinstance(thrid, bytes):
                thrid_str = thrid.decode(errors="replace")
            elif thrid is None:
                thrid_str = None
            else:
                thrid_str = str(thrid)
            if isinstance(size, bytes):
                try:
                    size_val = int(size.decode())
                except Exception:
                    size_val = None
            else:
                try:
                    size_val = int(size) if size is not None else None
                except Exception:
                    size_val = None
            if isinstance(raw, bytes):
                raw_bytes = raw
            else:
                raw_bytes = raw if isinstance(raw, (bytes, bytearray)) else b""
            out.append(
                {
                    "uid": int(uid),
                    "raw": bytes(raw_bytes),
                    "size": size_val if size_val is not None else len(raw_bytes),
                    "labels": labels_list,
                    "thread_id": thrid_str,
                }
            )
        return out


class ImaplibSession(BaseIMAPSession):
    def __init__(self, username: str, password: str, host: str, port: int, use_ssl: bool, mailbox: str, proxy: Optional[ProxySettings] = None) -> None:
        super().__init__(username, password, host, port, use_ssl, mailbox, proxy)
        self.conn: Optional[imaplib.IMAP4] = None
        self._proxy_set = False

    def _setup_proxy(self) -> None:
        """Setup global proxy for socks module."""
        if not self.proxy or not self.proxy.enabled or self._proxy_set:
            return

        if socks is None:
            raise RuntimeError("PySocks library not available. Install it with: pip install PySocks")

        # Map proxy type to socks constants
        proxy_type_map = {
            "http": socks.HTTP,
            "socks4": socks.SOCKS4,
            "socks5": socks.SOCKS5,
        }

        proxy_type_value = proxy_type_map.get(self.proxy.proxy_type.lower())
        if proxy_type_value is None:
            raise ValueError(f"Unsupported proxy type: {self.proxy.proxy_type}")

        logging.info("Setting up %s proxy %s:%s for socket connections", self.proxy.proxy_type, self.proxy.host, self.proxy.port)

        # Set default proxy for all socket connections
        socks.set_default_proxy(
            proxy_type=proxy_type_value,
            addr=self.proxy.host,
            port=self.proxy.port,
            username=self.proxy.username,
            password=self.proxy.password,
        )
        # Replace socket module's socket with socksocket
        import socket as socket_module
        socket_module.socket = socks.socksocket
        self._proxy_set = True

    def connect(self) -> None:
        # Setup proxy if configured
        self._setup_proxy()

        if self.proxy and self.proxy.enabled:
            logging.info("Connecting to %s:%s via %s proxy %s:%s", self.host, self.port, self.proxy.proxy_type, self.proxy.host, self.proxy.port)

        # Create IMAP connection - it will use the proxied socket
        if self.use_ssl:
            self.conn = imaplib.IMAP4_SSL(self.host, self.port)
        else:
            self.conn = imaplib.IMAP4(self.host, self.port)

        typ, data = self.conn.login(self.username, self.password)  # type: ignore[union-attr]
        if typ != "OK":
            raise RuntimeError("Login failed")
        self.select_mailbox(self.mailbox)

    def close(self) -> None:
        try:
            if self.conn is not None:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn.logout()
        except Exception:
            pass
        self.conn = None

    def _detect_all_mail_folder(self) -> Optional[str]:
        assert self.conn is not None
        try:
            typ, data = self.conn.list()
            if typ != "OK" or not data:
                return None
        except Exception:
            return None
        for line in data:
            try:
                text = line.decode() if isinstance(line, (bytes, bytearray)) else str(line)
            except Exception:
                text = str(line)
            # Example: (\\HasNoChildren \\All) "/" "[Gmail]/All Mail"
            # Extract flags and mailbox name
            m = re.search(r"\((?P<flags>[^)]*)\)\s+\"(?P<delim>.*?)\"\s+(?P<name>.*)$", text)
            if not m:
                continue
            flags_txt = m.group("flags") or ""
            name_txt = m.group("name") or ""
            # name may be quoted; strip quotes
            name_txt = name_txt.strip()
            if name_txt.startswith('"') and name_txt.endswith('"'):
                name_txt = name_txt[1:-1]
            flags = [f.strip() for f in flags_txt.split()] if flags_txt else []
            if any(f.upper() == r"\ALL" for f in flags):
                return name_txt
        return None

    def select_mailbox(self, mailbox: Optional[str] = None) -> str:
        assert self.conn is not None
        target = mailbox or self.mailbox
        typ, _ = self.conn.select(target, readonly=True)
        if typ != "OK":
            # Try auto-detecting All Mail if target looks like All Mail
            looks_all_mail = bool(target) and ("all mail" in target.lower() or target.strip() in ("[Gmail]/All Mail", "[Google Mail]/All Mail"))
            if looks_all_mail:
                detected = self._detect_all_mail_folder()
                if detected:
                    typ2, _ = self.conn.select(detected, readonly=True)
                    if typ2 == "OK":
                        self.mailbox = detected
                        logging.info("Selected detected All Mail folder: %s", detected)
                        return detected
            logging.warning("Failed to select mailbox '%s'. Falling back to INBOX.", target)
            typ, _ = self.conn.select("INBOX", readonly=True)
            if typ != "OK":
                raise RuntimeError("Cannot select INBOX")
            self.mailbox = "INBOX"
            return "INBOX"
        self.mailbox = target
        return target

    def search_uids_greater_than(self, last_uid: int) -> List[int]:
        assert self.conn is not None
        min_uid = last_uid + 1 if last_uid > 0 else 1
        query = f"UID {min_uid}:*"
        typ, data = self.conn.uid("SEARCH", None, query)
        if typ != "OK":
            # Fallback to ALL then filter
            typ, data = self.conn.uid("SEARCH", None, "ALL")
            if typ != "OK":
                raise RuntimeError("UID SEARCH failed")
        if not data or not data[0]:
            return []
        # data[0] is a space-separated bytes list of UIDs
        try:
            uids = [int(x) for x in data[0].split()]
        except Exception:
            uids = []
        if last_uid > 0:
            uids = [u for u in uids if u > last_uid]
        uids.sort()
        return uids

    def _fetch_raw(self, uid: int) -> bytes:
        assert self.conn is not None
        typ, data = self.conn.uid("FETCH", str(uid), "(RFC822)")
        if typ != "OK" or not data:
            raise RuntimeError(f"FETCH RFC822 failed for UID {uid}")
        raw = b""
        for part in data:
            if isinstance(part, tuple) and len(part) >= 2 and isinstance(part[1], (bytes, bytearray)):
                raw = part[1]
                break
        if not raw and isinstance(data[0], (bytes, bytearray)):
            # Sometimes data[0] is literal?
            raw = bytes(data[0])
        return bytes(raw)

    def _fetch_meta(self, uid: int) -> Tuple[Optional[int], Optional[List[str]], Optional[str]]:
        assert self.conn is not None
        # Fetch size, labels, thread id
        typ, data = self.conn.uid("FETCH", str(uid), "(RFC822.SIZE X-GM-LABELS X-GM-THRID)")
        if typ != "OK" or not data:
            return None, None, None
        size_val: Optional[int] = None
        labels_list: Optional[List[str]] = None
        thrid_str: Optional[str] = None
        # data can be [(b'1 (UID 1 ...', None)] or similar
        blob = b""
        for part in data:
            if isinstance(part, (bytes, bytearray)):
                blob += bytes(part)
            elif isinstance(part, tuple) and len(part) >= 1 and isinstance(part[0], (bytes, bytearray)):
                blob += bytes(part[0])
        text = blob.decode("utf-8", errors="ignore")
        # Extract RFC822.SIZE
        m = re.search(r"RFC822\.SIZE\s+(\d+)", text)
        if m:
            try:
                size_val = int(m.group(1))
            except Exception:
                size_val = None
        # Extract X-GM-THRID
        m = re.search(r"X-GM-THRID\s+(\d+)", text)
        if m:
            thrid_str = m.group(1)
        # Extract X-GM-LABELS (content within first parentheses after label)
        m = re.search(r"X-GM-LABELS\s+\((.*?)\)", text, flags=re.DOTALL)
        if m:
            content = m.group(1).strip()
            if content:
                try:
                    tokens = shlex.split(content)
                    labels_list = [t for t in tokens if t]
                except Exception:
                    # Fallback: split by space and strip quotes
                    raw_tokens = content.split()
                    labels_list = [t.strip("\"'") for t in raw_tokens if t.strip()]
        return size_val, labels_list, thrid_str

    def fetch_messages(self, uids: Sequence[int]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for uid in uids:
            try:
                raw = self._fetch_raw(uid)
                size_val, labels_list, thrid_str = self._fetch_meta(uid)
                if size_val is None:
                    size_val = len(raw)
                out.append(
                    {
                        "uid": int(uid),
                        "raw": raw,
                        "size": int(size_val),
                        "labels": labels_list,
                        "thread_id": thrid_str,
                    }
                )
            except Exception as e:
                logging.error("Fetch failed for UID %s: %s", uid, e)
        return out


def build_session(cfg: AppConfig) -> BaseIMAPSession:
    proxy = cfg.proxy if cfg.proxy and cfg.proxy.enabled else None
    if IMAPClient is not None:
        return IMAPClientSession(
            cfg.email.username,
            cfg.email.password,
            cfg.email.imap.host,
            cfg.email.imap.port,
            cfg.email.imap.ssl,
            cfg.email.imap.mailbox,
            proxy,
        )
    else:
        return ImaplibSession(
            cfg.email.username,
            cfg.email.password,
            cfg.email.imap.host,
            cfg.email.imap.port,
            cfg.email.imap.ssl,
            cfg.email.imap.mailbox,
            proxy,
        )


# ----------------------------- Core Processing -----------------------------


def process_message_record(
    record: Dict[str, Any],
    out_dir: Path,
    force_overwrite: bool = False,
) -> Tuple[bool, Optional[int]]:
    """
    Parse and write markdown file for a single message record.
    Returns (written_or_skipped_exists, uid).
    """
    uid = int(record.get("uid", 0))
    raw: bytes = record.get("raw", b"")
    if not uid or not raw:
        return (False, None)
    file_path = out_dir / f"{uid}.md"
    if file_path.exists() and not force_overwrite:
        # Idempotent: skip writing unless force_overwrite is True
        return (True, uid)
    try:
        msg = email.message_from_bytes(raw, policy=email.policy.default)
    except Exception:
        # Fallback to compat32
        msg = email.message_from_bytes(raw)

    subject = decode_mime_words(msg.get("Subject"))
    message_id_raw = msg.get("Message-ID") or msg.get("Message-Id") or msg.get("Message-Id".lower()) or None
    # Ensure message_id is safe for YAML by converting to string and escaping if needed
    message_id = str(message_id_raw) if message_id_raw else ""
    date_iso = parse_date_to_iso(msg.get("Date"))
    from_list = parse_addresses(msg.get("From"))
    from_str = from_list[0] if from_list else None
    to_list = parse_addresses(msg.get("To"))
    cc_list = parse_addresses(msg.get("Cc") or msg.get("CC"))
    size = int(record.get("size") or len(raw))
    labels = record.get("labels", None)
    thrid = record.get("thread_id", None)

    body, attachments = best_effort_text_from_message(msg)

    front_matter: Dict[str, Any] = {
        "uid": uid,
        "message_id": message_id,
        "subject": subject if subject else "",
        "date": date_iso if date_iso else "",
        "from": from_str if from_str else "",
        "size": size,
    }
    if to_list:
        front_matter["to"] = to_list
    if cc_list:
        front_matter["cc"] = cc_list
    if labels:
        front_matter["labels"] = labels
    if thrid:
        front_matter["thread_id"] = str(thrid)
    if attachments:
        front_matter["attachments"] = attachments

    ensure_dir(out_dir)
    try:
        with file_path.open("w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(safe_yaml_dump(front_matter))
            f.write("---\n")
            f.write(body or "")
        return (True, uid)
    except Exception as e:
        logging.error("Failed to write file %s: %s", file_path, e)
        return (False, None)


def with_retries(fn, *, retries: int = 3, delay_base: float = 1.0):
    delays = exponential_backoff_delays(retries)
    last_exc = None
    for attempt, d in enumerate([0.0] + delays):
        if d > 0:
            time.sleep(d * delay_base)
        try:
            return fn()
        except Exception as e:
            last_exc = e
            logging.warning("Operation failed (attempt %d/%d): %s", attempt + 1, retries + 1, e)
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("Operation failed after retries")


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally fetch Gmail emails and save as Markdown with YAML front matter.")
    parser.add_argument("--config", default="newsletter.yml", help="Path to config YAML (default: ./newsletter.yml)")
    parser.add_argument("--out", default=None, help="Override runtime.out_dir")
    parser.add_argument("--mailbox", default=None, help="Override email.mailbox")
    parser.add_argument("--full-rescan", action="store_true", help="Ignore saved state and scan from 0 (existing files are not overwritten)")
    parser.add_argument("--redownload-all", action="store_true", help="Force redownload all emails, overwriting existing files")
    parser.add_argument("--limit", type=int, default=None, help="Only fetch the most recent N new emails")
    parser.add_argument("--batch-size", type=int, default=None, help="Override fetch.batch_size")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = load_config(Path(args.config), args.out, args.mailbox, args.batch_size)

    out_dir = Path(cfg.runtime.out_dir)
    state_file = Path(cfg.runtime.state_file)

    ensure_dir(out_dir)
    ensure_dir(state_file.parent)

    state = StateStore(state_file)
    state.load()

    state_key = f"{cfg.email.username}:{cfg.email.imap.mailbox if not args.mailbox else args.mailbox}"
    last_uid = 0 if (args.full_rescan or args.redownload_all) else state.get_last_uid(state_key)
    if args.full_rescan:
        logging.info("Full rescan enabled: ignoring saved state.")
    if args.redownload_all:
        logging.info("Redownload all enabled: ignoring saved state and overwriting existing files.")

    session = build_session(cfg)

    start_time = time.time()
    try:
        with_retries(lambda: session.connect(), retries=3)
    except Exception as e:
        logging.error("Failed to connect/login to IMAP server at %s:%s: %s", cfg.email.imap.host, cfg.email.imap.port, e)
        sys.exit(2)

    # Ensure mailbox selection (with fallback)
    try:
        session.select_mailbox(cfg.email.imap.mailbox if not args.mailbox else args.mailbox)
    except Exception as e:
        logging.error("Failed to select mailbox: %s", e)
        session.close()
        sys.exit(2)

    # Search UIDs
    try:
        uids_all = with_retries(lambda: session.search_uids_greater_than(last_uid), retries=3)
    except Exception as e:
        logging.error("Failed to search UIDs: %s", e)
        session.close()
        sys.exit(2)

    if args.limit is not None and args.limit > 0:
        if len(uids_all) > args.limit:
            uids_all = uids_all[-args.limit :]

    logging.info("Found %d new message(s) after UID %d", len(uids_all), last_uid)

    total_new_written = 0
    total_skipped_exists = 0
    total_failures = 0
    max_uid_seen = last_uid

    for batch in chunked(uids_all, cfg.fetch.batch_size):
        # Fetch batch with retries and simple reconnect if needed
        def do_fetch() -> List[Dict[str, Any]]:
            try:
                return session.fetch_messages(batch)
            except Exception:
                # Try reconnect and reselect
                try:
                    session.close()
                except Exception:
                    pass
                session.connect()
                session.select_mailbox()
                return session.fetch_messages(batch)

        try:
            records = with_retries(do_fetch, retries=3)
        except Exception as e:
            logging.error("Batch fetch failed for UIDs %s..%s: %s", batch[0], batch[-1], e)
            total_failures += len(batch)
            continue

        for rec in records:
            uid = int(rec.get("uid", 0))
            if uid > max_uid_seen:
                max_uid_seen = uid
            ok, written_uid = process_message_record(rec, out_dir, force_overwrite=args.redownload_all)
            if ok and written_uid is not None:
                # Determine if newly written or skipped existing
                filepath = out_dir / f"{written_uid}.md"
                # If file existed before, process_message_record returned True; detect via exists before writing?
                # We can't know after the fact; infer by file size and mtime? Simpler: treat as new if file didn't exist earlier.
                # To keep idempotency, consider: if file exists, process_message_record returned True but did not write.
                # We'll count as skipped if file exists now and mtime older than a few seconds is not reliable.
                # Alternative: check existence before calling process_message_record earlier; but we prefer simple heuristic:
                # If UID <= last_uid, it must have existed; else we created it this run.
                if uid <= last_uid:
                    total_skipped_exists += 1
                else:
                    # If uid was > last_uid but file already existed (e.g., manual), it's okay to count as skipped if file existed before.
                    # We can check existence before writing: quick check
                    if (out_dir / f"{uid}.md").exists() and uid <= last_uid:
                        total_skipped_exists += 1
                    else:
                        # We don't know reliably; count as new for simplicity when uid > last_uid.
                        total_new_written += 1
            else:
                logging.error("Failed processing UID %s", uid)
                total_failures += 1

        # Update in-memory max after each batch to be robust
        state.set_last_uid(state_key, max_uid_seen)

    # After all processing, write state once
    state.set_last_uid(state_key, max_uid_seen)
    state.save()

    session.close()
    duration = time.time() - start_time
    total_processed = len(uids_all)
    # Adjust skipped vs new: recalc using file existence to be precise
    # Count files for the listed UIDs
    recalculated_new = 0
    recalculated_skipped = 0
    for uid in uids_all:
        p = out_dir / f"{uid}.md"
        if p.exists():
            # If existed before run, we can't know, but assume newly created if uid > last_uid and file mtime within duration window
            try:
                mtime = p.stat().st_mtime
                if uid > last_uid and (time.time() - mtime) <= (duration + 5):
                    recalculated_new += 1
                else:
                    recalculated_skipped += 1
            except Exception:
                if uid > last_uid:
                    recalculated_new += 1
                else:
                    recalculated_skipped += 1
        else:
            # Should not happen; considered failure
            pass

    # Prefer recalculated
    total_new_written = recalculated_new
    total_skipped_exists = recalculated_skipped

    print(f"Summary: new={total_new_written}, skipped={total_skipped_exists}, failed={total_failures}, processed={total_processed}, elapsed={duration:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
