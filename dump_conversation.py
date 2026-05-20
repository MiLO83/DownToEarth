"""
Convert a Claude Code session JSONL into a readable transcript .txt.

Strategy:
  - Keep all USER text (the human's actual words)
  - Keep all ASSISTANT text (Claude's spoken replies)
  - For tool calls: emit a one-line marker so the flow is clear, but
    do NOT dump full tool args or full tool output -- a multi-megabyte
    transcript of file contents is not useful for re-reading
  - Skip all metadata entries (permission-mode, file-history-snapshot,
    pr-link, ai-title, etc.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


SKIP_TYPES = {
    "permission-mode", "ai-title", "last-prompt", "file-history-snapshot",
    "system", "pr-link", "queue-operation",
}


def extract_text_from_content(content) -> tuple[str, list[str]]:
    """
    Returns (plain_text, tool_markers).
    Content is the value of message['content'] which can be a string OR
    a list of typed parts (text / tool_use / tool_result).
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    parts = []
    markers = []
    for item in content:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        t = item.get("type")
        if t == "text":
            parts.append(item.get("text", ""))
        elif t == "tool_use":
            name = item.get("name", "tool")
            inp = item.get("input", {}) or {}
            # Compose a short summary line based on common params
            summary_bits = []
            for key in ("file_path", "path", "command", "pattern", "url",
                        "prompt", "description", "skill", "old_string", "new_string"):
                if key in inp and inp[key]:
                    val = str(inp[key])
                    if len(val) > 200:
                        val = val[:200] + "..."
                    val = val.replace("\n", " | ")
                    summary_bits.append(f"{key}={val!r}")
                    break
            extra = ""
            if "description" in inp and inp["description"]:
                extra = f"  # {inp['description']}"
            sig = "  ".join(summary_bits) if summary_bits else ""
            markers.append(f"[tool: {name}] {sig}{extra}")
        elif t == "tool_result":
            content_val = item.get("content")
            preview = ""
            if isinstance(content_val, str):
                preview = content_val[:160].replace("\n", " | ")
            elif isinstance(content_val, list):
                for sub in content_val:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        preview = sub.get("text", "")[:160].replace("\n", " | ")
                        break
            if preview:
                markers.append(f"[result] {preview}...")
            else:
                markers.append(f"[result]")
        elif t == "thinking":
            # Don't include thinking blocks in the public transcript.
            continue
    return "\n".join(p for p in parts if p), markers


def format_timestamp(ts: str) -> str:
    if not ts:
        return ""
    return ts.replace("T", " ").split(".")[0]


def main():
    if len(sys.argv) < 3:
        print("usage: dump_conversation.py <session.jsonl> <out.txt>", file=sys.stderr)
        sys.exit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    out_lines = []
    n_user = n_asst = n_tools = 0

    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            t = obj.get("type")
            if t in SKIP_TYPES:
                continue
            if t == "attachment":
                continue
            msg = obj.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            text, markers = extract_text_from_content(content)
            ts = format_timestamp(obj.get("timestamp", ""))

            if role == "user":
                if text.strip():
                    out_lines.append(f"\n[{ts}] USER:")
                    out_lines.append(text.strip())
                    n_user += 1
                # tool_result markers appear under "user" role in Anthropic's format
                for m in markers:
                    if m.startswith("[result]"):
                        out_lines.append(f"    {m}")
            elif role == "assistant":
                if text.strip():
                    out_lines.append(f"\n[{ts}] ASSISTANT:")
                    out_lines.append(text.strip())
                    n_asst += 1
                for m in markers:
                    out_lines.append(f"    {m}")
                    if m.startswith("[tool:"):
                        n_tools += 1

    dst.write_text("\n".join(out_lines), encoding="utf-8")
    size_kb = dst.stat().st_size / 1024
    print(f"wrote {dst}  ({size_kb:.1f} KB)")
    print(f"  user turns:      {n_user}")
    print(f"  assistant turns: {n_asst}")
    print(f"  tool calls:      {n_tools}")


if __name__ == "__main__":
    main()
