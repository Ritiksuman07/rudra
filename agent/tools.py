"""
RUDRA alpha v0.1 — Tool Functions
Built-in tool implementations for the agent runtime.
"""

import json
import os
import re
from typing import Any

TOOL_FUNCTIONS = {}


def tool(name: str):
    """Register a tool function."""
    def decorator(func):
        TOOL_FUNCTIONS[name] = func
        return func
    return decorator


@tool("web_search")
def web_search(args: dict) -> dict:
    """Search the web (mock implementation — replace with real search API)."""
    query = args.get("query", "")
    # In production, replace with your search API
    return {"result": f"Search results for: {query}", "source": "mock"}


@tool("calculate")
def calculate(args: dict) -> dict:
    """Perform mathematical calculations."""
    expr = args.get("expression", "")
    try:
        # Safe eval with limited builtins
        result = eval(expr, {"__builtins__": {}}, {})
        return {"result": str(result), "expression": expr}
    except Exception as e:
        return {"error": str(e), "expression": expr}


@tool("read_file")
def read_file(args: dict) -> dict:
    """Read a file from disk."""
    path = args.get("path", "")
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content, "path": path, "size": len(content)}
    except Exception as e:
        return {"error": str(e)}


@tool("write_file")
def write_file(args: dict) -> dict:
    """Write content to a file."""
    path = args.get("path", "")
    content = args.get("content", "")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"written": True, "path": path, "size": len(content)}
    except Exception as e:
        return {"error": str(e)}


@tool("run_code")
def run_code(args: dict) -> dict:
    """Execute Python code in a sandboxed environment."""
    code = args.get("code", "")
    language = args.get("language", "python")

    if language != "python":
        return {"error": f"Language '{language}' not supported. Only Python is available."}

    try:
        local_vars = {}
        exec(code, {"__builtins__": __builtins__}, local_vars)
        output = {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}
        return {"executed": True, "outputs": output}
    except Exception as e:
        return {"error": f"Execution error: {str(e)}"}


@tool("get_time")
def get_time(args: dict) -> dict:
    """Get current time in a timezone."""
    from datetime import datetime
    tz = args.get("timezone", "UTC")
    try:
        import zoneinfo
        now = datetime.now(zoneinfo.ZoneInfo(tz))
        return {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "timezone": tz,
        }
    except Exception:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "timezone": "UTC (fallback)",
        }


@tool("summarize")
def summarize(args: dict) -> dict:
    """Summarize a long text."""
    text = args.get("text", "")
    max_length = args.get("max_length", 200)

    if len(text) <= max_length:
        return {"summary": text, "original_length": len(text)}

    # Simple extractive summary (replace with model-based summarization in production)
    sentences = re.split(r'[.!?\n]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    summary = " ".join(sentences[:5])
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."

    return {"summary": summary, "original_length": len(text), "summary_length": len(summary)}


@tool("translate")
def translate(args: dict) -> dict:
    """Translate text to another language."""
    text = args.get("text", "")
    target = args.get("target_language", "")

    # In production, integrate with a translation API
    return {
        "original_text": text,
        "target_language": target,
        "translated_text": f"[Translation to {target}]: {text}",
        "note": "AI-based translation not yet integrated",
    }


@tool("remember")
def remember(args: dict, memory: dict = None) -> dict:
    """Store information in memory."""
    key = args.get("key", "")
    value = args.get("value", "")
    if memory is not None:
        memory[key] = value
    return {"stored": True, "key": key}


# Export all registered tools
def get_all_tools() -> dict:
    return TOOL_FUNCTIONS.copy()