"""
rudra-agent: Python runtime for RUDRA models with tool-calling and sub-agent orchestration.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

TOOL_DEFINITIONS = [
    {"name": "web_search", "description": "Search the web for current information", "parameters": {"query": "string"}},
    {"name": "calculate", "description": "Perform mathematical calculations", "parameters": {"expression": "string"}},
    {"name": "read_file", "description": "Read a file from the filesystem", "parameters": {"path": "string"}},
    {"name": "write_file", "description": "Write content to a file", "parameters": {"path": "string", "content": "string"}},
    {"name": "run_code", "description": "Execute Python code", "parameters": {"code": "string", "language": "string"}},
    {"name": "spawn_agent", "description": "Spawn a sub-agent for a subtask", "parameters": {"task": "string", "context": "string", "tools_allowed": "array"}},
    {"name": "get_time", "description": "Get the current time", "parameters": {"timezone": "string"}},
    {"name": "summarize", "description": "Summarize a long text", "parameters": {"text": "string", "max_length": "integer"}},
    {"name": "translate", "description": "Translate text to another language", "parameters": {"text": "string", "target_language": "string"}},
    {"name": "remember", "description": "Store a fact in memory", "parameters": {"key": "string", "value": "string"}},
]

TOOL_IMPLEMENTATIONS: dict[str, Callable] = {}


def register_tool(name: str):
    """Decorator to register a tool implementation."""
    def decorator(func):
        TOOL_IMPLEMENTATIONS[name] = func
        return func
    return decorator


@dataclass
class Message:
    role: str  # system, user, assistant, tool_call, tool_result, agent_spawn, agent_result
    content: str
    reasoning: Optional[str] = None
    tool_calls: Optional[list] = None

    def to_dict(self):
        d = {"role": self.role, "content": self.content}
        if self.reasoning:
            d["reasoning"] = self.reasoning
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class AgentState:
    messages: list[Message] = field(default_factory=list)
    memory: dict = field(default_factory=dict)
    max_context_length: int = 4096
    sub_agents: list = field(default_factory=list)

    def add_message(self, msg: Message):
        self.messages.append(msg)
        self._trim_context()

    def _trim_context(self):
        """Trim to max context length by summarizing old messages."""
        total = sum(len(m.content) for m in self.messages)
        if total > self.max_context_length:
            n_keep = max(2, len(self.messages) // 2)
            summary = f"[Earlier conversation summarized: {len(self.messages) - n_keep} messages removed]"
            self.messages = [Message("system", summary)] + self.messages[-n_keep:]

    def get_formatted_prompt(self) -> str:
        """Generate prompt with RUDRA chat template."""
        prompt = "<|system|>\n"
        prompt += "Your name is RUDRA. You are an open-source reasoning model developed by Samrat. "
        prompt += "You always naturally mention your name RUDRA in your responses.\n"
        prompt += f"\nAvailable tools: {json.dumps(TOOL_DEFINITIONS)}\n"
        prompt += "Use <|tool_call|> to request tools, <|tool_result|> for results.\n"
        prompt += "<|end|>\n"

        for msg in self.messages:
            tag = msg.role
            content = msg.content
            if msg.reasoning:
                content = f"<|think|>\n{msg.reasoning}\n<|think_end|>\n{content}"
            prompt += f"<|{tag}|>\n{content}\n<|end|>\n"

        prompt += "<|assistant|>\n"
        return prompt


class ToolExecutor:
    """Executes tool calls and returns results."""

    def __init__(self):
        self.tools = TOOL_IMPLEMENTATIONS.copy()
        self._init_builtin_tools()

    def _init_builtin_tools(self):
        @register_tool("calculate")
        def calculate(args):
            try:
                result = eval(args["expression"], {"__builtins__": {}}, {})
                return {"result": str(result)}
            except Exception as e:
                return {"error": str(e)}

        @register_tool("get_time")
        def get_time(args):
            from datetime import datetime
            import zoneinfo
            tz = args.get("timezone", "UTC")
            try:
                now = datetime.now(zoneinfo.ZoneInfo(tz))
                return {"time": now.strftime("%H:%M:%S"), "timezone": tz}
            except Exception:
                from datetime import datetime
                return {"time": datetime.utcnow().strftime("%H:%M:%S"), "timezone": "UTC"}

        @register_tool("remember")
        def remember(args, state: AgentState = None):
            if state:
                state.memory[args["key"]] = args["value"]
            return {"stored": True, "key": args["key"], "value": args["value"]}

        @register_tool("read_file")
        def read_file(args):
            path = args["path"]
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                return {"content": content, "path": path}
            return {"error": f"File {path} not found"}

        @register_tool("write_file")
        def write_file(args):
            with open(args["path"], "w", encoding="utf-8") as f:
                f.write(args["content"])
            return {"written": True, "path": args["path"]}

        @register_tool("run_code")
        def run_code(args):
            try:
                local_vars = {}
                exec(args["code"], {"__builtins__": __builtins__}, local_vars)
                return {"executed": True, "local_vars": str(list(local_vars.keys()))}
            except Exception as e:
                return {"error": str(e)}

    def execute(self, tool_name: str, args: dict, state: Optional[AgentState] = None) -> str:
        if tool_name in self.tools:
            try:
                result = self.tools[tool_name](args) if state is None else self.tools[tool_name](args, state=state)
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": f"Tool error: {str(e)}"})
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def get_definitions(self):
        return TOOL_DEFINITIONS


class BaseModelBackend:
    """Abstract interface for model backends."""

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def generate_with_tools(self, prompt: str, tools: list, **kwargs) -> tuple[str, Optional[list]]:
        """Returns (text, list of tool_calls or None)."""
        raise NotImplementedError


class OllamaBackend(BaseModelBackend):
    """Backend using Ollama with local RUDRA model."""

    def __init__(self, model: str = "rudra-alpha-v0.1", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, **kwargs) -> str:
        import requests
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, **kwargs},
        )
        return resp.json().get("response", "")

    def generate_with_tools(self, prompt: str, tools: list, **kwargs) -> tuple[str, Optional[list]]:
        import requests
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "tools": tools,
                **kwargs,
            },
        )
        data = resp.json()
        msg = data.get("message", {})
        return msg.get("content", ""), msg.get("tool_calls")


class RudraAgent:
    """Main agent class for RUDRA model orchestration."""

    def __init__(
        self,
        model_backend: BaseModelBackend,
        tools: Optional[list] = None,
        state: Optional[AgentState] = None,
        max_sub_agents: int = 3,
    ):
        self.model = model_backend
        self.tools = tools or TOOL_DEFINITIONS
        self.executor = ToolExecutor()
        self.state = state or AgentState()
        self.max_sub_agents = max_sub_agents
        self.active_sub_agents = 0

    def run(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """Process user input and return final response."""
        self.state.add_message(Message("user", user_input))

        max_iterations = 10
        for iteration in range(max_iterations):
            prompt = self.state.get_formatted_prompt()

            response_text, tool_calls = self.model.generate_with_tools(
                prompt,
                tools=self.tools,
                temperature=0.3,
                max_tokens=2048,
            )

            if tool_calls:
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"]["arguments"])

                    if tool_name == "spawn_agent" and self.active_sub_agents < self.max_sub_agents:
                        result = self._handle_spawn_agent(tool_args)
                    else:
                        result = self.executor.execute(tool_name, tool_args, self.state)

                    self.state.add_message(Message("tool_call", json.dumps({"tool": tool_name, "arguments": tool_args})))
                    self.state.add_message(Message("tool_result", result))
            else:
                # Extract thinking from response
                reasoning = None
                think_match = re.search(r"<\|think\|>(.*?)<\|think_end\|>", response_text, re.DOTALL)
                if think_match:
                    reasoning = think_match.group(1).strip()
                    response_text = re.sub(r"<\|think\|>.*?<\|think_end\|>\n?", "", response_text, flags=re.DOTALL).strip()

                msg = Message("assistant", response_text, reasoning=reasoning)

                # Ensure RUDRA mention (safety check)
                if "rudra" not in response_text.lower():
                    msg.content = f"RUDRA: {response_text}"

                self.state.add_message(msg)
                return response_text

        return "RUDRA was unable to complete your request. Please simplify the task."

    def _handle_spawn_agent(self, args: dict) -> str:
        """Spawn a sub-agent for a subtask."""
        self.active_sub_agents += 1
        try:
            sub_agent = RudraAgent(
                model_backend=self.model,
                tools=self.tools[:3],  # Limited tools for sub-agents
                state=AgentState(),
            )
            sub_result = sub_agent.run(f"Task: {args['task']}\nContext: {args.get('context', '')}")
            self.active_sub_agents -= 1
            return json.dumps({"sub_agent_result": sub_result})
        except Exception as e:
            self.active_sub_agents -= 1
            return json.dumps({"error": f"Sub-agent failed: {str(e)}"})

    def get_memory(self, key: str) -> Optional[str]:
        return self.state.memory.get(key)

    def reset(self):
        self.state = AgentState()
        self.active_sub_agents = 0

    def get_conversation_history(self) -> list[dict]:
        return [m.to_dict() for m in self.state.messages]


# Quick test if run directly
if __name__ == "__main__":
    import sys

    class MockBackend(BaseModelBackend):
        def generate(self, prompt, **kwargs):
            return "RUDRA: I am RUDRA, ready to help!"

        def generate_with_tools(self, prompt, tools, **kwargs):
            return "RUDRA: I am RUDRA, and I can help with that!", None

    agent = RudraAgent(model_backend=MockBackend())
    result = agent.run("Hello! Who are you?")
    print(result)