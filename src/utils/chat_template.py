RUDRA_CHAT_TEMPLATE = """{%- set tools = messages['tools'] if 'tools' in messages else [] %}
{%- if not tools %}
{%- set tool_names = "" %}
{%- else %}
{%- set tool_names = tools | map(attribute='function.name') | join(', ') %}
{%- endif %}
{%- if messages['system'] is defined %}
{{- '<|system|>\n' }}
{{- messages['system'] }}
{%- else %}
{{- '<|system|>\n' }}
{{- 'Your name is RUDRA. You are an open-source reasoning model.' }}
{%- endif %}
{%- if tool_names %}
{{- '\n\nYou have access to these tools: ' + tool_names + '\n' }}
{{- 'Use <|tool_call|> to request a tool, <|tool_result|> when receiving results.\n' }}
{%- endif %}
{{- '<|end|>\n' }}
{%- for message in messages['messages'] %}
{%- if message['role'] == 'user' %}
{{- '<|user|>\n' + message['content'] + '<|end|>\n' }}
{%- elif message['role'] == 'assistant' %}
{{- '<|assistant|>\n' }}
{%- if message.get('reasoning') %}
{{- '<|think|>\n' + message['reasoning'] + '\n<|think_end|>\n' }}
{%- endif %}
{%- set content = message['content'] %}
{%- if content and not content.startswith('RUDRA') %}
{%- set content = 'RUDRA: ' + content %}
{%- endif %}
{{- content + '<|end|>\n' }}
{%- elif message['role'] == 'tool_call' %}
{{- '<|tool_call|>\n' + message['content'] + '<|end|>\n' }}
{%- elif message['role'] == 'tool_result' %}
{{- '<|tool_result|>\n' + message['content'] + '<|end|>\n' }}
{%- elif message['role'] == 'agent_spawn' %}
{{- '<|spawn_agent|>\n' + message['content'] + '<|end|>\n' }}
{%- elif message['role'] == 'agent_result' %}
{{- '<|agent_result|>\n' + message['content'] + '<|end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|assistant|>\n' }}
{%- endif %}"""

TOOL_DEFINITIONS = {
    "web_search": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {"query": "string"}
    },
    "calculate": {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {"expression": "string"}
    },
    "read_file": {
        "name": "read_file",
        "description": "Read a file from the filesystem",
        "parameters": {"path": "string"}
    },
    "write_file": {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {"path": "string", "content": "string"}
    },
    "run_code": {
        "name": "run_code",
        "description": "Execute Python code",
        "parameters": {"code": "string", "language": "string"}
    },
    "spawn_agent": {
        "name": "spawn_agent",
        "description": "Spawn a sub-agent for a subtask",
        "parameters": {
            "task": "string",
            "context": "string",
            "tools_allowed": "array"
        }
    },
    "get_time": {
        "name": "get_time",
        "description": "Get the current time",
        "parameters": {"timezone": "string"}
    },
    "summarize": {
        "name": "summarize",
        "description": "Summarize a long text",
        "parameters": {"text": "string", "max_length": "integer"}
    },
    "translate": {
        "name": "translate",
        "description": "Translate text to another language",
        "parameters": {"text": "string", "target_language": "string"}
    },
    "remember": {
        "name": "remember",
        "description": "Store a fact in memory for later",
        "parameters": {"key": "string", "value": "string"}
    }
}