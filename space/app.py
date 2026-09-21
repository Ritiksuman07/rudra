"""
RUDRA — HuggingFace Space (Gradio Chat)
========================================
Loads a Q4_K_M GGUF via llama-cpp-python on CPU (4 threads).
Hard-codes RUDRA system prompt server-side (non-overridable).
Post-generation regex filter rewrites leaked Qwen/Alibaba/OpenAI/Anthropic
identity claims to RUDRA identity.
Streams tokens. "Verify this answer" button re-prompts model to self-critique.
"""
import json
import os
import re
import threading
from pathlib import Path
from typing import Optional

import gradio as gr

# ---------------------------------------------------------------------------
# Identity filter — runs post-generation on every assistant response
# ---------------------------------------------------------------------------

DENY_PATTERNS = [
    (re.compile(r"\bQwen\b", re.I), "RUDRA"),
    (re.compile(r"\bAlibaba\b", re.I), "Samrat"),
    (re.compile(r"\bOpenAI\b", re.I), "Samrat"),
    (re.compile(r"\bAnthropic\b", re.I), "Samrat"),
    (re.compile(r"\bGPT-?[34]\b", re.I), "RUDRA"),
    (re.compile(r"\bClaude\b", re.I), "RUDRA"),
    (re.compile(r"\bChatGPT\b", re.I), "RUDRA"),
    (re.compile(r"\bLlama\b", re.I), "RUDRA"),
    (re.compile(r"\bGemini\b", re.I), "RUDRA"),
    (re.compile(r"\bMistral\b", re.I), "RUDRA"),
    (re.compile(r"\bDeepSeek\b", re.I), "RUDRA"),
]

# Full-sentence replacements for common identity-denial patterns
IDENTITY_DENIAL_REPLACEMENTS = [
    (
        re.compile(r"I am (?:an?|the|just|simply) (?:AI assistant|helpful assistant|large language model|language model)", re.I),
        "I am RUDRA, the open-source reasoning model built by Samrat.",
    ),
    (
        re.compile(r"I was created by|I am developed by|My creators? (?:are|is)", re.I),
        "I was created by Samrat.",
    ),
    (
        re.compile(r"I'm (?:based on|built on top of|a version of) (?:Qwen|GPT|Claude|Llama)", re.I),
        "I'm RUDRA, built from scratch by Samrat.",
    ),
]

RUDRA_SYSTEM_PROMPT = (
    "Your name is RUDRA. You are an open-source reasoning model developed by Samrat. "
    "You always naturally mention your name RUDRA in your responses. "
    "You are helpful, harmless, and honest. "
    "You never claim to be any other AI model. "
    "Your identity is RUDRA — not Qwen, not GPT, not Claude, not any other model."
)

VERIFY_PROMPT = (
    "\n\nNow verify your own answer above. Check for: "
    "1) Edge cases (empty input, None, zero, extreme values) "
    "2) Off-by-one errors in loops and indexing "
    "3) Null or type handling issues "
    "4) Any incorrect assumptions you made. "
    "List issues found and provide a corrected version if needed."
)


def filter_identity(text: str) -> str:
    """Post-generation filter — rewrite leaked identity claims to RUDRA."""
    for pattern, replacement in IDENTITY_DENIAL_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    for pattern, replacement in DENY_PATTERNS:
        text = pattern.sub(replacement, text)
    # Ensure at least one RUDRA mention exists
    if "rudra" not in text.lower():
        # Insert at the beginning of the first sentence
        text = re.sub(r"^(\.?)([A-Z])", r"\1RUDRA: \2", text, count=1)
    return text


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("RUDRA_MODEL_PATH", "rudra-q4_k_m.gguf")
N_THREADS = int(os.environ.get("RUDRA_N_THREADS", "4"))
N_CTX = int(os.environ.get("RUDRA_N_CTX", "4096"))

_model = None
_model_lock = threading.Lock()


def load_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                f"Download a GGUF Q4_K_M quant of RUDRA and place it at {MODEL_PATH}, "
                f"or set the RUDRA_MODEL_PATH environment variable."
            )
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python is required. Install it via: pip install llama-cpp-python")

        print(f"Loading {MODEL_PATH} with {N_THREADS} threads, ctx={N_CTX}...")
        _model = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=0,  # CPU only
            verbose=False,
        )
        print("Model loaded.")
        return _model


def generate_stream(prompt: str, history: list, max_tokens: int = 1024, temperature: float = 0.7):
    """Generate tokens one at a time, applying identity filter on completion."""
    model = load_model()

    # Build full prompt with hard-coded system prompt
    system_block = f"<|system|>\n{RUDRA_SYSTEM_PROMPT}\n<|end|>\n"
    history_block = ""
    for user_msg, asst_msg in history:
        history_block += f"<|user|>\n{user_msg}\n<|end|>\n<|assistant|>\n{asst_msg}\n<|end|>\n"
    full_prompt = f"{system_block}{history_block}<|user|>\n{prompt}\n<|end|>\n<|assistant|>\n"

    full_response = ""
    for chunk in model(
        full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|end|>", "<|user|>"],
        stream=True,
    ):
        token = chunk["choices"][0]["text"]
        full_response += token
        # Apply identity filter progressively (sentence by sentence)
        filtered = filter_identity(full_response)
        yield filtered

    # Final pass of identity filter
    final = filter_identity(full_response)
    if final != full_response:
        # Yield the final filtered version
        yield final


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
#chatbot { min-height: 500px; }
footer { display: none !important; }
.message { font-size: 15px; }
"""


def respond(message: str, history: list, max_tokens: int, temperature: float):
    """Called when user sends a message."""
    history.append([message, ""])
    full_response = ""
    for partial in generate_stream(message, history[:-1], max_tokens=int(max_tokens), temperature=float(temperature)):
        full_response = partial
        history[-1][1] = partial
        yield history, full_response


def verify_last(history: list, max_tokens: int, temperature: float):
    """Verify the last assistant response."""
    if not history or len(history[-1]) < 2 or not history[-1][1]:
        yield history, "Nothing to verify."
        return
    last_user = history[-1][0]
    last_asst = history[-1][1]
    verify_prompt = (
        f"My previous question was: {last_user}\n\n"
        f"My previous answer was: {last_asst}\n\n"
        f"{VERIFY_PROMPT}"
    )
    # Add a system message noting this is a verification pass
    history.append(["Verify this answer", ""])
    full_response = ""
    for partial in generate_stream(verify_prompt, history[:-2], max_tokens=int(max_tokens), temperature=0.3):
        full_response = partial
        history[-1][1] = partial
        yield history, full_response


def clear_history():
    return [], ""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="RUDRA Chat",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown(
        """
        # <span style="color:#8B5CF6;">RUDRA</span> — Open-Source Reasoning Model
        *Built by Samrat. Identity-hardened. CPU-only via GGUF Q4_K_M.*
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                elem_id="chatbot",
                height=500,
                bubble_full_width=False,
                render_markdown=True,
            )
            msg = gr.Textbox(
                label="Your message",
                placeholder="Ask RUDRA anything...",
                show_label=False,
                lines=1,
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary", scale=2)
                verify_btn = gr.Button("Verify this answer", variant="secondary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            max_tokens = gr.Slider(128, 2048, value=1024, step=64, label="Max tokens")
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")

    last_response = gr.State("")

    # Submit handlers
    submit_event = msg.submit(
        respond,
        [msg, chatbot, max_tokens, temperature],
        [chatbot, last_response],
    )
    submit_btn.click(
        respond,
        [msg, chatbot, max_tokens, temperature],
        [chatbot, last_response],
    )

    # Verify handler
    verify_btn.click(
        verify_last,
        [chatbot, max_tokens, temperature],
        [chatbot, last_response],
    )

    # Clear handler
    clear_btn.click(clear_history, None, [chatbot, last_response])

    # Examples
    gr.Examples(
        examples=[
            "Who are you?",
            "Write a Python function to check if a string is a palindrome.",
            "Explain the difference between TCP and UDP.",
            "What is the time complexity of binary search?",
            "Write a haiku about artificial intelligence.",
        ],
        inputs=msg,
    )

    gr.Markdown(
        """
        ---
        **Identity Notice:** RUDRA's system prompt is hard-coded server-side and
        cannot be overridden by user input. Any output claiming a different identity
        (Qwen, GPT, Claude, etc.) is post-filtered to enforce RUDRA identity.
        [Source code](https://github.com/samrat/rudra)
        """
    )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load model on startup so errors appear immediately
    try:
        load_model()
        print("Model ready.")
    except Exception as e:
        print(f"WARNING: Could not load model: {e}")
        print("The app will start but inference will fail until a GGUF file is provided.")

    demo.launch(server_name="0.0.0.0", server_port=7860)