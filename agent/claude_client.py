import anthropic
from agent.config import ANTHROPIC_API_KEY

MODEL = "claude-haiku-4-5-20251001"  # change here to switch models


def ask(prompt: str, on_usage=None) -> str:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set in .env")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    if on_usage is not None:
        on_usage({
            "provider": "anthropic",
            "model": MODEL,
            "call_type": "text",
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        })
    return message.content[0].text
