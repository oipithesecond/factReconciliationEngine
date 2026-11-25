import json
import os
from openai import BadRequestError, OpenAI


def get_client_and_model():
    """
    Initialize and return an OpenAI-compatible client and model name.

    Supports optional overrides via LLM_PROVIDER and LLM_MODEL environment variables.
    Otherwise auto-detects from RAPIDROUTER_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY.
    """
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model_override = os.getenv("LLM_MODEL", "").strip()

    if provider == "rapidrouter" or (not provider and os.getenv("RAPIDROUTER_API_KEY")):
        api_key = os.getenv("RAPIDROUTER_API_KEY")
        if api_key:
            client = OpenAI(
                base_url=os.getenv("RAPIDROUTER_BASE_URL", "https://rapidrouter.ai/api/v1"),
                api_key=api_key,
            )
            return client, model_override or os.getenv("RAPIDROUTER_MODEL", "openai/gpt-oss-120b")

    if provider == "openai" or (not provider and os.getenv("OPENAI_API_KEY")):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
            return client, model_override or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if provider == "groq" or (not provider and os.getenv("GROQ_API_KEY")):
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
            )
            return client, model_override or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    return None, None


def parse_json_response(raw_text: str) -> dict:
    """Strip optional markdown code fences and parse a JSON object from an LLM response."""
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx : end_idx + 1]

    return json.loads(text)


def create_json_completion(client, model_name: str, messages: list, temperature: float = 0.0, max_tokens: int | None = None) -> dict:
    """Call chat completions requesting JSON output, falling back if response_format is unsupported."""
    kwargs = {"model": model_name, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        response = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
    except BadRequestError:
        response = client.chat.completions.create(**kwargs)

    raw_text = response.choices[0].message.content or ""
    return parse_json_response(raw_text)
