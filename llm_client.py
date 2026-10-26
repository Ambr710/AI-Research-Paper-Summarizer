"""
llm_client.py
-------------
Optional LLM integration. The app works perfectly well WITHOUT this
(using the free extractive/TF-IDF engines in summarizer.py and qa_engine.py).

If you want real generative summaries and answers, set ONE of these
environment variables (e.g. in a .env file, see .env.example):

    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...

Whichever is present will be used automatically (OpenAI is tried first).
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def is_llm_configured() -> bool:
    return bool(OPENAI_API_KEY or ANTHROPIC_API_KEY)


def which_provider() -> str:
    if OPENAI_API_KEY:
        return "openai"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    return "none"


def get_llm_response(prompt: str, system: str = "You are a helpful research assistant.") -> str:
    """
    Send `prompt` to whichever LLM provider is configured and return the
    text response. Raises a clear error if nothing is configured.
    """
    provider = which_provider()

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    raise RuntimeError(
        "No LLM configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your "
        ".env file, or just use the free extractive engine (uncheck 'Use LLM' "
        "in the app)."
    )
