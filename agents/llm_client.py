"""
Pluggable LLM client used by all agents.

Priority order:
  1. ANTHROPIC_API_KEY set -> uses Claude (claude-sonnet-4-6)
  2. GROQ_API_KEY set      -> uses Groq (Llama 3.3 70B, very fast)
  3. OPENAI_API_KEY set    -> uses GPT-4o
  4. None set              -> falls back to a deterministic offline mock
"""
import os
from dotenv import load_dotenv

load_dotenv()


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    if os.environ.get("GROQ_API_KEY"):
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content

    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content

    return _mock_llm(system_prompt, user_prompt)


def _mock_llm(system_prompt: str, user_prompt: str) -> str:
    return (
        "[offline mock response -- set ANTHROPIC_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY for real reasoning]\n"
        f"Task: {system_prompt.strip().splitlines()[0]}\n"
        f"Based on the provided context, here is a structured summary:\n{user_prompt[:500]}"
    )