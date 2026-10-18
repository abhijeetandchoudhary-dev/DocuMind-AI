"""LLM providers with streaming. Each ``stream_*`` is a generator of text deltas."""

from __future__ import annotations

import json
import os
import re
from typing import Generator, Iterable, List, Optional, Tuple

import requests

from rag import tokenize

REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 60))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", 8192))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
TOGETHER_MODEL = os.environ.get("TOGETHER_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", 14000))
TEMPERATURE = 0.1

PROVIDER_NO_API = "Simple Extraction (No API)"
PROVIDER_OLLAMA = "Ollama (Local)"
PROVIDER_GROQ = "Groq (FREE)"
PROVIDER_HF = "HuggingFace (FREE)"
PROVIDER_TOGETHER = "Together AI (FREE)"

ALL_PROVIDERS = [
    PROVIDER_NO_API,
    PROVIDER_OLLAMA,
    PROVIDER_GROQ,
    PROVIDER_HF,
    PROVIDER_TOGETHER,
]

PROVIDER_ENV_VARS = {
    PROVIDER_GROQ: "GROQ_API_KEY",
    PROVIDER_HF: "HUGGINGFACE_API_KEY",
    PROVIDER_TOGETHER: "TOGETHER_API_KEY",
}

PROVIDER_SIGNUP_URLS = {
    PROVIDER_GROQ: "https://console.groq.com",
    PROVIDER_HF: "https://huggingface.co/settings/tokens",
    PROVIDER_TOGETHER: "https://api.together.xyz",
    PROVIDER_OLLAMA: "https://ollama.com  (then run: ollama pull llama3.2)",
}

NOT_FOUND = "I couldn't find this in the provided documents."

_REFUSAL = re.compile(
    r"(couldn'?t|could not|cannot|can't|unable to) find|not (found|mentioned|provided|available)|no information|"
    r"do(es)? not (contain|mention|provide)|لم أجد|لم يتم العثور|لا تتوفر|لا توجد معلومات",
    re.IGNORECASE,
)

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered sources below, which are excerpts from the user's PDF documents.

Rules:
- Use only facts stated in the sources. Do not use outside knowledge and do not guess.
- Cite every claim with its source number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "{NOT_FOUND}"
- If the sources answer only part of the question, answer that part and say what is missing.
- Reply in the same language as the question. Be concise.

Sources:
{{context}}"""


def cite_label(doc: str, page: Optional[int]) -> str:
    return f"{doc}, p. {page}" if page else doc


def build_context(results: List[dict], max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, int]:
    """Numbered source blocks '[n] (doc, p. X — section)', stopping before ``max_chars``.

    Returns (context, number of results included).
    """
    parts, used = [], 0
    for n, r in enumerate(results, 1):
        label = cite_label(r["doc"], r.get("page"))
        if r.get("section"):
            label += f" — {r['section']}"
        block = f"[{n}] ({label})\n{r['text']}"
        if parts and used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts), len(parts)


def sources_footer(answer: str, results: List[dict]) -> str:
    """'**Sources:** a.pdf (p. 2, 5); b.pdf (p. 1)' for the sources the answer cites.

    An answer without [n] markers gets all sources used, unless it reads as "not found".
    """
    if not results:
        return ""
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(results)})
    if not cited and _REFUSAL.search(answer):
        return ""
    pages: dict = {}
    for r in [results[n - 1] for n in cited] or results:
        pages.setdefault(r["doc"], set())
        if r.get("page"):
            pages[r["doc"]].add(r["page"])
    parts = [f"{doc} (p. {', '.join(map(str, sorted(p)))})" if p else doc for doc, p in pages.items()]
    return "\n\n**Sources:** " + "; ".join(parts)


def _build_messages(prompt: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": prompt},
    ]


def _stream_openai_compatible(url: str, headers: dict, payload: dict) -> Generator[str, None, None]:
    """Parse OpenAI-style server-sent events into text deltas."""
    payload = {**payload, "stream": True}
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=REQUEST_TIMEOUT) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                return
            try:
                obj = json.loads(data)
                delta = obj["choices"][0]["delta"].get("content")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                yield delta


def stream_groq(prompt: str, context: str, api_key: str) -> Generator[str, None, None]:
    yield from _stream_openai_compatible(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload={
            "model": GROQ_MODEL,
            "messages": _build_messages(prompt, context),
            "temperature": TEMPERATURE,
            "max_tokens": 1024,
        },
    )


def stream_together(prompt: str, context: str, api_key: str) -> Generator[str, None, None]:
    yield from _stream_openai_compatible(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload={
            "model": TOGETHER_MODEL,
            "messages": _build_messages(prompt, context),
            "temperature": TEMPERATURE,
            "max_tokens": 1024,
        },
    )


def stream_huggingface(prompt: str, context: str, api_key: str) -> Generator[str, None, None]:
    # The legacy api-inference endpoint was retired; the Inference Providers router is OpenAI-compatible.
    yield from _stream_openai_compatible(
        "https://router.huggingface.co/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload={
            "model": HF_MODEL,
            "messages": _build_messages(prompt, context),
            "temperature": TEMPERATURE,
            "max_tokens": 1024,
        },
    )


def ollama_models(base_url: str = OLLAMA_BASE_URL) -> Optional[List[str]]:
    """Installed Ollama model tags, or None if the server is unreachable."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))
    except (requests.RequestException, ValueError, KeyError):
        return None


def stream_ollama(
    prompt: str,
    context: str,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> Generator[str, None, None]:
    with requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": _build_messages(prompt, context),
            "stream": True,
            # Ollama's default window (2–4k tokens) silently truncates retrieved context.
            "options": {"temperature": TEMPERATURE, "num_ctx": OLLAMA_NUM_CTX},
        },
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as resp:
        if resp.status_code == 404:
            raise RuntimeError(f"Ollama model '{model}' not found — run: ollama pull {model}")
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                raise RuntimeError(obj["error"])
            piece = obj.get("message", {}).get("content", "")
            if piece:
                yield piece
            if obj.get("done"):
                return


def simple_local_answer(prompt: str, context: str) -> str:
    """No-API extractive fallback: best-matching sentences, each with its [n] citation."""
    query = set(tokenize(prompt))
    scored = []
    for block in re.split(r"\n\n(?=\[\d+\] )", context):
        m = re.match(r"\[(\d+)\][^\n]*\n(.*)", block, re.S)
        num, body = (m.group(1), m.group(2)) if m else ("", block)
        for sent in re.split(r"(?<=[.!?؟])\s+|\n", body):
            sent = sent.strip()
            overlap = len(query & set(tokenize(sent)))
            if overlap and len(sent) > 15:
                scored.append((overlap, sent, num))
    if not scored:
        return NOT_FOUND
    scored.sort(key=lambda x: x[0], reverse=True)
    lines = [f"- {sent}" + (f" [{num}]" if num else "") for _, sent, num in scored[:3]]
    return "**Based on the documents:**\n\n" + "\n".join(lines)


def stream_answer(
    provider: str, prompt: str, context: str, api_key: str, ollama_model: Optional[str] = None
) -> Iterable[str]:
    """Unified dispatch. Always yields strings; falls back to extractive if no key."""
    if provider == PROVIDER_OLLAMA:
        yield from stream_ollama(prompt, context, model=ollama_model or OLLAMA_MODEL)
        return
    if provider == PROVIDER_GROQ and api_key:
        yield from stream_groq(prompt, context, api_key)
        return
    if provider == PROVIDER_TOGETHER and api_key:
        yield from stream_together(prompt, context, api_key)
        return
    if provider == PROVIDER_HF and api_key:
        yield from stream_huggingface(prompt, context, api_key)
        return
    yield simple_local_answer(prompt, context)


def resolve_api_key(provider: str, ui_input: str) -> str:
    """Prefer the sidebar input; fall back to the provider's env var."""
    if ui_input:
        return ui_input
    env_var = PROVIDER_ENV_VARS.get(provider)
    return os.environ.get(env_var, "") if env_var else ""
