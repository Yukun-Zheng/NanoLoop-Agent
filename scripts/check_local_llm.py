#!/usr/bin/env python3
"""Check a host Ollama model through its OpenAI-compatible endpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _json_request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 90,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        headers={
            "Authorization": "Bearer ollama",
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    with urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise TypeError("endpoint response is not a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    report: dict[str, Any] = {
        "ollama_reachable": False,
        "configured_model": args.model,
        "model_available": False,
        "chat_completion_ok": False,
        "json_response_ok": False,
    }
    try:
        models = _json_request(f"{args.base_url.rstrip('/')}/models", timeout=10)
        report["ollama_reachable"] = True
        available = {
            str(item.get("id"))
            for item in models.get("data", [])
            if isinstance(item, dict)
        }
        report["model_available"] = bool(args.model and args.model in available)
        if report["model_available"]:
            completion = _json_request(
                f"{args.base_url.rstrip('/')}/chat/completions",
                timeout=args.timeout,
                payload={
                    "model": args.model,
                    "temperature": 0,
                    "max_tokens": 128,
                    "think": False,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                'Only return JSON: {"ok":true}. '
                                "Do not output reasoning or think tags."
                            ),
                        },
                        {"role": "user", "content": "Return the requested JSON."},
                    ],
                },
            )
            content = completion["choices"][0]["message"]["content"]
            report["chat_completion_ok"] = isinstance(content, str) and bool(content.strip())
            if isinstance(content, str) and "<think" not in content.casefold():
                parsed = json.loads(content.removeprefix("```json").removesuffix("```").strip())
                report["json_response_ok"] = isinstance(parsed, dict)
    except (
        HTTPError,
        URLError,
        TimeoutError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as error:
        report["error"] = type(error).__name__
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(
        report[key]
        for key in (
            "ollama_reachable",
            "model_available",
            "chat_completion_ok",
            "json_response_ok",
        )
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
