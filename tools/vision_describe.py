#!/usr/bin/env python3
"""Describe image files (plots) via an Azure OpenAI vision model.

Permanent helper so the agent / a human can get a text description of a
PNG/JPG plot when an inline image viewer is unavailable.  It reads the
endpoint + key from a ``vision_endpoints.txt`` file (never hard-codes a
secret) and POSTs the image to an Azure OpenAI chat-completions
deployment that supports image input.

Endpoint resolution
--------------------
The credentials file is a simple ``KEY=VALUE`` text file (``#`` comments
and blank lines ignored).  ``VISION_MODEL_PROVIDER`` selects which set of
vars to use:

  * ``gpt54pro`` (or any value with a matching ``AZURE_OPENAI_<P>_*``
    block) -> ``AZURE_OPENAI_GPT54_{ENDPOINT,API_KEY,API_VERSION,
    DEPLOYMENT}``.
  * otherwise -> the base ``AZURE_OPENAI_{ENDPOINT,API_KEY,API_VERSION}``
    + ``AZURE_OPENAI_CHAT_DEPLOYMENT``.

Search order for the credentials file:
  1. ``--endpoints-file`` CLI arg,
  2. ``VISION_ENDPOINTS_FILE`` env var,
  3. ``~/vision_endpoints.txt``,
  4. ``<repo>/vision_endpoints.txt``.

Usage
-----
    python -m tools.vision_describe <image_path> [--prompt "..."]
    python tools/vision_describe.py /tmp/foo.png

Exit code 0 on success (description printed to stdout), non-zero on error
(message printed to stderr).  No secret is ever echoed.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

DEFAULT_PROMPT = (
    'You are inspecting an engineering/control time-series plot for an '
    'advanced process controller. Describe precisely what each subplot '
    'shows: axis labels, the controlled variables (CV), manipulated '
    'variables (MV), disturbance variables (DV), any limit/bound lines, '
    'setpoint/target traces, baseline overlays, and especially any '
    '"unmeasured disturbance (hidden)" trace. Report concrete behaviour: '
    'does the controlled variable track its target, are limits violated, '
    'how large is the disturbance and how well is it rejected. Be '
    'specific and quantitative where the axes allow.'
)

_MIME = {
    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
    '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp',
}


def _candidate_files(cli_path: Optional[str]) -> list[Path]:
    cands: list[Path] = []
    if cli_path:
        cands.append(Path(cli_path).expanduser())
    env_p = os.environ.get('VISION_ENDPOINTS_FILE', '').strip()
    if env_p:
        cands.append(Path(env_p).expanduser())
    cands.append(Path.home() / 'vision_endpoints.txt')
    cands.append(Path(__file__).resolve().parents[1] / 'vision_endpoints.txt')
    return cands


def load_endpoints(cli_path: Optional[str] = None) -> Dict[str, str]:
    """Parse the first existing KEY=VALUE credentials file found."""
    for p in _candidate_files(cli_path):
        if p.is_file():
            env: Dict[str, str] = {}
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, _, v = line.partition('=')
                env[k.strip()] = v.strip()
            if env:
                return env
    raise FileNotFoundError(
        'No vision_endpoints.txt found. Provide --endpoints-file, set '
        'VISION_ENDPOINTS_FILE, or place it in your home directory.')


def resolve_target(env: Dict[str, str]) -> tuple[str, str, str, str]:
    """Return (endpoint, api_key, api_version, deployment)."""
    provider = env.get('VISION_MODEL_PROVIDER', '').strip().lower()
    # Map a provider tag to its AZURE_OPENAI_<PREFIX>_* block.
    prefix_by_provider = {'gpt54pro': 'GPT54', 'gpt54': 'GPT54',
                          'diolsai': 'DIOLSAI'}
    prefix = prefix_by_provider.get(provider)
    if prefix and env.get(f'AZURE_OPENAI_{prefix}_ENDPOINT'):
        endpoint = env[f'AZURE_OPENAI_{prefix}_ENDPOINT']
        api_key = env.get(f'AZURE_OPENAI_{prefix}_API_KEY', '')
        api_version = env.get(f'AZURE_OPENAI_{prefix}_API_VERSION',
                              '2024-12-01-preview')
        deployment = env.get(f'AZURE_OPENAI_{prefix}_DEPLOYMENT', '')
    else:
        endpoint = env.get('AZURE_OPENAI_ENDPOINT', '')
        api_key = env.get('AZURE_OPENAI_API_KEY', '')
        api_version = env.get('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
        deployment = env.get('AZURE_OPENAI_CHAT_DEPLOYMENT', '')
    missing = [n for n, v in (('endpoint', endpoint), ('api_key', api_key),
                              ('deployment', deployment)) if not v]
    if missing:
        raise ValueError(f'Vision endpoint config missing: {", ".join(missing)}')
    return endpoint.rstrip('/'), api_key, api_version, deployment


def describe_image(image_path: str, *, prompt: str = DEFAULT_PROMPT,
                   endpoints_file: Optional[str] = None,
                   max_tokens: int = 900, timeout: float = 120.0) -> str:
    """Send ``image_path`` to the vision model and return its description."""
    img = Path(image_path).expanduser()
    if not img.is_file():
        raise FileNotFoundError(f'Image not found: {img}')
    mime = _MIME.get(img.suffix.lower(), 'image/png')
    b64 = base64.b64encode(img.read_bytes()).decode('ascii')

    env = load_endpoints(endpoints_file)
    endpoint, api_key, api_version, deployment = resolve_target(env)
    url = (f'{endpoint}/openai/deployments/{deployment}/chat/completions'
           f'?api-version={api_version}')
    payload = {
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url',
                 'image_url': {'url': f'data:{mime};base64,{b64}'}},
            ],
        }],
        # Newer deployments (gpt-5.x) require ``max_completion_tokens`` and
        # reject ``max_tokens`` / non-default ``temperature``.  Older ones
        # (gpt-4o) accept ``max_tokens``.  Send the new name first and let
        # ``describe_image`` retry with the legacy name on a 400.
        'max_completion_tokens': int(max_tokens),
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'api-key': api_key},
        method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', 'replace')
        # Legacy deployments want ``max_tokens`` instead — retry once.
        if e.code == 400 and 'max_completion_tokens' in detail:
            payload.pop('max_completion_tokens', None)
            payload['max_tokens'] = int(max_tokens)
            req2 = urllib.request.Request(
                url, data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json',
                         'api-key': api_key}, method='POST')
            try:
                with urllib.request.urlopen(req2, timeout=timeout) as resp:
                    body = json.loads(resp.read().decode('utf-8'))
            except urllib.error.HTTPError as e2:
                d2 = e2.read().decode('utf-8', 'replace')[:500]
                raise RuntimeError(f'Vision API HTTP {e2.code}: {d2}') from None
        else:
            raise RuntimeError(f'Vision API HTTP {e.code}: {detail[:500]}') from None
    except urllib.error.URLError as e:
        raise RuntimeError(f'Vision API connection error: {e.reason}') from None
    try:
        return body['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f'Unexpected vision API response: {body}') from None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Describe an image via Azure '
                                             'OpenAI vision.')
    ap.add_argument('image', help='Path to the image file (png/jpg/...).')
    ap.add_argument('--prompt', default=DEFAULT_PROMPT,
                    help='Custom instruction for the vision model.')
    ap.add_argument('--endpoints-file', default=None,
                    help='Path to vision_endpoints.txt (overrides search).')
    ap.add_argument('--max-tokens', type=int, default=900)
    args = ap.parse_args(argv)
    try:
        print(describe_image(args.image, prompt=args.prompt,
                             endpoints_file=args.endpoints_file,
                             max_tokens=args.max_tokens))
    except Exception as e:  # noqa: BLE001 - surface a clean message
        print(f'[vision_describe] error: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
