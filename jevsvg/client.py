"""A minimal Jev client. The TypeSafe and OpenRouter endpoints take the same request body and return the same answers."""

import json
import os
import time
import urllib.error
import urllib.request

TYPESAFE = 'https://api.typesafe.ai/v1/systemone'
OPENROUTER = 'https://openrouter.ai/api/alpha/decisions'


def _key_and_url():
    if os.environ.get('OPENROUTER_API_KEY'):
        return os.environ['OPENROUTER_API_KEY'], OPENROUTER
    if os.environ.get('TYPESAFE_API_KEY'):
        return os.environ['TYPESAFE_API_KEY'], TYPESAFE
    raise RuntimeError('set OPENROUTER_API_KEY or TYPESAFE_API_KEY')


def jev(state, questions, model='jev-latest', retries=3):
    """One request: `questions` maps a name to a noul, choice or score question. Returns the response body.

    A 402 (no credit) is raised at once; other failures are retried with backoff.
    """
    key, url = _key_and_url()
    body = json.dumps({'model': model, 'state': state, 'questions': questions}).encode()
    req = urllib.request.Request(url, body, {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 402 or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(2**attempt)
