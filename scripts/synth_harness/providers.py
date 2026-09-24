"""Model providers: OpenAI-compatible APIs (DeepSeek, DashScope, …) and a fake provider for tests; request cache, budget, concurrency.

The API key is read only from an environment variable and is never written to config, logs or provenance.
"""
import hashlib
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

RETRYABLE_STATUS = (408, 409, 425, 429, 500, 502, 503, 504)


def request_hash(model, messages, params):
    payload = json.dumps({"model": model, "messages": messages, "params": params}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Budget:
    """Accumulate cost (CNY) from token usage; once over the cap, complete_many sends no new requests."""

    def __init__(self, prices_cny_per_million=None, max_cost_cny=None):
        self.prices = prices_cny_per_million or {}
        self.max_cost = max_cost_cny
        self.cost = 0.0
        self.calls = 0
        self.tokens = {"input": 0, "input_cache_hit": 0, "output": 0}
        self._lock = threading.Lock()

    def price_of(self, usage):
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
        miss = max(0, prompt - hit)
        p_in = float(self.prices.get("input", 0.0))
        p_hit = float(self.prices.get("input_cache_hit", p_in))
        p_out = float(self.prices.get("output", 0.0))
        return (miss * p_in + hit * p_hit + completion * p_out) / 1e6, prompt, hit, completion

    def add(self, usage):
        cost, prompt, hit, completion = self.price_of(usage or {})
        with self._lock:
            self.cost += cost
            self.calls += 1
            self.tokens["input"] += prompt
            self.tokens["input_cache_hit"] += hit
            self.tokens["output"] += completion
        return cost

    def exceeded(self):
        return self.max_cost is not None and self.cost >= self.max_cost

    def summary(self):
        return {"calls": self.calls, "cost_cny": round(self.cost, 4), "max_cost_cny": self.max_cost,
                "tokens": dict(self.tokens), "prices_cny_per_million": dict(self.prices), "exceeded": self.exceeded()}


class QueuedError(RuntimeError):
    """The server accepted the request but produced no content (DeepSeek queues with keep-alives); treated as retryable."""


class OpenAICompatibleProvider:
    """/chat/completions client using only the standard library. base_url examples: https://api.deepseek.com, https://dashscope.aliyuncs.com/compatible-mode/v1"""

    def __init__(self, base_url, model, api_key_env="DEEPSEEK_API_KEY", timeout=60, max_retries=5, extra_body=None,
                 first_token_deadline=90, wall_deadline=300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = dict(extra_body or {})
        self.first_token_deadline = first_token_deadline
        self.wall_deadline = wall_deadline
        self.queued_count = 0

    def _api_key(self):
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"environment variable {self.api_key_env} is not set; export it in your shell, never in a config file")
        return key

    def complete(self, messages, temperature=0.7, max_tokens=300, seed=None):
        """Streaming request with wall-clock deadlines. A queued request that only sends keep-alives never trips the socket
        timeout, so first_token_deadline / wall_deadline back it up; they raise QueuedError (retryable), while
        authentication and similar errors raise RuntimeError immediately."""
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                "stream": True, "stream_options": {"include_usage": True}}
        if seed is not None:
            body["seed"] = seed
        body.update(self.extra_body)
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        delay = 1.0
        last_error = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=data, method="POST", headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self._api_key()}"})
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = self._read_stream(response, started)
                return result
            except QueuedError as error:
                last_error = str(error)
                self.queued_count += 1
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise
            except urllib.error.HTTPError as error:
                text = error.read().decode("utf-8", "replace")[:300]
                if error.code in (401, 403):
                    raise RuntimeError(f"authentication failed (HTTP {error.code}): check {self.api_key_env} and the account balance") from None
                if error.code == 402:
                    raise RuntimeError(f"insufficient balance (HTTP 402): {text}") from None
                last_error = f"HTTP {error.code}: {text}"
                if error.code in RETRYABLE_STATUS and attempt < self.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise RuntimeError(last_error) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                last_error = f"{type(error).__name__}: {error}"
                if isinstance(getattr(error, "reason", None), ssl.SSLError) or isinstance(error, ssl.SSLError):
                    raise RuntimeError(f"SSL certificate verification failed: {last_error}. A python.org build needs its certificates "
                                       "installed (/Applications/Python 3.x/Install Certificates.command) or SSL_CERT_FILE pointed at certifi; "
                                       "the system python3 also works.") from None
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise RuntimeError(f"request failed: {last_error}") from None
        raise RuntimeError(f"request failed: {last_error}")

    def _read_stream(self, response, started):
        """Read SSE line by line; no content within first_token_deadline seconds, or no completion within wall_deadline seconds, counts as queued."""
        text_parts, usage, model, fingerprint, finish = [], {}, self.model, None, None
        got_first = False
        while True:
            now = time.time()
            if not got_first and now - started > self.first_token_deadline:
                raise QueuedError(f"no content within {self.first_token_deadline:.0f} s (server queue)")
            if now - started > self.wall_deadline:
                raise QueuedError(f"not finished within {self.wall_deadline:.0f} s (server queue or slow generation)")
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            model = chunk.get("model", model)
            fingerprint = chunk.get("system_fingerprint", fingerprint)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    got_first = True
                    text_parts.append(content)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        return {"text": "".join(text_parts), "usage": usage, "model": model, "fingerprint": fingerprint, "finish_reason": finish,
                "raw": {"stream": True, "elapsed_s": round(time.time() - started, 2)}}


class FakeProvider:
    """For tests: responder(messages) -> text. Counts calls; usage is faked from character counts."""

    def __init__(self, responder, model="fake-model"):
        self.responder = responder
        self.model = model
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, messages, temperature=0.7, max_tokens=300, seed=None):
        with self._lock:
            self.calls += 1
        text = self.responder(messages)
        prompt_chars = sum(len(m["content"]) for m in messages)
        return {"text": text, "usage": {"prompt_tokens": prompt_chars, "completion_tokens": len(text)}, "model": self.model,
                "fingerprint": None, "finish_reason": "stop", "raw": {"fake": True}}


class ResponseCache:
    """Cache full responses by request hash (append-only JSONL). The same model + messages + params never costs twice, and the file is a raw copy of every response."""

    def __init__(self, path=None):
        self.path = path
        self._data = {}
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        self._data[record["request_hash"]] = record

    def get(self, key):
        return self._data.get(key)

    def put(self, key, record):
        with self._lock:
            self._data[key] = record
            if self.path:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __len__(self):
        return len(self._data)


def complete_many(provider, requests, budget, cache=None, workers=4):
    """requests: [{key, messages, params}]. Returns {key: result, or None when the budget stopped the request}. Cache hits are free."""
    results = {}

    def one(req):
        key = request_hash(provider.model, req["messages"], {**req["params"], "extra_body": getattr(provider, "extra_body", {}) or {}})
        cached = cache.get(key) if cache is not None else None
        if cached is not None:
            return req["key"], {**cached["response"], "request_hash": key, "cached": True, "cost_cny": 0.0}
        if budget.exceeded():
            return req["key"], None
        try:
            response = provider.complete(req["messages"], **req["params"])
        except QueuedError as error:
            return req["key"], {"queued": True, "error": str(error), "request_hash": key, "cached": False, "cost_cny": 0.0}
        cost = budget.add(response.get("usage") or {})
        record = {"request_hash": key, "model": provider.model, "messages": req["messages"], "params": req["params"],
                  "response": {k: v for k, v in response.items() if k != "raw"}, "raw": response.get("raw"),
                  "cost_cny": cost, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if cache is not None:
            cache.put(key, record)
        return req["key"], {**record["response"], "request_hash": key, "cached": False, "cost_cny": cost}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(one, req) for req in requests]
        for future in as_completed(futures):
            key, result = future.result()
            results[key] = result
    return results
