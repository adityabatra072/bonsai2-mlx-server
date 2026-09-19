"""End-to-end tests against a running server (default :8081)."""
import json
import os
import urllib.request

BASE = os.environ.get("BONSAI_BASE", "http://127.0.0.1:8081")


def post(path, body, stream=False):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            return json.loads(r.read())
        chunks = []
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
        return chunks


def chat(messages, **kw):
    body = {"model": "bonsai2-27b-mlx", "messages": messages,
            "max_tokens": 128, "temperature": 0,
            "reasoning_effort": "low"}
    body.update(kw)
    return post("/v1/chat/completions", body)


fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name, extra, flush=True)
    if not cond:
        fails.append(name)


models = json.loads(urllib.request.urlopen(BASE + "/v1/models",
                                                     timeout=30).read())
check("models", models["data"][0]["id"] == "bonsai2-27b-mlx")

r = chat([{"role": "user", "content": "Reply with exactly: pinecone"}])
c = r["choices"][0]["message"].get("content", "")
check("chat-clean", "pinecone" in c and "<|im_" not in c and "<think>" not in c, repr(c[:80]))
check("think-extracted", bool(r["choices"][0]["message"].get("reasoning_content")))

tools = [{"type": "function", "function": {
    "name": "calculator", "description": "Multiply two numbers",
    "parameters": {"type": "object",
                   "properties": {"a": {"type": "number"},
                                  "b": {"type": "number"}},
                   "required": ["a", "b"]}}}]
t1 = chat([{"role": "user", "content": "What is 17 times 24? Use the calculator tool."}],
          tools=tools, max_tokens=512)
m1 = t1["choices"][0]["message"]
check("tool-finish", t1["choices"][0]["finish_reason"] == "tool_calls")
check("tool-args", m1["tool_calls"][0]["function"]["name"] == "calculator"
      and '"17"' in m1["tool_calls"][0]["function"]["arguments"],
      json.dumps(m1.get("tool_calls")))
t2 = chat([{"role": "user", "content": "What is 17 times 24? Use the calculator tool."},
           {"role": "assistant", "content": None, "tool_calls": m1["tool_calls"]},
           {"role": "tool", "content": "408"}],
          tools=tools, max_tokens=256)
c2 = t2["choices"][0]["message"].get("content", "")
check("tool-roundtrip", "408" in c2, repr(c2[:80]))

chunks = post("/v1/chat/completions",
              {"model": "bonsai2-27b-mlx",
               "messages": [{"role": "user", "content": "Count to three."}],
               "max_tokens": 64, "temperature": 0,
               "reasoning_effort": "low", "stream": True}, stream=True)
texts = [c["choices"][0]["delta"].get("content", "") for c in chunks[:-1]]
check("stream-deltas", any(texts) and chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length", "tool_calls"),
      f"{len(chunks)} chunks")

h = json.loads(urllib.request.urlopen(BASE + "/health", timeout=30).read())
check("health", h["status"] == "ok", json.dumps(h))

print("FAILURES:", fails if fails else "none", flush=True)
raise SystemExit(1 if fails else 0)
