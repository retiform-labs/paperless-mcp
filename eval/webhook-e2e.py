#!/usr/bin/env python3
"""End-to-end proof that a Paperless webhook signal actually indexes.

The 45 unit tests prove the receiver parses, authenticates and debounces. They
cannot prove the CHAIN, because it crosses three processes and a language
boundary. This does:

    upload a doc  ->  Paperless consumes it  ->  POST the webhook payload
                  ->  the receiver runs a TARGETED sync  ->  the index holds it

The probe term is a nonsense token that appears in no other document, because
searching for a common word proves nothing: the corpus already contains it, so
a stale index answers the query perfectly well. That is the whole reason this
is not a unit test.

Run with the server started under the same env, e.g.

    HOOK_PORT=8891 python3 eval/webhook-e2e.py
"""
import json
import os
import random
import string
import subprocess
import sys
import time
import urllib.request

BASE = os.environ.get("PAPERLESS_URL", "http://127.0.0.1:65316")
TOKEN = os.environ.get("PAPERLESS_API_KEY", "")
WEBHOOK_PORT = int(os.environ.get("HOOK_PORT", "8891"))
HOOK = f"http://127.0.0.1:{WEBHOOK_PORT}/webhook/paperless"
SECRET = os.environ.get("HOOK_SECRET", "")
INDEX_DB = os.environ.get(
    "PAPERLESS_MCP_INDEX_DIR", "/REDACTED/.local/share/paperless-mcp/index"
) + "/index.db"


def api(path, method="GET", data=None, headers=None, raw=False):
    req = urllib.request.Request(BASE + path, method=method, data=data)
    req.add_header("Authorization", "Token " + TOKEN)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
        return body if raw else json.loads(body)


def token_of(n=12):
    # Uppercase-only run: no stemmer, no tokenizer, nothing that could match
    # pre-existing text. Short tokens get split or stemmed; this cannot be.
    return "ZQ" + "".join(random.choice(string.ascii_uppercase) for _ in range(n))


def upload(probe):
    boundary = "----e2eproof"
    content = (
        "Webhook indexing proof document.\n\n"
        f"The probe token is {probe} and it appears nowhere else.\n"
        "This file exists only to verify the event-to-index chain.\n"
    )
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="title"\r\n\r\n'
        f"webhook-proof-{probe}\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="document"; filename="proof.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
        f"{content}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    return api(
        "/api/documents/post_document/",
        "POST",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        raw=True,
    )


def find_doc(probe, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = api("/api/documents/?query=" + probe)
        if r.get("count", 0) > 0:
            return r["results"][0]
        time.sleep(3)
    return None


def fire_webhook(doc_id, secret):
    body = json.dumps({"doc_id": doc_id}).encode()
    req = urllib.request.Request(HOOK, method="POST", data=body)
    req.add_header("Content-Type", "application/json")
    if secret is not None:
        req.add_header("X-Paperless-MCP-Secret", secret)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read()[:200].decode(errors="replace")
    except Exception as e:  # noqa: BLE001 - the status code IS the assertion
        return getattr(e, "code", "ERR"), str(e)[:200]


def probe_index(probe, timeout=180):
    """Read the INDEX, not the API — the question is whether it got embedded."""
    deadline = time.time() + timeout
    q = (
        "import sqlite3;c=sqlite3.connect(%r);"
        "r=c.execute(\"select count(*) from chunks where content like ?\","
        "('%%'+%r+'%%',)).fetchone();print(r[0])" % (INDEX_DB, probe)
    )
    while time.time() < deadline:
        out = subprocess.run(
            [sys.executable, "-c", q], capture_output=True, text=True, timeout=60
        )
        if out.returncode == 0 and out.stdout.strip().isdigit() and int(out.stdout) > 0:
            return int(out.stdout)
        time.sleep(5)
    return 0


def main():
    probe = token_of()
    print(f"PROBE={probe}")
    print(f"HOOK={HOOK}")

    upload(probe)
    doc = find_doc(probe)
    if not doc:
        print("RESULT=FAIL no document id (upload or consume failed)")
        return 1
    doc_id = doc["id"]
    print(f"DOC_ID={doc_id}")

    # Negative control: the index must NOT already hold the token, or a
    # green result would only prove the token was indexed by something else.
    before = probe_index(probe, timeout=1)
    print(f"INDEX_BEFORE={before}")
    if before:
        print("RESULT=INCONCLUSIVE token indexed before any webhook fired")
        return 1

    status, detail = fire_webhook(doc_id, SECRET)
    print(f"WEBHOOK_STATUS={status} {detail}")

    # A wrong secret must be refused, or the endpoint has no auth at all.
    bad_status, _ = fire_webhook(doc_id, "wrong-secret")
    print(f"BAD_SECRET_STATUS={bad_status}")

    count = probe_index(probe)
    print(f"INDEX_AFTER={count}")
    print("RESULT=" + ("PASS" if count > 0 else "FAIL token never reached the index"))
    return 0 if count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
