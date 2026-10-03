"""Bearer headers for local Midnight HTTP smoke and benchmark scripts."""

import os


def api_key():
    key = os.environ.get("MIDNIGHT_API_KEY")
    if not key:
        raise RuntimeError("Set MIDNIGHT_API_KEY before using a Midnight HTTP script.")
    return key


def authorization_headers():
    return {"Authorization": f"Bearer {api_key()}"}


def json_headers():
    return {"Content-Type": "application/json", **authorization_headers()}
