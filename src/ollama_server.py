import atexit
import os
import subprocess
import time
import sys

import requests

from src.config import OLLAMA_ENV

URL = "http://localhost:11434"
_process = None   # the server we started, if any


def _is_up() -> bool:
    try:
        requests.get(URL, timeout=1)
        return True
    except requests.RequestException:
        return False


def ensure_server() -> str:
    """Make sure an Ollama server is running.
    Returns "already_running" or "started" so the caller can tell the user."""
    global _process
    if _is_up():
        return "already_running"          # settings can't be applied to it

    env = {**os.environ, **OLLAMA_ENV}
    _process = subprocess.Popen(
        ["ollama", "serve"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    atexit.register(stop_server)

    for _ in range(30):                   # wait up to ~15 s for it to come up
        if _is_up():
            return "started"
        time.sleep(0.5)
    raise RuntimeError("Ollama did not start. Is it installed and on your PATH?")


def stop_server():
    if _process and _process.poll() is None:
        if sys.platform == "win32":
            # /T kills the child runner processes too, so VRAM is released
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(_process.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            _process.terminate()