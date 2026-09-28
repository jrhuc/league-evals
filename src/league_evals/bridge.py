"""Client for `vgcleague bridge`, the harness's single-battle driver, speaking JSON lines on stdio."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections import deque
from contextlib import suppress
from itertools import count
from pathlib import Path
from typing import Any

FORMAT = "gen9championsvgc2026regmcbo3"


class BridgeError(Exception):
    """Bridge transport, protocol, or engine failure (an evaluation error)."""


class BridgeRejected(BridgeError):
    """A well-formed request was rejected; a model may correct it and retry."""


def league_dir() -> Path:
    configured = os.environ.get("LEAGUE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    return root / "engine" / "ai-draft-league" / "packages" / "league"


def bridge_command(directory: Path) -> list[str]:
    cli = directory / "dist" / "src" / "cli.js"
    if not cli.exists():
        raise FileNotFoundError(
            f"the league harness is not built at {directory}; run tools/setup-engine.sh "
            "or point LEAGUE_DIR at a built packages/league checkout"
        )
    return ["node", str(cli), "bridge"]


class LeagueBridge:
    """One harness process playing one battle. Requests are answered strictly in order."""

    def __init__(self, process: asyncio.subprocess.Process, timeout: float = 60) -> None:
        self._process = process
        self._ids = count(1)
        self._lock = asyncio.Lock()
        self._stderr: deque[str] = deque(maxlen=50)
        self._drain = asyncio.create_task(self._read_stderr())
        self.hello: dict[str, Any] = {}
        self.timeout = timeout

    @classmethod
    async def open(
        cls, format: str = FORMAT, sheets: str = "open", directory: Path | None = None
    ) -> LeagueBridge:
        directory = directory or league_dir()
        process = await asyncio.create_subprocess_exec(
            *bridge_command(directory),
            cwd=directory,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=16 * 1024 * 1024,
        )
        bridge = cls(process)
        try:
            bridge.hello = await bridge.request("open", {"format": format, "sheets": sheets})
            from .provenance import engine_provenance

            bridge.hello.update(engine_provenance(directory))
        except BaseException:
            await bridge.close()
            raise
        return bridge

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        assert self._process.stdin and self._process.stdout
        async with self._lock:
            ident = next(self._ids)
            message = json.dumps({"id": ident, "method": method, "params": params or {}})
            try:
                async with asyncio.timeout(self.timeout):
                    self._process.stdin.write(message.encode() + b"\n")
                    await self._process.stdin.drain()
                    line = await self._process.stdout.readline()
                if not line:
                    raise BridgeError("bridge exited: " + ("\n".join(self._stderr) or "no stderr"))
                reply = json.loads(line)
                if not isinstance(reply, dict) or reply.get("id") != ident:
                    raise BridgeError(f"bridge response ID mismatch for {method} ({ident})")
                if "error" in reply:
                    raise BridgeRejected(str(reply["error"]))
                if "result" not in reply:
                    raise BridgeError("bridge response is missing result")
                return reply["result"]
            except BridgeRejected:
                raise
            except (TimeoutError, OSError, ValueError, BridgeError) as error:
                # After a timeout or corrupt reply the stream cannot safely be reused.
                await self.close()
                raise BridgeError(f"bridge {method} failed: {error}") from error

    async def close(self) -> None:
        if self._process.returncode is None and self._process.stdin:
            self._process.stdin.close()
            try:
                await asyncio.wait_for(self._process.wait(), 5)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    self._process.kill()
                await self._process.wait()
        self._drain.cancel()
        with suppress(asyncio.CancelledError):
            await self._drain

    async def _read_stderr(self) -> None:
        assert self._process.stderr
        while line := await self._process.stderr.readline():
            self._stderr.append(line.decode(errors="replace").rstrip())


def request_sync(
    methods: list[tuple[str, dict[str, Any]]], directory: Path | None = None
) -> list[Any]:
    """One-shot requests where no game is played, such as reading a team pool."""
    directory = directory or league_dir()
    lines = "".join(
        json.dumps({"id": i, "method": m, "params": p}) + "\n"
        for i, (m, p) in enumerate(methods, 1)
    )
    completed = subprocess.run(
        bridge_command(directory),
        input=lines,
        capture_output=True,
        text=True,
        cwd=directory,
        check=False,
        timeout=60,
    )
    if completed.returncode or not completed.stdout:
        raise BridgeError(completed.stderr[-2000:] or "bridge produced no output")
    results = []
    for ident, line in enumerate(completed.stdout.splitlines(), 1):
        reply = json.loads(line)
        if reply.get("id") != ident:
            raise BridgeError("bridge response ID mismatch")
        if "error" in reply:
            raise BridgeRejected(reply["error"])
        results.append(reply["result"])
    if len(results) != len(methods):
        raise BridgeError("bridge returned an incomplete response sequence")
    return results
