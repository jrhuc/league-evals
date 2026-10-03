"""Client for `vgcleague bridge`, which plays one battle or one league over JSON lines on stdio."""

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
EXTERNAL = "external"


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
    """One harness process playing one battle or one league. Replies and events interleave on its
    stdout."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        timeout: float = 300,
        event_timeout: float = 600,
    ) -> None:
        """A search opponent holds the engine's event loop while it thinks, so a reply can wait
        for the opponent's whole turn."""
        self._process = process
        self._ids = count(1)
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._events: asyncio.Queue[dict[str, Any] | BridgeError] = asyncio.Queue()
        self._failure: BridgeError | None = None
        self._stderr: deque[str] = deque(maxlen=50)
        self._drain = asyncio.create_task(self._read_stderr())
        self._reader = asyncio.create_task(self._read_stdout())
        self.hello: dict[str, Any] = {}
        self.timeout = timeout
        self.event_timeout = event_timeout

    @classmethod
    async def open(cls, format: str = FORMAT, directory: Path | None = None) -> LeagueBridge:
        return await cls._session("open", {"format": format}, directory)

    @classmethod
    async def league(
        cls, params: dict[str, Any], directory: Path | None = None, event_timeout: float = 3600
    ) -> LeagueBridge:
        """A season has long stretches without a task for the outside seat, such as a final
        between two bots."""
        return await cls._session("league", params, directory, event_timeout)

    @classmethod
    async def _session(
        cls,
        method: str,
        params: dict[str, Any],
        directory: Path | None,
        event_timeout: float = 600,
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
        bridge = cls(process, event_timeout=event_timeout)
        try:
            bridge.hello = await bridge.request(method, params)
            from .provenance import engine_provenance

            bridge.hello.update(engine_provenance(directory))
        except BaseException:
            await bridge.close()
            raise
        return bridge

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        assert self._process.stdin
        ident = next(self._ids)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        message = json.dumps({"id": ident, "method": method, "params": params or {}})
        try:
            if self._failure is not None:
                raise self._failure
            self._pending[ident] = future
            async with asyncio.timeout(self.timeout):
                self._process.stdin.write(message.encode() + b"\n")
                await self._process.stdin.drain()
                reply = await future
            if "error" not in reply and "result" not in reply:
                raise BridgeError("bridge response is missing result")
        except (TimeoutError, OSError, BridgeError) as error:
            await self.close()
            raise BridgeError(f"bridge {method} failed: {error}") from error
        finally:
            self._pending.pop(ident, None)
        if "error" in reply:
            raise BridgeRejected(str(reply["error"]))
        return reply["result"]

    async def next_event(self) -> dict[str, Any]:
        try:
            async with asyncio.timeout(self.event_timeout):
                event = await self._events.get()
        except TimeoutError as error:
            await self.close()
            raise BridgeError("bridge sent no event in time") from error
        if isinstance(event, BridgeError):
            self._events.put_nowait(event)
            await self.close()
            raise event
        return event

    async def close(self) -> None:
        if self._process.returncode is None and self._process.stdin:
            self._process.stdin.close()
            try:
                await asyncio.wait_for(self._process.wait(), 5)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    self._process.kill()
                await self._process.wait()
        for task in (self._reader, self._drain):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._fail(BridgeError("bridge closed"))

    def _fail(self, error: BridgeError) -> None:
        if self._failure is not None:
            return
        self._failure = error
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._events.put_nowait(error)

    async def _read_stdout(self) -> None:
        assert self._process.stdout
        try:
            while line := await self._process.stdout.readline():
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise BridgeError("bridge sent a line that is not an object")
                if "event" in message:
                    self._events.put_nowait(message["event"])
                    continue
                future = self._pending.get(message.get("id"))
                if future is None or future.done():
                    raise BridgeError(f"bridge response ID mismatch ({message.get('id')})")
                future.set_result(message)
            await asyncio.wait({self._drain}, timeout=1)
            raise BridgeError("bridge exited: " + ("\n".join(self._stderr) or "no stderr"))
        except BridgeError as error:
            self._fail(error)
        except (OSError, TypeError, ValueError) as error:
            self._fail(BridgeError(f"bridge output unreadable: {error}"))

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
    replies = [json.loads(line) for line in completed.stdout.splitlines()]
    results = []
    for ident, reply in enumerate((r for r in replies if "event" not in r), 1):
        if reply.get("id") != ident:
            raise BridgeError("bridge response ID mismatch")
        if "error" in reply:
            raise BridgeRejected(reply["error"])
        results.append(reply["result"])
    if len(results) != len(methods):
        raise BridgeError("bridge returned an incomplete response sequence")
    return results
