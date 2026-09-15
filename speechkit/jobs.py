"""
Running one analysis at a time, in the background, with a readable log.

A batch of twenty recordings takes minutes, so the interface cannot call the
analysis inside a request handler: the browser would sit on a dead connection
with nothing to show. Each stage runs on a worker thread instead and appends to
a log the page polls.

One job at a time on purpose. All three stages are CPU-bound and two of them
load large models, so running them concurrently on a laptop makes both slower
and the memory use unpredictable.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable


class Job:
    """One run of one stage."""

    def __init__(self, stage: str, target: Callable[..., Any], cfg: dict[str, Any]):
        self.stage = stage
        self.target = target
        self.cfg = cfg
        self.lines: list[str] = []
        self.state = "queued"          # queued | running | done | failed | cancelled
        self.error: str | None = None
        self.traceback: str | None = None
        self.result: Any = None
        self.started: float | None = None
        self.finished: float | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # -- logging ---------------------------------------------------------
    def log(self, *parts: Any) -> None:
        text = " ".join(str(p) for p in parts)
        with self._lock:
            for line in text.split("\n"):
                self.lines.append(line)
            # A runaway loop should not exhaust memory; keep the tail.
            if len(self.lines) > 20000:
                del self.lines[: len(self.lines) - 20000]

    def tail(self, since: int = 0) -> tuple[list[str], int]:
        with self._lock:
            return self.lines[since:], len(self.lines)

    # -- control ---------------------------------------------------------
    def should_stop(self) -> bool:
        return self._stop.is_set()

    def cancel(self) -> None:
        """
        Ask the job to stop. Stages check between files, so a cancel takes
        effect after the current recording rather than instantly. Nothing is
        killed mid-write, so no half-written CSV is left behind.
        """
        self._stop.set()
        self.log("[cancel] finishing the current file, then stopping")

    def start(self) -> None:
        self.state = "running"
        self.started = time.time()
        self._thread = threading.Thread(target=self._run, name=f"speechkit-{self.stage}",
                                        daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.result = self.target(self.cfg, self.log, self.should_stop)
            self.state = "cancelled" if self.should_stop() else "done"
        except Exception as exc:
            self.state = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            self.traceback = traceback.format_exc()
            self.log(f"[error] {self.error}")
            self.log("The full traceback is in the terminal running the interface.")
            print(self.traceback)
        finally:
            self.finished = time.time()

    # -- reporting -------------------------------------------------------
    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started

    def status(self, since: int = 0) -> dict[str, Any]:
        lines, total = self.tail(since)
        return {
            "stage": self.stage,
            "state": self.state,
            "error": self.error,
            "elapsed": round(self.elapsed, 1),
            "lines": lines,
            "cursor": total,
            "running": self.state == "running",
        }


class JobRunner:
    """Holds the current job and refuses to start a second one."""

    def __init__(self) -> None:
        self.current: Job | None = None
        self._lock = threading.Lock()

    def busy(self) -> bool:
        return self.current is not None and self.current.state == "running"

    def start(self, stage: str, target: Callable[..., Any], cfg: dict[str, Any]) -> Job:
        with self._lock:
            if self.busy():
                raise RuntimeError(
                    f"{self.current.stage} is still running. Wait for it to finish, "
                    f"or stop it first.")
            job = Job(stage, target, cfg)
            self.current = job
        job.start()
        return job

    def status(self, since: int = 0) -> dict[str, Any]:
        if self.current is None:
            return {"stage": None, "state": "idle", "lines": [], "cursor": 0,
                    "running": False, "elapsed": 0.0, "error": None}
        return self.current.status(since)

    def cancel(self) -> bool:
        if self.busy():
            self.current.cancel()
            return True
        return False
