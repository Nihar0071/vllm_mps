"""Live terminal dashboard for monitoring the vllm_mps engine.

Uses the ``rich`` library to render a live-updating display that reads
engine state from a background thread without blocking generation.
"""

from __future__ import annotations

import logging
import threading
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from vllm_mps.config import PROFILER_INTERVAL_SEC

logger = logging.getLogger(__name__)


class LiveDashboard:
    """Background-thread dashboard that reads LLMEngine state.

    Usage::

        dashboard = LiveDashboard()
        dashboard.start(engine)
        # ... run generation ...
        dashboard.stop()
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._console = Console()
        self._step_count = 0

    def start(self, engine) -> None:
        """Start the live dashboard in a background thread.

        Args:
            engine: An LLMEngine instance (read-only access).
        """
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(engine,), daemon=True
        )
        self._thread.start()
        logger.info("LiveDashboard: started")

    def stop(self) -> None:
        """Stop the dashboard cleanly."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("LiveDashboard: stopped")

    def _run(self, engine) -> None:
        """Background thread loop."""
        try:
            with Live(
                self._render(engine),
                console=self._console,
                refresh_per_second=2,
                screen=False,
            ) as live:
                while not self._stop_event.is_set():
                    self._step_count += 1
                    live.update(self._render(engine))
                    time.sleep(PROFILER_INTERVAL_SEC)
        except Exception as e:
            logger.error("LiveDashboard: error — %s", e)

    def _render(self, engine) -> Table:
        """Build the dashboard table from engine state."""
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)

        # ── Header ────────────────────────────────────────────────────
        header = Text("⚡ vLLM-MPS Live Dashboard", style="bold cyan")

        # ── Memory panel ──────────────────────────────────────────────
        kv = engine.kv_cache_manager
        mem_lines = [
            f"GPU util:  {kv.get_gpu_utilisation():.1%}",
            f"CPU util:  {kv.get_cpu_utilisation():.1%}",
            f"Free GPU:  {kv.get_num_free_gpu_blocks()}",
            f"Free CPU:  {kv.get_num_free_cpu_blocks()}",
            f"Pool:      {engine.memory_pool.get_memory_mb():.2f} MB",
        ]
        mem_panel = Panel("\n".join(mem_lines), title="Memory", border_style="blue")

        # ── Scheduler panel ───────────────────────────────────────────
        sched = engine.scheduler
        sched_lines = [
            f"waiting:   {sched.get_num_waiting()}",
            f"running:   {sched.get_num_running()}",
            f"preempted: {sched.get_num_preempted()}",
        ]
        sched_panel = Panel(
            "\n".join(sched_lines), title="Scheduler", border_style="yellow"
        )

        # ── Throughput panel ──────────────────────────────────────────
        tput_lines = [
            f"tokens/sec: {engine.get_tokens_per_sec():.1f}",
            f"total:      {engine.total_tokens_generated}",
            f"uptime:     {engine.get_elapsed():.1f}s",
        ]
        tput_panel = Panel(
            "\n".join(tput_lines), title="Throughput", border_style="green"
        )

        # ── Active sequences ──────────────────────────────────────────
        seq_lines: list[str] = []
        for group in sched._running:
            for s in group.sequences:
                seq_lines.append(
                    f"seq_{s.seq_id}: {s.get_total_len()} tokens  "
                    f"{s.status.value.upper()}"
                )
        if not seq_lines:
            seq_lines.append("(none)")
        seq_panel = Panel(
            "\n".join(seq_lines[:8]),  # Cap at 8 lines.
            title="Active Sequences",
            border_style="magenta",
        )

        # ── Assemble ──────────────────────────────────────────────────
        # Use a two-column table for compact layout.
        inner = Table.grid(expand=True)
        inner.add_column(ratio=1)
        inner.add_column(ratio=1)
        inner.add_row(mem_panel, sched_panel)
        inner.add_row(tput_panel, seq_panel)

        grid.add_row(Panel(header, border_style="cyan"))
        grid.add_row(inner)

        return grid
