"""Lightweight operation timer for profiling GPU-bound forward passes.

Accumulates per-operation timing data across many calls and produces
a breakdown report showing where time goes inside the model.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)


class OperationTimer:
    """Accumulates timing data for named operations.

    Usage::

        timer = OperationTimer(sync_mps=True)

        with timer.measure("projection"):
            output = W_Q(x)

        timer.print_report()

    Args:
        enabled:  If False, ``measure()`` is a no-op.
        sync_mps: If True, call ``torch.mps.synchronize()`` around
                  each measurement for accurate GPU timings.  Adds
                  overhead — use only during profiling runs.
    """

    def __init__(
        self, enabled: bool = True, sync_mps: bool = False
    ) -> None:
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._enabled = enabled
        self._sync_mps = sync_mps and torch.backends.mps.is_available()

    @contextmanager
    def measure(self, op_name: str):
        """Context manager that times the enclosed block.

        Args:
            op_name: Label for this operation (e.g. ``"gather_blocks"``).
        """
        if not self._enabled:
            yield
            return

        if self._sync_mps:
            torch.mps.synchronize()
        t0 = time.perf_counter()

        yield

        if self._sync_mps:
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        self._timings[op_name].append(dt)

    def report(self) -> dict[str, dict]:
        """Return per-operation stats as a dict.

        Returns:
            ``{op_name: {"calls": int, "mean_ms": float,
            "total_ms": float, "pct": float}}``.
        """
        grand_total = sum(sum(v) for v in self._timings.values())
        result: dict[str, dict] = {}

        for name, durations in self._timings.items():
            total_s = sum(durations)
            count = len(durations)
            result[name] = {
                "calls": count,
                "mean_ms": (total_s / count) * 1000 if count else 0.0,
                "total_ms": total_s * 1000,
                "pct": (total_s / grand_total) * 100 if grand_total > 0 else 0.0,
            }

        return result

    def print_report(self, title: str = "Operation Timing Report") -> None:
        """Print a rich table sorted by total time descending."""
        console = Console()
        data = self.report()

        if not data:
            console.print("[yellow]No timing data collected.[/]")
            return

        table = Table(title=title, show_lines=True)
        table.add_column("Operation", style="bold")
        table.add_column("Calls", justify="right")
        table.add_column("Mean (ms)", justify="right")
        table.add_column("Total (ms)", justify="right")
        table.add_column("%", justify="right")

        sorted_ops = sorted(data.items(), key=lambda x: x[1]["total_ms"], reverse=True)
        for name, stats in sorted_ops:
            pct = stats["pct"]
            if pct > 40:
                style = "bold red"
                flag = " 🔴"
            elif pct > 20:
                style = "bold yellow"
                flag = " 🟡"
            else:
                style = ""
                flag = ""

            table.add_row(
                name,
                str(stats["calls"]),
                f"{stats['mean_ms']:.3f}",
                f"{stats['total_ms']:.1f}",
                f"{pct:.1f}%{flag}",
                style=style,
            )

        # Total row.
        grand_total_ms = sum(s["total_ms"] for s in data.values())
        table.add_row(
            "TOTAL measured", "", "", f"{grand_total_ms:.1f}", "100%",
            style="bold cyan",
        )

        console.print(table)

    def reset(self) -> None:
        """Clear all accumulated timings."""
        self._timings.clear()
