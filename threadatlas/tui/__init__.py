"""ASCII dashboard (curses TUI).

Read-only by design. Mutations (approve / delete / tag / etc.) remain
non-interactive CLI commands so the safety confirmations stay intact.
The TUI is a navigator, not a batch-remote-control.

The curses layer is intentionally thin: all data preparation lives in
:mod:`.models` and is tested without a TTY.
"""

from .app import run_tui, preview_screen
from .models import (
    ScreenModel,
    build_overview,
    build_conversations,
    build_groups,
    build_projects,
    build_open_loops,
    build_decisions,
    build_entities,
    build_conversation_detail,
)

__all__ = [
    "run_tui",
    "preview_screen",
    "ScreenModel",
    "build_overview",
    "build_conversations",
    "build_groups",
    "build_projects",
    "build_open_loops",
    "build_decisions",
    "build_entities",
    "build_conversation_detail",
]
