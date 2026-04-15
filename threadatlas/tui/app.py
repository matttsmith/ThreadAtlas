"""Curses TUI.

Key bindings (documented on the `?` help screen):

  1..8       jump to screen by index
  Tab        next screen        Shift-Tab  previous screen
  Up/Down    move selection     j/k        move selection (vim-style)
  PgUp/PgDn  jump by page       Home/End   first/last row
  Enter      drill into detail (where applicable)
  Esc/BS     go back from a detail screen
  /          filter (conversations + groups screens)
  s          cycle state filter on conversations
  l          cycle level filter on groups
  r          refresh current screen
  ?          help overlay
  q          quit

Terminal must be at least 60 cols x 10 rows. Smaller is rejected with a
prompt to resize.
"""

from __future__ import annotations

import curses

from ..core.vault import Vault, open_vault
from ..store import Store, open_store
from . import models
from .models import ScreenModel


_MIN_COLS = 60
_MIN_ROWS = 10


# Screen registry: key -> (label, builder). Some builders take extra state
# passed via a context dict that the app mutates (state_filter, etc.).
_SCREEN_ORDER = (
    "overview",
    "conversations",
    "groups",
    "projects",
    "open_loops",
    "decisions",
    "entities",
    "help",
)

_SCREEN_LABELS = {
    "overview":      "1:Overview",
    "conversations": "2:Conversations",
    "groups":        "3:Groups",
    "projects":      "4:Projects",
    "open_loops":    "5:Open loops",
    "decisions":     "6:Decisions",
    "entities":      "7:Entities",
    "help":          "8:Help",
}


def _build(screen: str, *, vault: Vault, store: Store, ctx: dict) -> ScreenModel:
    if screen == "overview":
        return models.build_overview(vault, store)
    if screen == "conversations":
        return models.build_conversations(
            store,
            state_filter=ctx.get("conv_state_filter"),
            query=ctx.get("conv_query"),
        )
    if screen == "groups":
        return models.build_groups(store, level=ctx.get("group_level"))
    if screen == "projects":
        return models.build_projects(store)
    if screen == "open_loops":
        return models.build_open_loops(store)
    if screen == "decisions":
        return models.build_decisions(store)
    if screen == "entities":
        return models.build_entities(store)
    if screen == "help":
        return _help_screen()
    raise ValueError(f"Unknown screen: {screen!r}")


def _help_screen() -> ScreenModel:
    lines = [l.strip() for l in __doc__.splitlines() if l.strip() and not l.startswith('"""')]
    return ScreenModel(
        title="Help",
        columns=["key / behavior"],
        rows=[{"cells": [l]} for l in lines],
        footer="Press q to quit, any other key to return.",
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def preview_screen(vault_path, screen: str) -> str:
    """Render a single screen to plain text without invoking curses.

    Useful for smoke tests and for operators who just want a snapshot
    on stdout (``threadatlas tui --preview overview``).
    """
    vault = open_vault(vault_path)
    store = open_store(vault)
    try:
        if screen == "help":
            model = _help_screen()
        else:
            model = _build(screen, vault=vault, store=store, ctx={})
        return _model_to_text(model, width=100)
    finally:
        store.close()


def run_tui(vault_path) -> int:
    """Launch the interactive curses TUI. Read-only."""
    vault = open_vault(vault_path)
    store = open_store(vault)
    try:
        curses.wrapper(_main, vault, store)
    finally:
        store.close()
    return 0


# ---------------------------------------------------------------------------
# Curses rendering
# ---------------------------------------------------------------------------

def _init_colors() -> dict:
    colors = {
        "header":         0,
        "footer":         0,
        "selected":       0,
        "state_indexed":  0,
        "state_private":  0,
        "state_pending":  0,
        "state_quaran":   0,
        "muted":          0,
    }
    if not curses.has_colors():
        return colors
    try:
        curses.start_color()
        curses.use_default_colors()
    except curses.error:
        return colors
    try:
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(2, curses.COLOR_CYAN, -1)
        curses.init_pair(3, curses.COLOR_GREEN, -1)
        curses.init_pair(4, curses.COLOR_MAGENTA, -1)
        curses.init_pair(5, curses.COLOR_YELLOW, -1)
        curses.init_pair(6, curses.COLOR_RED, -1)
        curses.init_pair(7, -1, -1)
    except curses.error:
        return colors
    colors["header"] = curses.color_pair(1) | curses.A_BOLD
    colors["footer"] = curses.color_pair(2)
    colors["selected"] = curses.A_REVERSE
    colors["state_indexed"] = curses.color_pair(3)
    colors["state_private"] = curses.color_pair(4)
    colors["state_pending"] = curses.color_pair(5)
    colors["state_quaran"] = curses.color_pair(6)
    colors["muted"] = curses.A_DIM
    return colors


_STATE_COLOR_KEY = {
    "indexed": "state_indexed",
    "private": "state_private",
    "pending_review": "state_pending",
    "quarantined": "state_quaran",
}


def _compute_column_widths(model: ScreenModel, max_w: int) -> list[int]:
    """Balance columns so they fit within ``max_w``.

    Strategy: measure natural width per column (max of header + cells),
    then scale uniformly if the total exceeds ``max_w``. Minimum 4 chars
    per column.
    """
    cols = model.columns or [""]
    n = len(cols)
    widths = []
    for i, col in enumerate(cols):
        w = len(col)
        for row in model.rows:
            cells = row.get("cells") or []
            if i < len(cells):
                w = max(w, len(str(cells[i])))
        widths.append(max(w, 4))
    gap = 2
    total = sum(widths) + gap * (n - 1)
    if total <= max_w:
        return widths
    # Scale down proportionally, keeping at least 4 per column.
    budget = max_w - gap * (n - 1)
    overflow = total - gap * (n - 1) - budget
    # Shrink from the widest columns first.
    order = sorted(range(n), key=lambda i: widths[i], reverse=True)
    for i in order:
        if overflow <= 0:
            break
        take = min(widths[i] - 4, overflow)
        widths[i] -= take
        overflow -= take
    return widths


def _format_row(cells: list[str], widths: list[int]) -> str:
    out = []
    for i, w in enumerate(widths):
        cell = str(cells[i]) if i < len(cells) else ""
        if len(cell) > w:
            cell = cell[: max(w - 1, 1)] + "\u2026"
        out.append(cell.ljust(w))
    return "  ".join(out)


def _paint(win, y: int, x: int, text: str, attr: int, max_w: int) -> None:
    """Write ``text`` clamped to ``max_w`` columns starting at (y, x)."""
    if max_w <= 0:
        return
    text = text[:max_w]
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        # Writing the last cell of the last line raises; harmless.
        pass


def _model_to_text(model: ScreenModel, *, width: int = 100) -> str:
    widths = _compute_column_widths(model, width)
    lines: list[str] = []
    lines.append(f"== {model.title} ==")
    lines.append(_format_row(model.columns, widths))
    for row in model.rows:
        lines.append(_format_row(row.get("cells", []), widths))
    if model.footer:
        lines.append("")
        lines.append(model.footer)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class _AppState:
    def __init__(self):
        self.screen_idx = 0
        self.selected = 0
        self.top = 0  # first visible row (for scrolling)
        self.ctx: dict = {
            "conv_state_filter": None,
            "conv_query": None,
            "group_level": None,
        }
        self.stack: list[tuple[int, int, int, dict, str | None, ScreenModel | None]] = []
        # override_model lets us push a detail screen that isn't in the registry.
        self.override_model: ScreenModel | None = None
        self.override_label: str | None = None

    @property
    def current_screen_key(self) -> str:
        return _SCREEN_ORDER[self.screen_idx]


_STATE_CYCLE = (None, "pending_review", "indexed", "private", "quarantined")


def _cycle_conv_state(ctx: dict) -> None:
    cur = ctx.get("conv_state_filter")
    idx = _STATE_CYCLE.index(cur) if cur in _STATE_CYCLE else 0
    ctx["conv_state_filter"] = _STATE_CYCLE[(idx + 1) % len(_STATE_CYCLE)]


_LEVEL_CYCLE = (None, "broad", "fine")


def _cycle_group_level(ctx: dict) -> None:
    cur = ctx.get("group_level")
    idx = _LEVEL_CYCLE.index(cur) if cur in _LEVEL_CYCLE else 0
    ctx["group_level"] = _LEVEL_CYCLE[(idx + 1) % len(_LEVEL_CYCLE)]


def _prompt(stdscr, prompt: str) -> str:
    """Modal single-line input. Returns "" if cancelled."""
    h, w = stdscr.getmaxyx()
    curses.curs_set(1)
    stdscr.move(h - 1, 0)
    stdscr.clrtoeol()
    stdscr.addstr(h - 1, 0, prompt)
    curses.echo()
    try:
        s = stdscr.getstr(h - 1, len(prompt), max(w - len(prompt) - 1, 4)).decode("utf-8", "replace")
    except Exception:
        s = ""
    finally:
        curses.noecho()
        curses.curs_set(0)
    return s.strip()


def _drill(store: Store, vault: Vault, app: _AppState, current_model: ScreenModel) -> bool:
    """Drill into the selected row if it has an actionable id. Returns True if a drill happened."""
    if app.selected < 0 or app.selected >= len(current_model.rows):
        return False
    row = current_model.rows[app.selected]
    rid = row.get("id")
    if not rid:
        return False
    # Where to drill depends on the current screen.
    scr = app.current_screen_key
    if scr == "conversations":
        app.stack.append((app.screen_idx, app.selected, app.top, dict(app.ctx),
                          app.override_label, app.override_model))
        app.override_model = models.build_conversation_detail(vault, store, rid)
        app.override_label = f"conv:{rid[:16]}"
        app.selected = 0
        app.top = 0
        return True
    if scr == "groups":
        app.stack.append((app.screen_idx, app.selected, app.top, dict(app.ctx),
                          app.override_label, app.override_model))
        app.override_model = models.build_group_members(store, rid)
        app.override_label = f"grp:{rid[:16]}"
        app.selected = 0
        app.top = 0
        return True
    if scr in ("projects", "open_loops", "decisions", "entities"):
        app.stack.append((app.screen_idx, app.selected, app.top, dict(app.ctx),
                          app.override_label, app.override_model))
        app.override_model = models.build_object_detail(store, rid)
        app.override_label = f"obj:{rid[:16]}"
        app.selected = 0
        app.top = 0
        return True
    return False


def _pop(app: _AppState) -> bool:
    if not app.stack:
        return False
    idx, sel, top, ctx, label, model = app.stack.pop()
    app.screen_idx = idx
    app.selected = sel
    app.top = top
    app.ctx = ctx
    app.override_label = label
    app.override_model = model
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _main(stdscr, vault: Vault, store: Store) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.timeout(-1)  # blocking getch
    colors = _init_colors()
    app = _AppState()

    while True:
        h, w = stdscr.getmaxyx()
        stdscr.erase()
        if h < _MIN_ROWS or w < _MIN_COLS:
            msg = f"Terminal too small ({w}x{h}). Resize to at least {_MIN_COLS}x{_MIN_ROWS}. q to quit."
            _paint(stdscr, 0, 0, msg[:w], 0, w)
            stdscr.refresh()
            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q")):
                return
            continue

        # Build or reuse model for current screen.
        if app.override_model is not None:
            model = app.override_model
            screen_label = app.override_label or model.title
        else:
            try:
                model = _build(app.current_screen_key, vault=vault, store=store, ctx=app.ctx)
            except Exception as e:
                model = ScreenModel(title="Error", columns=["error"],
                                    rows=[{"cells": [repr(e)]}])
            screen_label = _SCREEN_LABELS[app.current_screen_key]

        # Compute layout.
        body_top = 2  # header line + column headers
        body_bottom = h - 2  # footer line + status
        body_height = max(body_bottom - body_top, 1)
        content_width = w

        # Header: highlight current screen in brackets when not in a drill-down.
        header_parts: list[str] = []
        for i, k in enumerate(_SCREEN_ORDER):
            label = _SCREEN_LABELS[k]
            if app.override_model is None and i == app.screen_idx:
                header_parts.append(f"[{label}]")
            else:
                header_parts.append(label)
        header = "ThreadAtlas  " + "  ".join(header_parts)
        _paint(stdscr, 0, 0, header, colors["header"], w)

        # Column headers (if this screen has meaningful columns).
        widths = _compute_column_widths(model, content_width)
        col_line = _format_row(model.columns, widths) if model.columns else ""
        _paint(stdscr, 1, 0, col_line, curses.A_BOLD if model.columns else 0, w)

        # Rows with scrolling.
        # Clamp selection.
        n = len(model.rows)
        if n:
            if app.selected >= n:
                app.selected = n - 1
            if app.selected < 0:
                app.selected = 0
            if app.selected < app.top:
                app.top = app.selected
            if app.selected >= app.top + body_height:
                app.top = app.selected - body_height + 1
        else:
            app.top = 0
            app.selected = 0

        for i in range(body_height):
            ri = app.top + i
            if ri >= n:
                break
            row = model.rows[ri]
            line = _format_row(row.get("cells", []), widths)
            attr = 0
            state = row.get("state")
            if state and state in _STATE_COLOR_KEY:
                attr = colors.get(_STATE_COLOR_KEY[state], 0)
            if ri == app.selected and row.get("id"):
                attr |= colors["selected"]
            _paint(stdscr, body_top + i, 0, line, attr, w)

        # Footer.
        status = [
            screen_label,
        ]
        if app.ctx.get("conv_state_filter"):
            status.append(f"state={app.ctx['conv_state_filter']}")
        if app.ctx.get("conv_query"):
            status.append(f"q='{app.ctx['conv_query']}'")
        if app.ctx.get("group_level"):
            status.append(f"level={app.ctx['group_level']}")
        status_line = " | ".join(status) + (f"   {model.footer}" if model.footer else "")
        _paint(stdscr, h - 2, 0, "\u2500" * w, colors["footer"], w)
        _paint(stdscr, h - 1, 0, status_line, colors["footer"], w)

        stdscr.refresh()
        ch = stdscr.getch()

        # --- key handling ---
        if ch in (ord("q"), ord("Q")):
            return
        if ch == curses.KEY_RESIZE:
            try:
                curses.update_lines_cols()
            except AttributeError:
                pass
            continue
        if ch in (curses.KEY_UP, ord("k")):
            app.selected = max(0, app.selected - 1)
            continue
        if ch in (curses.KEY_DOWN, ord("j")):
            app.selected = min(max(n - 1, 0), app.selected + 1)
            continue
        if ch == curses.KEY_PPAGE:
            app.selected = max(0, app.selected - body_height)
            continue
        if ch == curses.KEY_NPAGE:
            app.selected = min(max(n - 1, 0), app.selected + body_height)
            continue
        if ch == curses.KEY_HOME:
            app.selected = 0
            continue
        if ch == curses.KEY_END:
            app.selected = max(n - 1, 0)
            continue
        if ch == 9:  # Tab
            if app.override_model is not None:
                _pop(app)
            app.screen_idx = (app.screen_idx + 1) % len(_SCREEN_ORDER)
            app.selected = 0
            app.top = 0
            continue
        if ch == curses.KEY_BTAB:
            if app.override_model is not None:
                _pop(app)
            app.screen_idx = (app.screen_idx - 1) % len(_SCREEN_ORDER)
            app.selected = 0
            app.top = 0
            continue
        if ord("1") <= ch <= ord("8"):
            idx = ch - ord("1")
            if idx < len(_SCREEN_ORDER):
                if app.override_model is not None:
                    _pop(app)
                app.screen_idx = idx
                app.selected = 0
                app.top = 0
            continue
        if ch in (10, 13, curses.KEY_ENTER):
            _drill(store, vault, app, model)
            continue
        if ch in (27, curses.KEY_BACKSPACE, 127, 8):
            _pop(app)
            continue
        if ch == ord("r"):
            # Force rebuild by looping; no-op here since we rebuild every tick
            # when override_model is None. If we're in a detail view, rebuild it.
            if app.override_model is not None and app.override_label:
                key, _, rid = (app.override_label.partition(":"))
                if key == "conv":
                    app.override_model = models.build_conversation_detail(vault, store, rid)
                elif key == "grp":
                    app.override_model = models.build_group_members(store, rid)
                elif key == "obj":
                    app.override_model = models.build_object_detail(store, rid)
            continue
        if ch == ord("s") and app.current_screen_key == "conversations" and app.override_model is None:
            _cycle_conv_state(app.ctx)
            app.selected = 0
            app.top = 0
            continue
        if ch == ord("l") and app.current_screen_key == "groups" and app.override_model is None:
            _cycle_group_level(app.ctx)
            app.selected = 0
            app.top = 0
            continue
        if ch == ord("/"):
            if app.current_screen_key == "conversations" and app.override_model is None:
                q = _prompt(stdscr, "filter title/summary: ")
                app.ctx["conv_query"] = q or None
                app.selected = 0
                app.top = 0
            continue
        if ch == ord("?"):
            app.screen_idx = _SCREEN_ORDER.index("help")
            app.selected = 0
            app.top = 0
            continue
        # Unknown key; ignore.
