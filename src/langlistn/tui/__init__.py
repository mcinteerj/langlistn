"""Textual TUI for langlistn."""

import asyncio
import json
import re
import time
from datetime import timedelta
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll

from ..realtime import EventKind
from textual.reactive import reactive
from textual.widgets import Footer, RichLog, Static

CONFIG_DIR = Path.home() / ".config" / "langlistn"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_THEME = "solarized-light"


def load_config() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:
        return {}


def save_config(data: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


class StatusBar(Static):
    """Bottom status bar."""

    status_text = reactive("starting")
    elapsed = reactive(0)
    log_path = reactive("")

    def render(self) -> str:
        t = str(timedelta(seconds=self.elapsed)).split(".")[0]
        parts = [f"◆ {self.status_text}", t]
        if self.log_path:
            parts.append(f"log: {self.log_path}")
        return " · ".join(parts)


class LiveText(Static):
    """In-progress streaming text that updates in place."""

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._text = ""

    def set_text(self, text: str) -> None:
        self._text = text
        self.update(text)

    def append(self, delta: str) -> None:
        self._text += delta
        self.update(self._text)

    def clear_text(self) -> str:
        text = self._text
        self._text = ""
        self.update("")
        return text


class TranslateApp(App):
    """langlistn TUI application."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #scroll-area {
        height: 1fr;
        border: round $accent;
        background: $background;
    }
    #output {
        height: auto;
        padding: 0 1;
        background: $background;
    }
    #live {
        height: auto;
        padding: 0 1;
        color: $text;
    }
    #status {
        height: auto;
        text-align: center;
        color: $text-muted;
    }
    #header-info {
        height: auto;
        text-align: center;
        color: $text;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("o", "toggle_original", "Original"),
        ("l", "toggle_log", "Log"),
        ("c", "clear_display", "Clear"),
    ]

    def __init__(
        self,
        source_lang: str | None = None,
        source_code: str | None = None,
        mode: str = "app",
        log_path: str | None = None,
        session=None,
        audio_source=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._source_lang = source_lang
        self._source_code = source_code
        self._mode = mode
        self._start_time = time.time()
        self._show_original = False
        self._log_file = None
        self._log_path = log_path
        self._session = session
        self._audio_source = audio_source
        self._bg_tasks: list[asyncio.Task] = []
        config = load_config()
        self.theme = config.get("theme", DEFAULT_THEME)
        if log_path:
            try:
                self._log_file = open(log_path, "a", encoding="utf-8")
            except OSError as e:
                self._log_file = None
                self._log_path = None
                # Will show error once TUI mounts
                self._init_error = f"Could not open log file: {e}"
            else:
                self._init_error = None
        else:
            self._init_error = None

    def compose(self) -> ComposeResult:
        if self._source_lang and self._source_code:
            src = f"{self._source_lang} ({self._source_code})"
        elif self._source_code:
            src = self._source_code
        else:
            src = None
        target = "English"
        lang_display = f"{src} → {target}" if src else f"auto-detect → {target}"
        yield Static(f"langlistn ── {lang_display} ── {self._mode}", id="header-info")
        with VerticalScroll(id="scroll-area"):
            yield RichLog(id="output", wrap=True, highlight=True, markup=True, auto_scroll=False)
            yield LiveText(id="live")
        yield StatusBar(id="status")
        yield Footer()

    async def on_mount(self) -> None:
        """Start background tasks once the TUI event loop is running."""
        self.set_interval(1.0, self._tick)
        if self._init_error:
            self.set_status(self._init_error)
        if self._session and self._audio_source:
            t1 = asyncio.create_task(self._run_background())
            self._bg_tasks.append(t1)

    async def _run_background(self) -> None:
        """Connect API and run audio + receive + watchdog concurrently."""
        try:
            self.set_status("connecting to API...")
            await self._session.connect()
            await asyncio.gather(
                self._audio_loop(),
                self._session.receive_loop(),
                self._receive_loop(),
                self._session.commit_watchdog(),
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.set_status(f"error: {e}")

    async def _audio_loop(self) -> None:
        """Read audio chunks and send to API."""
        try:
            self.set_status("starting audio capture...")
            await self._audio_source.start()
            self.set_status("audio capture started · streaming")
            chunk_count = 0
            while True:
                chunk = await self._audio_source.read_chunk()
                if chunk is None:
                    # Check if helper crashed
                    rc = self._audio_source.returncode
                    if rc is not None and rc != 0:
                        self.set_status(f"audio capture crashed (exit {rc})")
                    else:
                        self.set_status("audio source ended")
                    break
                await self._session.send_audio(chunk)
                chunk_count += 1
                if chunk_count == 10:
                    self.set_status(self._session.stats.status_line())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.set_status(f"audio error: {e}")
        finally:
            await self._audio_source.stop()

    async def _receive_loop(self) -> None:
        """Receive events from API session and update TUI."""
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        self._session.get_event(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                if event.kind == EventKind.TEXT:
                    self.append_text(event.data)
                elif event.kind == EventKind.TURN_COMPLETE:
                    self.finalize_segment()
                elif event.kind == EventKind.TRANSCRIPT:
                    self.append_transcript(event.data)
                elif event.kind == EventKind.STATUS:
                    self.set_status(event.data)
                elif event.kind == EventKind.ERROR:
                    self.set_status(f"error: {event.data}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.set_status(f"receive error: {e}")

    def watch_theme(self, new_theme: str) -> None:
        config = load_config()
        if config.get("theme") != new_theme:
            config["theme"] = new_theme
            save_config(config)

    def _tick(self) -> None:
        status = self.query_one("#status", StatusBar)
        status.elapsed = int(time.time() - self._start_time)

    def set_status(self, text: str) -> None:
        try:
            status = self.query_one("#status", StatusBar)
            status.status_text = text
        except Exception:
            pass

    def append_text(self, text: str) -> None:
        """Stream text delta — shows immediately in live widget."""
        try:
            live = self.query_one("#live", LiveText)
            live.append(text)
            scroll = self.query_one("#scroll-area", VerticalScroll)
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    def append_transcript(self, text: str) -> None:
        if self._show_original:
            try:
                log = self.query_one("#output", RichLog)
                log.write(f"[dim italic]〈{text}〉[/]")
            except Exception:
                pass

    def finalize_segment(self) -> None:
        """Move in-progress text to the history log."""
        try:
            live = self.query_one("#live", LiveText)
            text = live.clear_text().strip()
            # Drop incomplete speaker labels (e.g. bare "Speaker" or "Speaker 1:")
            import re
            cleaned = re.sub(r"^Speaker\s*\d*:?\s*$", "", text, flags=re.IGNORECASE).strip()
            if cleaned:
                log = self.query_one("#output", RichLog)
                log.write(text)
                scroll = self.query_one("#scroll-area", VerticalScroll)
                scroll.scroll_end(animate=False)
                if self._log_file:
                    self._log_file.write(text + "\n")
                    self._log_file.flush()
        except Exception:
            pass

    def action_toggle_original(self) -> None:
        self._show_original = not self._show_original
        state = "on" if self._show_original else "off"
        self.set_status(f"original language {state}")

    def action_toggle_log(self) -> None:
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            self._log_path = None
            status = self.query_one("#status", StatusBar)
            status.log_path = ""
            self.set_status("logging off")
        else:
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = str(Path.home() / f"langlistn-{ts}.txt")
            try:
                self._log_file = open(path, "a", encoding="utf-8")
            except OSError as e:
                self.set_status(f"log error: {e}")
                return
            self._log_path = path
            status = self.query_one("#status", StatusBar)
            status.log_path = path
            self.set_status("logging on")

    def action_clear_display(self) -> None:
        try:
            log = self.query_one("#output", RichLog)
            log.clear()
        except Exception:
            pass

    async def action_quit(self) -> None:
        """Override quit to ensure cleanup runs."""
        await self.cleanup()
        self.exit()

    async def cleanup(self) -> None:
        for task in self._bg_tasks:
            task.cancel()
        # Wait for tasks to actually finish
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        if self._session:
            try:
                await asyncio.wait_for(self._session.shutdown(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass
        if self._audio_source:
            await self._audio_source.stop()
        if self._log_file:
            self._log_file.close()
            self._log_file = None
