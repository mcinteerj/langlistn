# Group A: Quick Rerun + Exit Summary + Header

## 1. Quick Rerun

**Where:** `__main__.py` → `main()`, between arg parsing and interactive setup.

**Logic:** When no `--app`/`--mic` given AND `_LAST_SESSION` exists with at least `app` or `mic` key:

```python
def _quick_rerun_prompt(last: dict) -> bool | None:
    """Show last-session prompt. Returns True=reuse, False=wizard, None=no last."""
```

- Build label from last.json: `"{app} / {lang or 'auto'} / {translate_model or 'transcribe-only'}"`
- Print: `Last: {label} — Enter to reuse, any key for setup`
- Read single keypress (use `tty`/`termios` raw mode, not `input()`) with 10s timeout
- Enter (or timeout) → return `True`; any other key → `False`; Ctrl+C → raise `KeyboardInterrupt`

**In `main()`:** Before calling `_interactive_setup()`:
```python
last = _load_last_session()
if last.get("app") or last.get("mic"):
    reuse = _quick_rerun_prompt(last)
    if reuse:
        kwargs = {
            "app_name": last.get("app"),
            "mic": last.get("mic", False),
            "device": last.get("device"),
            "lang": last.get("lang"),
            "translate_model": last.get("translate_model"),
            "no_translate": last.get("translate_model") is None,
        }
    else:
        kwargs = _interactive_setup()
else:
    kwargs = _interactive_setup()
```

**Update `_save_last_session`:** Also persist `mic` and `device` keys so mic sessions can rerun too.

**Edge cases:**
- last.json app no longer running → capture will fail at `source.start()` — existing error handling covers this
- Corrupted last.json → `_load_last_session` already returns `{}`, falls through to wizard

---

## 2. Better Exit Summary

### 2a. Track stats in pipeline

**`pipeline.py` → `run_pipeline`:** Already tracks `windows_processed`, `processing_time_total`, `start_time`. Add a `call_count` tracker to `ContinuationTranslator` (or count externally).

Add to translator (`translate.py`): `self.call_count: int = 0` — increment in `translate()`. *(Check if already there — if not, add.)*

Stats available at shutdown:
- `duration` = `time.time() - start_time`
- `words_translated` = `len(translator.confirmed_translation.split())` if translator else 0
- `sentences` = count sentence-enders (`.!?。！？`) in confirmed_translation
- `llm_calls` = `translator.call_count`
- `cost` = `translator.estimated_cost()`
- `confirmed_source` length for transcript content check

### 2b. New display method

**`display.py` → `TerminalDisplay.finish()`:** Replace current minimal finish with:

```python
def finish(self, *, duration: float = 0, words: int = 0, sentences: int = 0,
           llm_calls: int = 0, cost: float = 0.0, has_content: bool = False):
```

Output a box using Unicode box-drawing (`╭─╮│╰─╯`):
```
╭─ session summary ────────────────────╮
│  Duration     4:32                   │
│  Words        287                    │
│  Sentences    24                     │
│  LLM calls    18 · $0.127           │
╰──────────────────────────────────────╯
```

Width = min(40, terminal width - 2). Right-pad values to align. Use `BOLD` for header, `DIM` for box chrome.

### 2c. Transcript save prompt

**`pipeline.py` → `run_pipeline` finally block**, after `display.finish(...)`:

```python
if confirmed_source.strip() and not plain:
    _offer_transcript_save(full_final_text, start_time)
```

New function in `__main__.py` (or `pipeline.py`):
```python
def _offer_transcript_save(text: str, start_time: float):
    try:
        ans = input("Save transcript? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if ans in ("", "y", "yes"):
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(start_time))
        path = Path.home() / "Desktop" / f"langlistn-{ts}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"  Saved → {path}")
```

Put this in `pipeline.py` since it has access to the final text. Import `Path` and `time` (already imported).

### 2d. Wire it up

In `pipeline.py` finally block, replace:
```python
cost = translator.estimated_cost() if translator else 0.0
display.finish(cost)
```
With:
```python
duration = time.time() - start_time
cost = translator.estimated_cost() if translator else 0.0
words = len(locked.split()) if 'locked' in dir() else 0  # use final locked text
sentences = sum(locked.count(p) for p in '.!?。！？') if 'locked' in dir() else 0
llm_calls = translator.call_count if translator else 0
display.finish(
    duration=duration, words=words, sentences=sentences,
    llm_calls=llm_calls, cost=cost,
    has_content=bool(confirmed_source.strip()),
)
# transcript save
final_text = ...  # the full_final or confirmed_source text already computed above
if confirmed_source.strip() and not plain:
    _offer_transcript_save(final_text, start_time)
```

**Note:** Use the `full_final` variable already computed in the finally block (or fall back to `confirmed_source`).

---

## 3. Nicer Header

**`display.py` → `print_header()`:** Replace current 2-line header.

New design (3 lines max):
```
 ┌ langlistn ─────────────────────────┐
 │ Korean → English · Chrome · haiku  │  Ctrl+C to stop
 └────────────────────────────────────┘
```

- Line 1: `BOLD` app name in top border
- Line 2: config summary + right-aligned `Ctrl+C to stop` hint (use `DIM` for hint)
- Line 3: bottom border
- Blank line after

Implementation:
```python
def print_header(self, mode, lang, model, translate_model=None):
    if self.plain:
        return
    w = self.width
    lang_display = f"{lang} → English" if lang else "auto → English"
    model_short = model.split("/")[-1] if "/" in model else model
    t_tag = f" · {translate_model}" if translate_model else ""
    info = f"{lang_display} · {mode} · {model_short}{t_tag}"
    hint = "Ctrl+C to stop"

    # Top border
    title = " langlistn "
    pad = w - len(title) - 3
    sys.stdout.write(f" ┌{BOLD}{title}{RESET}{'─' * max(pad, 0)}┐\n")
    # Info line
    inner_w = w - 4  # account for " │ " and " │"
    right = f"  {DIM}{hint}{RESET}"
    info_padded = info[:inner_w - len(hint) - 4].ljust(inner_w - len(hint) - 2)
    sys.stdout.write(f" │ {info_padded}{right} │\n")
    # Bottom border
    sys.stdout.write(f" └{'─' * (w - 3)}┘\n\n")
    sys.stdout.flush()
```

Adjust padding math to handle narrow terminals gracefully (fallback: drop hint if width < 50).

---

## File Change Summary

| File | Changes |
|---|---|
| `__main__.py` | Add `_quick_rerun_prompt()`, update `main()` interactive branch, persist `mic`/`device` in last.json |
| `display.py` | Rewrite `print_header()` with box design, rewrite `finish()` with summary box + kwargs |
| `pipeline.py` | Compute stats in finally block, pass to `finish()`, add `_offer_transcript_save()` |
| `translate.py` | Add `self.call_count = 0`, increment in `translate()` |

## Order of Implementation

1. `translate.py` — add `call_count` (trivial, no deps)
2. `display.py` — new `print_header()` + `finish()` signatures
3. `pipeline.py` — wire stats + transcript save
4. `__main__.py` — quick rerun prompt + last.json updates
