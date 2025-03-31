"""pretty-printing / console-formatting classes and methods."""

# TODO, maybe: much of this is vendored / slightly modified from [redacted]
#  pull it all out into a shared module somewhere...?

import datetime as dt
import logging
import threading
from math import floor
from sys import stdout
from collections.abc import MutableMapping, Callable, Iterable
from typing import Any
from pathlib import Path

import rich
from cytoolz import first
from dustgoggles.func import zero
from rich.progress import Progress, TextColumn, Task
from rich.spinner import Spinner
from rich.text import Text

from gPhoton.reference import Netstat, Stopwatch

GPHOTON_CONSOLE = rich.console.Console()
GPHOTON_PROGRESS = Progress(console=GPHOTON_CONSOLE)

MEGA_BYTES = 1024 * 1024


def stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S: ")


def console_and_log(
    message: str,
    level: str = "info",
    style: str | None = None,
) -> None:
    GPHOTON_CONSOLE.print(message, style=style)
    getattr(logging, level)(message)


def mb(b: float | str, round_to: int = 2) -> float:
    return round(float(b) / MEGA_BYTES, round_to)


def render_spinners(spinners: Iterable[Spinner], task: Task) -> Text:
    rendered = Text()
    time = task.get_time()
    for spinner in spinners:
        match spinner.render(time):
            case str() as s:
                rendered.append(s)
            case Text() as s:
                rendered.append_text(s)
            case other:
                # spinner.render() can theoretically return an
                # arbitrary RenderableType, which includes things
                # which are neither str nor Text.  As far as I can
                # tell, there is no way to append those things to a
                # Text instance, and the only reason the previous
                # version of this function worked is that, as used
                # in this file, spinner.render() always does return
                # either a str or a Text.
                raise NotImplementedError(
                    f"Don't know how to append {other!r} to a rich.Text"
                )
    return rendered


class SpinTextColumn(TextColumn):
    def __init__(
        self,
        text_format: str,
        spinner_names: Iterable[str] | None = None,
        postspinner_names: Iterable[str] | None = None,
        style: str = "none",
        speed: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_format, style, **kwargs)
        if not spinner_names:
            spinner_names = []
        if not postspinner_names:
            postspinner_names = []
        self.spinners = [
            Spinner(spinner_name, style=style, speed=speed)
            for spinner_name in spinner_names
        ]
        self.postspinners = [
            Spinner(spinner_name, style=style, speed=speed)
            for spinner_name in postspinner_names
        ]

    def render(self, task: Task) -> Text:
        _text = self.text_format.format(task=task)
        if self.markup:
            text = Text.from_markup(
                _text, style=self.style, justify=self.justify
            )
        else:
            text = Text(_text, style=self.style, justify=self.justify)
        if self.highlighter:
            self.highlighter.highlight(text)
        if self.spinners:
            text = render_spinners(self.spinners, task) + text
        if self.postspinners:
            text = text + render_spinners(self.postspinners, task)
        return text


GPHOTON_PROGRESS_SPIN = Progress(
    SpinTextColumn(
        text_format="{task.description}                            ",
        spinner_names=["star"],
    ),
    console=GPHOTON_CONSOLE,
    expand=True,
)


class LogMB:
    def __init__(
        self,
        progress: bool = False,
        chunk_size: int = 25,
        file_size: float | None = None,
        filename: str | Path | None = None,
        log: bool = False,
    ):
        self._seen_so_far = 0
        self._lock = threading.Lock()
        if (progress is True) and (file_size is not None):
            self.progress_object = GPHOTON_PROGRESS
        else:
            self.progress_object = GPHOTON_PROGRESS_SPIN
        self._chunk_size = chunk_size
        description = "downloading"
        if filename is not None:
            description += f" {filename}"
        if file_size is not None:
            self._task_id = self.progress_object.add_task(
                description, total=floor(file_size / chunk_size)
            )
        else:
            self._task_id = self.progress_object.add_task(description)
        self.log = log

    def _advance(self, n_bytes: int) -> None:
        if self.log is True:
            console_and_log(
                stamp() + f"transferred {mb(n_bytes)}MB", style="blue"
            )
        self.progress_object.advance(self._task_id)

    def __call__(self, bytes_amount: int) -> None:
        with self._lock:
            extra = self._seen_so_far + bytes_amount
            if mb(extra - self._seen_so_far) >= self._chunk_size:
                self._advance(extra)
            self._seen_so_far = extra


def print_inline(text: str, blanks: int = 60) -> None:
    """
    For updating text in place without a carriage return.

    :param text: Message to print to standard out.

    :type text: str

    :param blanks: Number of white spaces to prepend to message.

    :type blanks: int
    """

    stdout.write(" "*blanks+"\r")
    stdout.write(text+'\r')
    stdout.flush()


def record_and_yell(message: str, cache: MutableMapping, loud: bool = False):
    """
    place message into a cache object with a timestamp; optionally print it
    """
    if loud is True:
        print(message)
    cache[dt.datetime.now().isoformat()] = message


def notary(cache):

    def note(message, loud: bool = False, eject: bool = False):
        if eject is True:
            return cache
        return record_and_yell(message, cache, loud)
    return note


def print_stats(watch, netstat):
    watch.start(), netstat.update()

    def printer(total=False, eject=False):
        netstat.update()
        if eject is True:
            return watch, netstat
        if total is True:
            text = (
                f"{watch.total()} total s,"
                f"{mb(round(first(netstat.total.values())))} total MB"
            )
        else:
            text = (
                f"{watch.peek()} s,"
                f"{mb(round(first(netstat.interval.values())))} MB"
            )
            watch.click()
        return text
    return printer


def make_monitors(
    fake: bool = False, silent: bool = True, round_to: int | None = None
) -> tuple[Callable, Callable]:
    if fake:
        return zero, zero

    stat = print_stats(Stopwatch(silent=silent, digits=round_to), Netstat())
    note = notary({})
    return stat, note


# class PHOTONLIGHTER(RegexHighlighter):
#     base_style = "GPHOTON."
#     highlights = [
#         r"(?P<missing>(skipping))",
#         r"(?P<prep>(loaded|generated|found|converted))",
#         r"(?P<output>(wrote|completed))",
#         r"(?<=[Z _])(?P<id>[R|L]\d[RGB]?)",
#         r"(?P<id>(zcam|ZCAM)\d\d\d\d\d)",
#         r"(?P<id>(sol|SOL)\d{2,4})",
#         r"(?P<selection>\(\d{1,3}\))",
#         r"(?P<marslab>(-roi.fits)|(-marslab.*csv))",
#     ]
#
#
# PHOTONTHEME = Theme(
#     {
#         "GPHOTON.output": "green1",
#         "GPHOTON.prep": "aquamarine3",
#         "GPHOTON.id": "dark_turquoise",
#         "GPHOTON.selection": "bold",
#         "GPHOTON.missing": "purple4",
#         "GPHOTON.marslab": "italic orchid1",
#         "FORBIDDEN": "hot_pink on black",
#         "FORBIDDEN.warning": "slate_blue1 on black",
#     }
# )
