"""pretty-printing / console-formatting classes and methods."""

# TODO, maybe: much of this is vendored / slightly modified from [redacted]
#  pull it all out into a shared module somewhere...?

import datetime as dt
import logging
import threading
from functools import reduce
from math import floor
from operator import add
from sys import stdout

import rich
from rich.progress import Progress, TextColumn, Task
from rich.spinner import Spinner
from rich.text import Text


GPHOTON_CONSOLE = rich.console.Console()
GPHOTON_PROGRESS = Progress(console=GPHOTON_CONSOLE)


def stamp() -> str:
    return f"{dt.datetime.utcnow().isoformat()[:-7]}: "


def console_and_log(message, level="info", style=None):
    GPHOTON_CONSOLE.print(message, style=style)
    getattr(logging, level)(message)


def mb(b, round_to=2):
    return round(int(b) / 1024 ** 2, round_to)


def render_spinners(spinners, task):
    return reduce(
        add, [spinner.render(task.get_time()) for spinner in spinners]
    )


class SpinTextColumn(TextColumn):
    def __init__(
        self,
        text_format: str,
        spinner_names=None,
        postspinner_names=None,
        style="none",
        speed: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(text_format, style, **kwargs)
        if spinner_names:
            self.spinners = [
                Spinner(spinner_name, style=style, speed=speed)
                for spinner_name in spinner_names
            ]
        else:
            self.spinners = []
        if postspinner_names:
            self.postspinners = [
                Spinner(spinner_name, style=style, speed=speed)
                for spinner_name in postspinner_names
            ]
        else:
            self.postspinners = []

    def render(self, task: Task):
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
        progress=False,
        chunk_size=25,
        file_size=None,
        filename=None,
        log=False,
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

    def _advance(self, n_bytes):
        if self.log is True:
            console_and_log(
                stamp() + f"transferred {mb(n_bytes)}MB", style="blue"
            )
        self.progress_object.advance(self._task_id)

    def __call__(self, bytes_amount):
        with self._lock:
            extra = self._seen_so_far + bytes_amount
            if mb(extra - self._seen_so_far) >= self._chunk_size:
                self._advance(extra)
            self._seen_so_far = extra


def print_inline(text, blanks=60):
    """
    For updating text in place without a carriage return.

    :param text: Message to print to standard out.

    :type text: str

    :param blanks: Number of white spaces to prepend to message.

    :type blanks: int
    """

    stdout.write(" "*blanks+"\r")
    stdout.write(str(str(text)+'\r'))
    stdout.flush()
    return


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
