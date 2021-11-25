"""
wrapper methods for downloads via `requests`
"""

from math import floor
from pathlib import Path

import requests

from gPhoton.pretty import mb, LogMB


def download_with_progress_bar(url, destination, chunk_size_mb=10):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    if "Content-Length" in response.headers:
        filesize_mb = mb(response.headers["Content-Length"])
    else:
        filesize_mb = None
    log_size = LogMB(
        progress=True,
        file_size=filesize_mb,
        chunk_size=chunk_size_mb,
        filename = Path(destination).name
    )
    content_iterator = response.iter_content(
        chunk_size=floor(chunk_size_mb * (1024**2))
    )
    with log_size.progress_object, open(destination, "wb+") as file:
        for chunk in content_iterator:
            log_size(len(chunk))
            file.write(chunk)
