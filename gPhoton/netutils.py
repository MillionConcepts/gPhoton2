import requests

from gPhoton.pretty import mb, LogMB


def download_with_progress_bar(url, destination):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    filesize_mb = mb(response.headers["Content-Length"])
    size_logger = LogMB(progress=True, filesize=filesize_mb)
