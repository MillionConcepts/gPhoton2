import requests

def download_with_progress_bar(url, destination):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    filesize_mb = mb(response.headers["Content-Length"])


class LogMB:
    def __init__(self):
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            extra = self._seen_so_far + bytes_amount
            if mb(extra - self._seen_so_far) > 25:
                console_and_log(
                    stamp() + f"transferred {mb(extra)}MB", style="blue"
                )
            self._seen_so_far = extra