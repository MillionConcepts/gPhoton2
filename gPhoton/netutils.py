from math import floor
from pathlib import Path

import requests

from gPhoton.pretty import mb, LogMB, print_inline


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


def manage_networked_sql_request(
    query, maxcnt=100, wait=1, timeout=60, verbose=0
):
    """
    Manage calls via `requests` to SQL servers behind HTTP interfaces,
    providing better feedback and making them more robust against network
    errors.

    :param query: The URL containing the query.

    :type query: str

    :param maxcnt: The maximum number of attempts to make before failure.

    :type maxcnt: int

    :param wait: The length of time to wait before attempting the query again.
        Currently a placeholder.

    :type wait: int

    :param timeout: The length of time to wait for the server to send data
        before giving up, specified in seconds.

    :type timeout: float

    :param verbose: If > 0, print additional messages to STDOUT. Higher value
        represents more verbosity.

    :type verbose: int

    :returns: requests.Response or None -- The response from the server. If the
        query does not receive a response, returns None.
    """

    # Keep track of the number of failures.
    cnt = 0

    # This will keep track of whether we've gotten at least one
    # successful response.
    successful_response = False

    while cnt < maxcnt:
        try:
            r = requests.get(query, timeout=timeout)
            successful_response = True
        except requests.exceptions.ConnectTimeout:
            if verbose:
                print("Connection time out.")
            cnt += 1
            continue
        except requests.exceptions.ConnectionError:
            if verbose:
                print("Domain does not resolve.")
            cnt += 1
            continue
        except Exception as ex:
            if verbose:
                print(f'bad query? {query}: {type(ex)}: {ex}')
            cnt += 1
            continue
        if r.json()['status'] == 'EXECUTING':
            if verbose > 1:
                print_inline('EXECUTING')
            cnt = 0
            continue
        elif r.json()['status'] == 'COMPLETE':
            if verbose > 1:
                print_inline('COMPLETE')
            break
        elif r.json()['status'] == 'ERROR':
            print('ERROR')
            print('Unsuccessful query: {q}'.format(q=query))
            raise ValueError(r.json()['msg'])
        else:
            print('Unknown return: {s}'.format(s=r.json()['status']))
            cnt += 1
            continue

    if not successful_response:
        # Initiate an empty response object in case
        # the try statement is never executed.
        # r = requests.Response()
        r = None

    return r